"""2026-11-07: 02 벤치마크는 모델을 유지하면서 문제마다 독립 세션 상태를 사용한다."""
import json
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
import tempfile

import psutil
from tools.native.gpu_memory import ProcessGpuMemorySampler

from .research import Experiment, spawn_native, write_json
from .memory_safety import MemoryGuard
from .execution_safety import execution_guard, poison_execution


class SessionCleanupError(RuntimeError):
    """세션 종료 확인 실패는 메모리 부족 재시도로 취급하지 않는다."""


class ResidentSession:
    def __init__(self, engine, config, request, stderr=None):
        self.lease = None
        self.lease_acquired = False
        self.guard = None
        self.gpu_sampler = None
        self.sample_lock = threading.Lock()
        self.process = None
        self.stop = threading.Event()
        self.responses = queue.Queue()
        self.failure = None
        self.peak_rss = None
        self.problem_peak_rss = None
        self.monitor = None
        self.reader = None
        self.preflight = None
        self.closed = False
        try:
            self.lease = execution_guard('native')
            self.lease.__enter__()
            self.lease_acquired = True
            self.guard = MemoryGuard(config, request)
            self.preflight = self.guard.preflight()
            self.process = spawn_native([str(engine), '--session'], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=stderr,
                                        text=True, encoding='utf-8', bufsize=1)
            self.gpu_sampler = ProcessGpuMemorySampler(self.process.pid)
            self.reader = threading.Thread(target=self._read_responses, daemon=True)
            self.reader.start()
            self.monitor = threading.Thread(target=self._monitor, daemon=True)
            self.monitor.start()
        except BaseException:
            self.close()
            raise

    def _read_responses(self):
        try:
            for line in self.process.stdout:
                self.responses.put(line)
        finally:
            self.responses.put(None)

    def _monitor(self):
        try:
            if self.process.poll() is not None:
                return
            process = psutil.Process(self.process.pid)
            while not self.stop.wait(.1):
                if self.process.poll() is not None:
                    return
                self.guard.check()
                try:
                    total = sum(member.memory_info().rss for member in [process, *process.children(recursive=True)])
                except psutil.NoSuchProcess:
                    if self.process.poll() is not None:
                        return
                    raise
                with self.sample_lock:
                    self.gpu_sampler.sample()
                self.peak_rss = max(self.peak_rss or 0, total)
                self.problem_peak_rss = max(self.problem_peak_rss or 0, total)
        except BaseException as error:
            self.failure = error
            try:
                self.process.kill()
            except Exception:
                pass

    def request(self, request_path, output_path):
        if self.failure:
            raise self.failure
        self.problem_peak_rss = None
        self.process.stdin.write(json.dumps({'request': str(Path(request_path).resolve()),
                                             'output': str(Path(output_path).resolve())}) + '\n')
        self.process.stdin.flush()
        while True:
            if self.failure:
                raise self.failure
            try:
                line = self.responses.get(timeout=.1)
            except queue.Empty:
                if self.process.poll() is not None:
                    raise RuntimeError('native session이 응답 전에 종료되었습니다')
                continue
            if line is None:
                raise RuntimeError('native session 응답 스트림이 종료되었습니다')
            try:
                response = json.loads(line)
            except ValueError as error:
                raise RuntimeError('native session 응답이 올바르지 않습니다') from error
            if response != {'status': 'ok'}:
                raise RuntimeError('native session이 실패를 반환했습니다')
            if self.failure:
                raise self.failure
            with self.sample_lock:
                gpu = self.gpu_sampler.snapshot()
            return {'peak_process_tree_rss_bytes': self.problem_peak_rss,
                    'session_peak_process_tree_rss_bytes': self.peak_rss,
                    'os_gpu_memory': {**gpu, 'scope': 'session_to_date'},
                    'memory_preflight': self.preflight, 'memory_last': dict(self.guard.last)}

    def close(self):
        if self.closed:
            return
        self.closed = True
        failure = None
        try:
            if self.process:
                try:
                    self.process.stdin.close()
                    self.process.wait(timeout=15)
                except BaseException as error:
                    try:
                        self.process.kill()
                        self.process.wait(timeout=15)
                    except BaseException as cleanup_error:
                        poison_execution(f'종료 확인 실패: {type(cleanup_error).__name__}')
                        failure = SessionCleanupError('native session 종료를 확인할 수 없습니다')
                    if failure is None and not isinstance(error, (BrokenPipeError, subprocess.TimeoutExpired)):
                        failure = error
                if self.process.poll() is None:
                    poison_execution('native session wait 후에도 실행 중')
                    failure = SessionCleanupError('native session 종료를 확인할 수 없습니다')
        finally:
            self.stop.set()
            if self.monitor:
                self.monitor.join(timeout=5)
                if self.monitor.is_alive():
                    poison_execution('memory monitor 종료 확인 실패')
                    failure = SessionCleanupError('메모리 감시 종료를 확인할 수 없습니다')
            if self.reader:
                self.reader.join(timeout=5)
                if self.reader.is_alive():
                    poison_execution('native 응답 reader 종료 확인 실패')
                    failure = SessionCleanupError('native 응답 reader 종료를 확인할 수 없습니다')
            if self.process and self.process.stdout and (not self.reader or not self.reader.is_alive()):
                self.process.stdout.close()
            try:
                if self.gpu_sampler and (not self.monitor or not self.monitor.is_alive()):
                    self.gpu_sampler.close()
                if self.guard and (not self.monitor or not self.monitor.is_alive()):
                    self.guard.close()
            finally:
                if self.lease_acquired:
                    self.lease.__exit__(None, None, None)
        if failure:
            raise failure


class ResidentExperiment:
    """기존 job 형식을 유지하며 필요한 최초 문제에서만 엔진을 시작한다."""
    def __init__(self, root, config):
        self.experiment = Experiment(root, config)
        self.root, self.config = Path(root), config
        self.session = None
        self.placement = None
        self.log = None
        self.log_path = None
        self.session_artifacts = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        health_error = self.session.failure if self.session else None
        try:
            self.close()
        except BaseException as close_error:
            if exc is not None:
                exc.add_note(f'상주 세션 정리 실패: {close_error}')
            else:
                raise
        if exc is None and health_error is not None:
            raise health_error
        return False

    def check(self):
        if self.session and self.session.failure:
            raise self.session.failure

    def close(self):
        try:
            if self.session:
                self.session.close()
        finally:
            self.session = None
            self.placement = None
            self.session_artifacts = None
            if self.log:
                self.log.close()
                self.log = None

    def run(self, name, request):
        from .gguf import sha256
        folder = self.root / 'jobs' / name
        request = json.loads(json.dumps(request, ensure_ascii=False, allow_nan=False))
        executable = Path(self.config['engine'])
        identity = {}
        for role, path in (('engine', executable), ('model', Path(request['model']))):
            stat = path.stat()
            key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            if key not in self.experiment._identities:
                self.experiment._identities[key] = {'path': key[0], 'bytes': key[1],
                                                     'mtime_ns': key[2], 'sha256': sha256(path)}
            identity[role] = self.experiment._identities[key]
        if self.session and identity != self.session_artifacts:
            raise ValueError('상주 세션 실행 중 모델 또는 엔진 파일이 변경되었습니다')
        if folder.exists():
            if (json.loads((folder / 'request.json').read_text(encoding='utf-8')) != request or
                    json.loads((folder / 'identity.json').read_text(encoding='utf-8')) != identity):
                raise ValueError('기존 job의 설정 또는 모델/엔진 해시가 다릅니다')
            if (folder / 'measurement.json').exists():
                return Experiment.result(folder)
            raise RuntimeError(f'미완료 job입니다: {folder}')
        placement = (request['n_gpu_layers'], request['offload_kqv'])
        if self.session and placement != self.placement:
            self.close()
        folder.mkdir(parents=True)
        write_json(folder / 'request.json', request)
        write_json(folder / 'identity.json', identity)
        self.experiment.manifest.setdefault('artifacts', {}).update({item['path']: item for item in identity.values()})
        write_json(self.root / 'manifest.json', self.experiment.manifest)
        start = time.perf_counter()
        start_offset = 0
        try:
            if self.session is None:
                sessions = self.root / 'sessions'
                sessions.mkdir(parents=True, exist_ok=True)
                fd, name = tempfile.mkstemp(prefix='native-', suffix='.log', dir=sessions)
                self.log_path = Path(name)
                self.log = os.fdopen(fd, 'wb')
                self.session = ResidentSession(executable, self.config, request, stderr=self.log)
                self.placement = placement
                self.session_artifacts = identity
            else:
                start_offset = self.log_path.stat().st_size
            measurement = self.session.request(folder / 'request.json', folder / 'events.jsonl')
            self.log.flush()
            # 각 결과의 실제 설정을 검증할 수 있도록 세션의 초기화 로그도 보존한다.
            shutil.copyfile(self.log_path, folder / 'stderr.log')
            measurement['wall_s'] = time.perf_counter() - start
            measurement['session_log'] = str(self.log_path.relative_to(self.root))
            measurement['session_log_offset'] = start_offset
            write_json(folder / 'measurement.json', measurement)
            return Experiment.result(folder)
        except BaseException as error:
            if self.log_path and self.log_path.exists():
                if self.log:
                    self.log.flush()
                shutil.copyfile(self.log_path, folder / 'stderr.log')
                with self.log_path.open('rb') as source, (folder / 'stderr-failure.log').open('wb') as dest:
                    source.seek(start_offset)
                    shutil.copyfileobj(source, dest)
            # 반드시 종료를 확인하고 난 다음에만 외부 호출자가 OOM 재시도를 할 수 있다.
            try:
                self.close()
            except BaseException as close_error:
                raise SessionCleanupError('실패 후 native session 종료를 확인할 수 없습니다') from close_error
            raise error
