"""Colab에서 공식 채점기를 별도 프로세스로 실행한다. 보안 격리는 제공하지 않는다."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time

import psutil


class GradingCleanupError(RuntimeError):
    """채점 프로세스 종료를 확인하지 못한 경우 후속 작업을 중단한다."""


# 2026-11-09: 문제별 worker가 전체 채점에 할당된 하나의 시간 예산을 공유한다.
class GradeDeadline:
    def __init__(self, timeout_s=7200, clock=None):
        if timeout_s <= 0:
            raise ValueError('채점 제한 시간은 양수여야 합니다')
        self.clock = clock or time.monotonic
        self.deadline = self.clock() + timeout_s

    def remaining(self):
        value = self.deadline - self.clock()
        if value <= 0:
            raise TimeoutError('전체 채점 제한 시간을 초과했습니다')
        return value


def identity():
    return {'execution_mode': 'unisolated', 'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


# 2026-11-09: 생성 코드를 커널 밖에서 실행하되 자식에게 인증 환경을 전달하지 않는다.
def worker_environment():
    return {'PATH': '/usr/bin:/bin', 'HOME': '/tmp', 'TMPDIR': '/tmp', 'LANG': 'C.UTF-8',
            'LC_ALL': 'C.UTF-8', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
            'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
            'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}


# 2026-11-09: 비격리 worker가 만든 경로를 따라 다른 파일을 읽거나 과대 결과를 보관하지 않는다.
def read_result(path, maximum_bytes=8 * 1024**2):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('채점 결과가 symlink입니다')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0))
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum_bytes:
            raise ValueError('채점 결과 크기 또는 파일 형식이 잘못되었습니다')
        data = stream.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise ValueError('채점 결과 크기 초과')
    return data.decode('utf-8')


def _group_members(group):
    members = []
    for member in psutil.process_iter(['pid', 'status']):
        try:
            if member.pid != os.getpid() and os.getpgid(member.pid) == group and member.info['status'] != psutil.STATUS_ZOMBIE:
                members.append(member)
        except (psutil.NoSuchProcess, ProcessLookupError):
            continue
    return members


def _cleanup(process, known):
    """이번 실행에서 추적한 프로세스만 종료하고 종료 여부를 확인한다."""
    if os.name == 'posix':
        # worker가 먼저 끝나도 같은 세션의 자식은 남을 수 있다.
        if _group_members(process.pid):
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    elif process.poll() is None:
        process.terminate()
    for child in known.values():
        try:
            if child.is_running():
                child.terminate()
        except psutil.NoSuchProcess:
            pass
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name == 'posix':
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        else:
            process.kill()
        process.wait(timeout=3)
    _, alive = psutil.wait_procs(list(known.values()), timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=3)
    if os.name == 'posix' and _group_members(process.pid):
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        deadline = time.monotonic() + 3
        while _group_members(process.pid) and time.monotonic() < deadline:
            time.sleep(.05)
        if _group_members(process.pid):
            raise RuntimeError('채점 프로세스 그룹의 종료를 확인하지 못했습니다')
    if alive or process.poll() is None:
        raise RuntimeError('채점 프로세스의 종료를 확인하지 못했습니다')


# 2026-11-09: 시간 초과와 중단 시 이번 worker만 정리하고, 성공한 결과만 새 파일에 기록한다.
def run(source, venv, input_path, output_path, adapter=None, timeout_s=7200, memory_config=None, *, guard=None):
    from .execution_safety import execution_guard, poison_execution

    if timeout_s <= 0:
        raise ValueError('채점 제한 시간은 양수여야 합니다')
    source, venv, input_path = (Path(p).resolve(strict=True) for p in (source, venv, input_path))
    adapter = Path(adapter or Path(__file__).resolve().parents[1] / 'tools/lcb_grade.py').resolve(strict=True)
    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('기존 채점 결과를 덮어쓰지 않습니다')
    expected = json.loads(input_path.read_text(encoding='utf-8'))
    if not isinstance(expected, dict) or not isinstance(expected.get('id'), str):
        raise ValueError('채점 입력 ID가 없습니다')
    python = venv / 'bin/python'
    if not python.is_file():
        raise FileNotFoundError('채점 환경 Python이 없습니다')
    scope = nullcontext(guard) if guard is not None else execution_guard('grading', memory_config=memory_config)
    with scope as guard:
        if guard is None:
            raise RuntimeError('채점 메모리 감시를 사용할 수 없습니다')
        try:
            with tempfile.TemporaryDirectory(prefix='putare-grade-') as temporary:
                result_path = Path(temporary) / 'result.json'
                command = [str(python), '-I', '-B', str(adapter), '--worker', '--official-source', str(source),
                           '--input', str(input_path), '--output', str(result_path)]
                process = subprocess.Popen(command, env=worker_environment(), stdin=subprocess.DEVNULL,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                           close_fds=True, start_new_session=(os.name == 'posix'))
                known = {}
                try:
                    deadline = time.monotonic() + timeout_s
                    while process.poll() is None:
                        if time.monotonic() >= deadline:
                            raise TimeoutError('채점 제한 시간을 초과했습니다')
                        guard.check()
                        try:
                            for child in psutil.Process(process.pid).children(recursive=True):
                                known[(child.pid, child.create_time())] = child
                        except psutil.NoSuchProcess:
                            pass
                        time.sleep(.1)
                    if process.returncode != 0:
                        raise RuntimeError(f'채점 worker 실패: returncode={process.returncode}')
                    if not result_path.is_file():
                        raise RuntimeError('채점 결과가 없습니다')
                    result = json.loads(read_result(result_path))
                    if (not isinstance(result, dict) or result.get('id') != expected['id'] or
                            type(result.get('passed')) is not bool or result.get('pass_at_1') != float(result['passed'])):
                        raise ValueError('채점 결과 ID 또는 판정이 잘못되었습니다')
                    result['_execution'] = {'mode': 'unisolated', 'memory_last': dict(getattr(guard, 'last', {})),
                                            'warning_events': list(getattr(guard, 'warning_events', []))}
                    payload = json.dumps(result, ensure_ascii=False, allow_nan=False)
                finally:
                    try:
                        _cleanup(process, known)
                    except BaseException as error:
                        poison_execution('채점 프로세스 종료 확인 실패')
                        raise GradingCleanupError('채점 프로세스 종료를 확인하지 못했습니다') from error
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open('x', encoding='utf-8') as stream:
                    stream.write(payload)
                return result
        finally:
            pass
