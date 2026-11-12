"""Colab에서 추론·채점의 동시 실행을 차단하고 종료 실패를 보존한다."""
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile

import psutil

_LOCK = Path(tempfile.gettempdir()) / 'wemakemodel-inference-wsl.lock'
_POISON = _LOCK.with_suffix('.unsafe')
_MODEL_NAMES = {'llama-expert-research', 'llama-expert-re', 'llama-server', 'llama-cli'}


def poison_execution(reason):
    _POISON.write_text(reason, encoding='utf-8')


def running_inference():
    found = []
    for process in psutil.process_iter(['pid', 'name']):
        try:
            name = process.info['name']
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            continue
        except psutil.AccessDenied as error:
            raise RuntimeError('프로세스 이름을 확인할 수 없습니다') from error
        if name is None:
            raise RuntimeError('프로세스 이름을 확인할 수 없습니다')
        if name.lower() in _MODEL_NAMES:
            found.append(name)
    return found


def check_released_memory(config=None):
    from .memory_safety import MemoryGuard
    guard = MemoryGuard({} if config is None else config, {'monitor_gpu': True})
    try:
        return guard.preflight()
    finally:
        guard.close()


# 2026-11-09: 같은 런타임의 독립 커널에서도 추론과 채점이 겹치지 않도록 잠근다.
@contextmanager
def execution_guard(kind, *, memory_config=None, monitor_gpu=True):
    if kind not in ('native', 'grading'):
        raise ValueError('unknown execution kind')
    import fcntl
    _LOCK.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK.open('a+b') as lock:
        monitor = None
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('다른 추론 또는 채점 작업이 실행 중입니다') from error
        try:
            if _POISON.exists():
                raise RuntimeError(f'이전 프로세스의 종료를 확인하지 못했습니다. 안전 상태 확인 후 {_POISON} 파일을 직접 제거하세요')
            if kind == 'grading':
                active = running_inference()
                if active:
                    raise RuntimeError(f"모델 프로세스가 실행 중이므로 채점하지 않습니다: {', '.join(active)}")
                from .memory_safety import MemoryGuard
                # 2026-11-11: 보존된 답안의 독립 채점은 GPU 없이도 RAM과 프로세스 잠금을 유지한다.
                monitor = MemoryGuard({} if memory_config is None else memory_config, {'monitor_gpu': monitor_gpu})
                monitor.preflight()
            yield monitor
        finally:
            try:
                if monitor is not None:
                    active_error = sys.exception()
                    try:
                        monitor.close()
                    except Exception as error:
                        if active_error is None:
                            raise
                        active_error.add_note(f'메모리 감시 종료 실패: {type(error).__name__}')
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
