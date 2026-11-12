"""06번 Colab 환경을 준비하고 같은 커널의 준비 결과를 재사용한다."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


PACKAGES = ('numpy', 'psutil', 'nbformat', 'datasets', 'evalplus', 'pynvml')


def _gpu():
    from tools.native.gpu_memory import NvmlDevice
    device = NvmlDevice()
    try:
        major, minor = device.nvml.nvmlDeviceGetCudaComputeCapability(device.handle)
        return {'name': device.name, 'uuid': device.uuid, 'target': f'{major}{minor}'}
    finally:
        device.close()


def _reusable(root, state):
    if not isinstance(state, dict):
        return False
    engine = root / '.cache/llama-build/bin/llama-expert-research'
    grader_python = root / '.cache/putare-lcb/venv/bin/python'
    if (state.get('root') != str(root) or state.get('python') != sys.executable
            or not engine.is_file() or not grader_python.is_file()
            or not all(shutil.which(name) for name in ('git', 'cmake', 'ninja', 'nvcc', 'c++'))
            or not all(importlib.util.find_spec(name) is not None for name in PACKAGES)):
        return False
    from tools.colab_archive import sha256_file
    if sha256_file(engine) != state.get('engine_sha256'):
        return False
    current = _gpu()
    return current['uuid'] == state.get('gpu_uuid') and current['target'] == state.get('gpu_target')


def prepare_environment(root, state=None):
    """환경을 준비하고 현재 커널에서만 보관할 재사용 상태를 돌려준다."""
    # 2026-11-09: 각 버전별 엔진과 채점 환경을 확인한 경우에만 반복 설치를 생략한다.
    root = Path(root).resolve()
    if not (3, 12) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError('Python 3.12가 필요합니다. Colab 런타임 버전을 확인하세요.')
    if _reusable(root, state):
        print('현재 커널의 준비된 환경을 재사용합니다.')
        return state
    missing = [name for name in ('git', 'cmake', 'nvcc', 'c++') if not shutil.which(name)]
    if missing:
        raise RuntimeError('필요한 빌드 도구가 없습니다: ' + ', '.join(missing) + '. Colab에 git, cmake, C++ 컴파일러와 CUDA Toolkit을 설치하세요.')
    if not shutil.which('ninja'):
        print('ninja가 없어 설치합니다.')
        subprocess.run([sys.executable, '-m', 'pip', '--isolated', 'install', 'ninja'], check=True)
        if not shutil.which('ninja'):
            raise RuntimeError('ninja 설치 후에도 실행 파일을 찾지 못했습니다. 커널 PATH를 확인하세요.')
    subprocess.run([sys.executable, '-m', 'pip', '--isolated', 'install', '-r', str(root / 'requirements-research.txt')], check=True)
    gpu = _gpu()
    print(f"GPU: {gpu['name']}, Compute Capability: {gpu['target']}")
    codes = subprocess.run(['nvcc', '--list-gpu-code'], check=True, capture_output=True, text=True).stdout.split()
    if f"sm_{gpu['target']}" not in codes:
        raise RuntimeError(f"현재 nvcc가 sm_{gpu['target']}을 지원하지 않습니다. GPU를 지원하는 CUDA Toolkit이 필요합니다.")
    subprocess.run([sys.executable, str(root / 'tools/native/prepare.py'), '--backend', 'cuda', '--gpu-target', gpu['target'], '--execute'], cwd=root, check=True)
    from moe_prune.benchmark_data import prepare_grader
    prepare_grader(root)
    from tools.colab_archive import sha256_file
    engine = root / '.cache/llama-build/bin/llama-expert-research'
    return {'root': str(root), 'python': sys.executable, 'gpu_uuid': gpu['uuid'],
            'gpu_target': gpu['target'], 'engine_sha256': sha256_file(engine)}
