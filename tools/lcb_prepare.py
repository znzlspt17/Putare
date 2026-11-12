"""Colab Linux에서 고정 공식 채점 코드와 전용 가상환경을 준비한다."""
from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

COMMIT = '28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24'
ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / '.cache/putare-lcb/venv'
SOURCE = ROOT / '.cache/putare-lcb/livecodebench' / COMMIT


def _ensure_pip(venv_python, env):
    # 2026-11-09: 부분 생성된 venv는 보존하고 pip만 현재 Python으로 복구한다.
    probe_command = [str(venv_python), '-I', '-m', 'pip', '--version']
    probe = subprocess.run(probe_command, capture_output=True, text=True, errors='replace', env=env)
    if probe.returncode == 0:
        return
    message = (probe.stderr or '').strip()
    if not message.endswith('No module named pip'):
        raise RuntimeError('채점기 Python의 pip 확인 실패: ' + (message[-300:] or f'종료 코드 {probe.returncode}'))
    command = [sys.executable, '-I', '-m', 'pip', '--isolated', '--python', str(venv_python),
               'install', '--disable-pip-version-check', '--no-input', 'pip']
    result = subprocess.run(command, capture_output=True, text=True, errors='replace', env=env)
    if result.returncode != 0:
        detail = (result.stderr or '').strip()[-300:]
        raise RuntimeError('채점기 pip 복구 실패: ' + (detail or f'종료 코드 {result.returncode}'))
    verified = subprocess.run(probe_command, capture_output=True, text=True, errors='replace', env=env)
    if verified.returncode != 0:
        detail = (verified.stderr or '').strip()[-300:]
        raise RuntimeError('채점기 pip 복구 후 확인 실패: ' + (detail or f'종료 코드 {verified.returncode}'))


def prepare(root=ROOT):
    root = Path(root).resolve()
    venv = root / '.cache/putare-lcb/venv'
    source = root / '.cache/putare-lcb/livecodebench' / COMMIT
    if os.name != 'posix' or not Path('/proc/version').is_file():
        raise RuntimeError('Colab Linux 환경에서 실행하세요')
    from moe_prune.grading import worker_environment
    env = worker_environment()
    if not (venv / 'bin/python').is_file():
        venv.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(venv)], check=True, env=env)
    _ensure_pip(venv / 'bin/python', env)
    # --isolated는 사용자 pip 설정과 인증 환경변수의 묵시적 사용을 배제한다.
    subprocess.run([str(venv / 'bin/python'), '-I', '-m', 'pip', '--isolated', 'install',
                    '--disable-pip-version-check', '--no-input',
                    'numpy>=2,<3', 'tqdm>=4,<5', 'datasets>=3.2,<4'], check=True, env=env)
    if not source.exists():
        url = f'https://github.com/LiveCodeBench/LiveCodeBench/archive/{COMMIT}.tar.gz'
        request = urllib.request.Request(url, headers={'User-Agent': 'putare-benchmark/1'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=90) as response:
            archive = response.read()
        source.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=source.parent) as tmp:
            with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
                tar.extractall(tmp, filter='data')
            matches = list(Path(tmp).glob('LiveCodeBench-*'))
            if len(matches) != 1 or not (matches[0] / 'lcb_runner/evaluation/testing_util.py').is_file():
                raise RuntimeError('고정 commit 아카이브 형상이 다릅니다')
            matches[0].rename(source)
    if not (source / 'lcb_runner/evaluation/testing_util.py').is_file():
        raise RuntimeError('공식 source 폴더가 불완전합니다')
    print(f'공식 LCB commit {COMMIT}, venv {venv}, source {source}')
    return {'venv': str(venv), 'source': str(source)}


if __name__ == '__main__':
    prepare()
