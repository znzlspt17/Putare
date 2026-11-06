"""2026-11-06: WSL 전용 가상환경과 고정 공식 채점 코드를 준비한다."""
from __future__ import annotations

import io
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import urllib.request

COMMIT = '28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24'
VENV = Path('/home/vivy/.venvs/wemakemodel-lcb')
SOURCE = Path('/home/vivy/.cache/wemakemodel/livecodebench') / COMMIT


def prepare():
    if os.name != 'posix' or not Path('/proc/version').is_file():
        raise RuntimeError('Ubuntu WSL 환경에서 실행하세요')
    if not (VENV / 'bin/python').is_file():
        VENV.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['python3', '-m', 'venv', str(VENV)], check=True)
    # --isolated는 사용자 pip 설정과 인증 환경변수의 묵시적 사용을 배제한다.
    subprocess.run([str(VENV / 'bin/python'), '-I', '-m', 'pip', '--isolated', 'install',
                    '--disable-pip-version-check', '--no-input',
                    'numpy>=2,<3', 'tqdm>=4,<5', 'datasets>=3.2,<4'], check=True)
    if not SOURCE.exists():
        url = f'https://github.com/LiveCodeBench/LiveCodeBench/archive/{COMMIT}.tar.gz'
        request = urllib.request.Request(url, headers={'User-Agent': 'wemakemodel-benchmark/1'})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=90) as response:
            archive = response.read()
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SOURCE.parent) as tmp:
            with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
                tar.extractall(tmp, filter='data')
            matches = list(Path(tmp).glob('LiveCodeBench-*'))
            if len(matches) != 1 or not (matches[0] / 'lcb_runner/evaluation/testing_util.py').is_file():
                raise RuntimeError('고정 commit 아카이브 형상이 다릅니다')
            matches[0].rename(SOURCE)
    if not (SOURCE / 'lcb_runner/evaluation/testing_util.py').is_file():
        raise RuntimeError('공식 source 폴더가 불완전합니다')
    print(f'공식 LCB commit {COMMIT}, venv {VENV}, source {SOURCE}')


if __name__ == '__main__':
    prepare()
