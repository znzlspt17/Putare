"""고정 revision 연구 엔진의 Colab Linux CUDA 빌드를 준비한다."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

COMMIT = "10d4846a572aebf21c0482d140b00e74c0f1ad20"
URL = "https://github.com/ggml-org/llama.cpp.git"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def preflight():
    return {name: shutil.which(name) for name in ("git", "cmake", "ninja", "nvcc", "c++")}


# 2026-11-09: 그래프 계측을 제공하는 기존 연구 실행 파일을 동일 revision의 CUDA backend로 빌드한다.
def build_commands(source: Path, build: Path, gpu_target: str) -> list[list[str]]:
    target = str(gpu_target).replace('.', '')
    if not target.isdecimal() or len(target) not in (2, 3):
        raise ValueError('CUDA compute capability 숫자가 필요합니다')
    return [["cmake", "-S", str(Path(__file__).resolve().parent), "-B", str(build),
             "-G", "Ninja", f"-DLLAMA_SOURCE_DIR={source}", "-DGGML_CUDA=ON",
             "-DGGML_HIP=OFF", f"-DCMAKE_CUDA_ARCHITECTURES={target}",
             "-DCMAKE_BUILD_TYPE=Release"],
            ["cmake", "--build", str(build), "--target", "llama-expert-research", "--parallel", "2"]]


def detected_target():
    from tools.native.gpu_memory import NvmlDevice
    device = NvmlDevice()
    try:
        major, minor = device.nvml.nvmlDeviceGetCudaComputeCapability(device.handle)
        return f'{major}{minor}'
    finally:
        device.close()


def _git_environment():
    # 2026-11-09: 공개 checkout이 사용자 credential helper나 인증 환경을 사용하지 않도록 제한한다.
    result = {name: value for name in ('PATH', 'LANG', 'LC_ALL')
              if (value := os.environ.get(name)) is not None}
    result.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
                  GIT_TERMINAL_PROMPT='0', GIT_ASKPASS='/bin/false')
    return result


def _build_environment():
    names = ('PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'LD_LIBRARY_PATH',
             'CUDA_HOME', 'CUDA_PATH', 'CUDACXX', 'CUDA_VISIBLE_DEVICES')
    return {name: value for name in names if (value := os.environ.get(name)) is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".cache/llama.cpp")
    parser.add_argument("--build", type=Path, default=ROOT / ".cache/llama-build")
    parser.add_argument("--backend", choices=['cuda'], default='cuda')
    parser.add_argument("--gpu-target", default="")
    parser.add_argument("--execute", action="store_true", help="직접 다운로드·빌드를 실행")
    args = parser.parse_args()
    if sys.platform != 'linux' and args.execute:
        raise SystemExit('Linux CUDA 환경에서 실행하세요')
    tools = preflight()
    print(json.dumps({"commit": COMMIT, "tools": tools, "python": sys.version.split()[0]}, ensure_ascii=False, indent=2))
    target = args.gpu_target.replace('.', '') if args.gpu_target else detected_target()
    source, build = args.source.resolve(), args.build.resolve()
    commands = build_commands(source, build, target)
    for command in commands:
        print(shlex.join(command))
    if not args.execute:
        return
    if not (3, 12) <= sys.version_info[:2] < (3, 13):
        raise SystemExit('프로젝트는 Python 3.12가 필요합니다')
    missing = [name for name, value in tools.items() if value is None]
    if missing:
        raise SystemExit(f"빌드 도구 없음: {', '.join(missing)}")
    if detected_target() != target:
        raise SystemExit('CUDA target과 실제 GPU 장치가 다릅니다')
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", URL, str(source)], check=True, env=_git_environment())
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", COMMIT], check=True, env=_git_environment())
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True, env=_git_environment()).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True, env=_git_environment()).strip()
    if actual != COMMIT or dirty:
        raise SystemExit("upstream revision 또는 작업 트리가 다릅니다. 기존 파일을 자동 수정하지 않습니다")
    for command in commands:
        subprocess.run(command, check=True, env=_build_environment())


if __name__ == "__main__":
    main()
