"""2026-11-05: 사용자가 실행하는 고정 revision 엔진 준비 도구. 모델을 실행하지 않는다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

COMMIT = "10d4846a572aebf21c0482d140b00e74c0f1ad20"
URL = "https://github.com/ggml-org/llama.cpp.git"


def preflight() -> dict[str, str | None]:
    return {name: shutil.which(name) for name in ("git", "cmake", "ninja", "clang", "clang++")}


def build_commands(source: Path, build: Path, gpu_target: str) -> list[list[str]]:
    if not gpu_target or not gpu_target.startswith("gfx") or not gpu_target[3:].isalnum():
        raise ValueError("실제 장비에서 확인한 gfx GPU target이 필요합니다")
    return [["cmake", "-S", str(Path(__file__).resolve().parent), "-B", str(build),
             "-G", "Ninja", f"-DLLAMA_SOURCE_DIR={source}", "-DGGML_HIP=ON",
             f"-DGPU_TARGETS={gpu_target}", "-DCMAKE_C_COMPILER=clang",
             "-DCMAKE_CXX_COMPILER=clang++", "-DCMAKE_BUILD_TYPE=Release"],
            ["cmake", "--build", str(build), "--target", "llama-expert-research", "--parallel", "2"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/llama.cpp"))
    parser.add_argument("--build", type=Path, default=Path(".cache/llama-build"))
    parser.add_argument("--gpu-target", default="")
    parser.add_argument("--execute", action="store_true", help="직접 다운로드·빌드를 실행")
    args = parser.parse_args()
    print(json.dumps({"commit": COMMIT, "tools": preflight()}, ensure_ascii=False, indent=2))
    if not args.gpu_target:
        if args.execute:
            raise SystemExit("--gpu-target에 실제 확인한 gfx 값을 지정하세요")
        return
    source, build = args.source.resolve(), args.build.resolve()
    commands = build_commands(source, build, args.gpu_target)
    for command in commands:
        print(subprocess.list2cmdline(command))
    if not args.execute:
        return
    missing = [name for name, value in preflight().items() if value is None]
    if missing:
        raise SystemExit(f"빌드 도구 없음: {', '.join(missing)}. Windows Native Tools/HIP SDK 환경을 준비하세요")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none", URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", COMMIT], check=True)
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if actual != COMMIT or dirty:
        raise SystemExit("upstream revision 또는 작업 트리가 다릅니다. 기존 파일을 자동 수정하지 않습니다")
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
