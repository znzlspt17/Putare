"""2026-11-05: 사용자가 실행하는 고정 revision 엔진 준비 도구. 모델을 실행하지 않는다."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

COMMIT = "10d4846a572aebf21c0482d140b00e74c0f1ad20"
URL = "https://github.com/ggml-org/llama.cpp.git"
ROOT = Path(__file__).resolve().parents[2]


def discover_local_tools(root: Path, python_prefix: Path) -> dict[str, Path | None]:
    sdk = python_prefix / 'Lib/site-packages/_rocm_sdk_devel'
    candidates = {'cmake': root / '.cache/native-tools/cmake/data/bin/cmake.exe',
                  'ninja': root / '.cache/native-tools/bin/ninja.exe',
                  'clang': sdk / 'lib/llvm/bin/clang.exe', 'clang++': sdk / 'lib/llvm/bin/clang++.exe'}
    return {**{name: path if path.is_file() else None for name, path in candidates.items()},
            'rocm': sdk if sdk.is_dir() else None}


def configure_local_environment() -> None:
    """2026-11-06: 임시 자식 환경만 구성하고 인증 등 전체 환경을 열람하지 않는다."""
    found = discover_local_tools(ROOT, Path(sys.prefix))
    if os.name == 'nt' and not (os.environ.get('INCLUDE') and os.environ.get('LIB')):
        program_files = Path(os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)'))
        choices = [program_files / f'Microsoft Visual Studio/{version}/{edition}/VC/Auxiliary/Build/vcvars64.bat'
                   for version in ('2022', '18') for edition in ('BuildTools', 'Community', 'Professional', 'Enterprise')]
        batch = next((path for path in choices if path.is_file()), None)
        if batch:
            result = subprocess.run(['cmd.exe', '/d', '/u', '/c', 'call', str(batch), '>nul',
                                     '&&', 'set', 'PATH', '&&', 'set', 'INCLUDE', '&&', 'set', 'LIB'], capture_output=True,
                                    encoding='utf-16le', errors='replace', check=True)
            for line in result.stdout.splitlines():
                key, separator, value = line.partition('=')
                if separator and key.upper() in {'PATH', 'INCLUDE', 'LIB', 'LIBPATH'}:
                    os.environ[key.upper()] = value
    sdk = Path(os.environ['ROCM_PATH']) if os.environ.get('ROCM_PATH') else found['rocm']
    if sdk:
        os.environ.setdefault('ROCM_PATH', str(sdk))
        os.environ.setdefault('HIP_PATH', str(sdk))
        bitcode = sdk_device_lib_path(sdk)
        if bitcode:
            os.environ.setdefault('HIP_DEVICE_LIB_PATH', str(bitcode))
    bins = [str(path.parent) for key, path in found.items() if key != 'rocm' and path]
    if sdk and (sdk / 'bin').is_dir():
        bins.append(str(sdk / 'bin'))
    if bins:
        os.environ['PATH'] = os.pathsep.join(dict.fromkeys(bins + [os.environ.get('PATH', '')]))


def sdk_device_lib_path(rocm_root: Path) -> Path | None:
    """2026-11-06: pip SDK의 bitcode 위치는 clang 기본 ROCm 경로와 다르다."""
    candidate = rocm_root / 'lib' / 'llvm' / 'amdgcn' / 'bitcode'
    return candidate if candidate.is_dir() else None


def preflight() -> dict[str, str | None]:
    local = discover_local_tools(ROOT, Path(sys.prefix))
    return {name: shutil.which(name) or (str(local[name]) if local.get(name) else None)
            for name in ("git", "cmake", "ninja", "clang", "clang++")}


def build_commands(source: Path, build: Path, gpu_target: str) -> list[list[str]]:
    if not gpu_target or not gpu_target.startswith("gfx") or not gpu_target[3:].isalnum():
        raise ValueError("실제 장비에서 확인한 gfx GPU target이 필요합니다")
    return [["cmake", "-S", str(Path(__file__).resolve().parent), "-B", str(build),
             "-G", "Ninja", f"-DLLAMA_SOURCE_DIR={source}", "-DGGML_HIP=ON",
             "-DHIP_PLATFORM=amd", f"-DGPU_TARGETS={gpu_target}", "-DCMAKE_C_COMPILER=clang",
             "-DCMAKE_CXX_COMPILER=clang++", "-DCMAKE_BUILD_TYPE=Release"],
            ["cmake", "--build", str(build), "--target", "llama-expert-research", "--parallel", "2"]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(".cache/llama.cpp"))
    parser.add_argument("--build", type=Path, default=Path(".cache/llama-build"))
    parser.add_argument("--gpu-target", default="")
    parser.add_argument("--execute", action="store_true", help="직접 다운로드·빌드를 실행")
    args = parser.parse_args()
    if args.execute:
        configure_local_environment()
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
    if not os.environ.get('HIP_DEVICE_LIB_PATH') and os.environ.get('ROCM_PATH'):
        bitcode = sdk_device_lib_path(Path(os.environ['ROCM_PATH']))
        if bitcode:
            os.environ['HIP_DEVICE_LIB_PATH'] = str(bitcode)
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
