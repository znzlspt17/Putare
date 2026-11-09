"""06번 노트북에 로컬 소스와 고정 채점 묶음을 안전하게 포함한다.

사용: python tools/notebook06_bundle.py
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'ipynb/06_expert80_benchmark.ipynb'
LCB = 'download/gemma4-lcb-first100-local-grading'
# 2026-11-09: 예상치 못한 로컬 파일과 인증 자료가 노트북에 포함되지 않도록 명시한다.
FILES = [
    'requirements-research.txt',
    *['moe_prune/' + name for name in (
        '__init__.py', 'base_benchmark.py', 'benchmark_data.py', 'datasets.py',
        'execution_safety.py', 'gguf.py', 'grading.py', 'memory_safety.py',
        'research.py', 'resident.py')],
    *['tools/' + name for name in (
        'benchmark80.py', 'manual80.py', 'manual80_grade.py', 'manual104.py',
        'manual104_grade.py', 'lcb_grade.py', 'lcb_prepare.py')],
    *['tools/native/' + name for name in (
        'CMakeLists.txt', 'expert_research.cpp', 'gpu_memory.py', 'prepare.py')],
    'runs/gemma4-ordered-e80-001/export-manifest.json',
    'runs/gemma4-lcb100-reordered-001/permutation-manifest.json',
]

# 이 함수는 노트북에 그대로 삽입된다. 표준 라이브러리만 사용한다.
RESTORE_SOURCE = '''\
def restore_embedded_bundle(root, encoded, expected_sha256):
    # 2026-11-09: 검증과 충돌 확인을 모두 마친 후에만 사용자 파일을 생성한다.
    import base64, hashlib, io, json, zipfile
    from pathlib import Path, PurePosixPath
    root = Path(root)
    if root.is_symlink():
        raise ValueError('복원 경로가 심볼릭 링크입니다')
    try:
        raw = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('내장 묶음 SHA256 불일치')
        archive = zipfile.ZipFile(io.BytesIO(raw))
        names = archive.namelist()
        if len(names) != len(set(names)) or '_putare_inventory.json' not in names:
            raise ValueError('내장 묶음 파일 목록 오류')
        expected = json.loads(archive.read('_putare_inventory.json'))
        if not isinstance(expected, dict) or set(names) != set(expected) | {'_putare_inventory.json'}:
            raise ValueError('내장 묶음 인벤토리 불일치')
        for name, digest in expected.items():
            p = PurePosixPath(name)
            if p.is_absolute() or not p.parts or any(x in ('', '.', '..') for x in name.split('/')) or '\\\\' in name or ':' in name:
                raise ValueError('안전하지 않은 묶음 경로')
            info = archive.getinfo(name)
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('묶음에 링크 또는 디렉터리가 있습니다')
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError('내장 파일 SHA256 불일치: ' + name)
            target = root.joinpath(*p.parts)
            for part in [root, *[root.joinpath(*p.parts[:i]) for i in range(1, len(p.parts) + 1)]]:
                if part.is_symlink():
                    raise ValueError('복원 경로에 심볼릭 링크가 있습니다: ' + str(part))
                if part != target and part.exists() and not part.is_dir():
                    raise FileExistsError('복원 폴더 경로에 파일이 있습니다: ' + str(part))
            if target.exists() and (not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest):
                raise FileExistsError('기존 파일이 달라 덮어쓰지 않습니다: ' + str(target))
        for name in expected:
            target = root.joinpath(*PurePosixPath(name).parts)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as output:
                    output.write(archive.read(name))
    except (zipfile.BadZipFile, KeyError, UnicodeError) as error:
        raise ValueError('내장 묶음이 손상되었습니다') from error
    return len(expected)
'''

SETTINGS_SOURCE = '''\
# 2026-11-09: 설치와 빌드는 명시적으로 켰을 때만 실행한다.
SETUP_COLAB = False
MODEL_PATH = None  # None이면 /content/Putare/models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf
if SETUP_COLAB:
    _PUTARE_SETUP_READY = False
'''

SETUP_SOURCE = '''\
if SETUP_COLAB:
    _PUTARE_SETUP_READY = False
    import os, shutil, subprocess, sys
    from pathlib import Path
    ROOT = Path('/content/Putare')
    if not (3, 12) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError('Python 3.12가 필요합니다. Colab 런타임 버전을 확인하세요.')
    missing = [name for name in ('git', 'cmake', 'ninja', 'nvcc', 'c++') if not shutil.which(name)]
    if missing:
        raise RuntimeError('필요한 빌드 도구가 없습니다: ' + ', '.join(missing) + '. Colab에 git, cmake, ninja, C++ 컴파일러와 CUDA Toolkit을 설치하세요.')
    restored = restore_embedded_bundle(ROOT, _BUNDLE_DATA, _BUNDLE_SHA256)
    print(f'소스와 자료 {restored}개 확인 완료')
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    subprocess.run([sys.executable, '-m', 'pip', '--isolated', 'install', '-r', str(ROOT / 'requirements-research.txt')], check=True)
    from tools.native.gpu_memory import NvmlDevice
    device = NvmlDevice()
    try:
        major, minor = device.nvml.nvmlDeviceGetCudaComputeCapability(device.handle)
        target = f'{major}{minor}'
        print(f'GPU: {device.name}, Compute Capability: {major}.{minor}')
    finally:
        device.close()
    codes = subprocess.run(['nvcc', '--list-gpu-code'], check=True, capture_output=True, text=True).stdout.split()
    if f'sm_{target}' not in codes:
        raise RuntimeError(f'현재 nvcc가 sm_{target}을 지원하지 않습니다. GPU를 지원하는 CUDA Toolkit이 필요합니다.')
    # 기존 소스/빌드가 있으면 prepare가 고정 revision과 오염 여부를 검증한다.
    subprocess.run([sys.executable, str(ROOT / 'tools/native/prepare.py'), '--backend', 'cuda', '--gpu-target', target, '--execute'], cwd=ROOT, check=True)
    from moe_prune.benchmark_data import prepare_grader
    prepare_grader(ROOT)
    _PUTARE_SETUP_READY = True
    _expected_model = Path(MODEL_PATH) if MODEL_PATH else ROOT / 'models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf'
    print('환경 준비 완료. 모델 경로:', _expected_model, '| 존재:', _expected_model.is_file())
    if not _expected_model.is_file():
        print('추론 전 모델 파일을 위 경로에 준비하세요.')
'''

GENERATION_SOURCE = '''\
if RUN_GENERATION:
    if not globals().get('_PUTARE_SETUP_READY', True):
        raise RuntimeError('환경 준비가 완료되지 않았습니다. 준비 셀을 다시 실행하세요.')
    if not MODEL.is_file():
        raise FileNotFoundError(f'모델 파일을 준비하세요: {MODEL}')
    # 2026-11-09: 모델 산출물에 실제 사용한 GPU 정보를 함께 기록한다.
    from tools.native.gpu_memory import NvmlDevice
    device = NvmlDevice()
    try:
        major, minor = device.nvml.nvmlDeviceGetCudaComputeCapability(device.handle)
        gpu_info = {'name': device.name, 'uuid': device.uuid, 'compute_capability': f'{major}.{minor}'}
    finally:
        device.close()
    import json
    result = trial.generate(RUN_DIR, model=MODEL, engine=ENGINE)
    with (RUN_DIR / 'gpu-info.json').open('x', encoding='utf-8') as output:
        json.dump(gpu_info, output, ensure_ascii=False, indent=2)
    print(result)
else:
    print('생성 실행 대기 중입니다.')
'''


def code_cell(source, tag):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {'putare_setup': tag},
            'outputs': [], 'source': source.splitlines(keepends=True)}


def bundle_paths():
    bundle = ROOT / LCB
    inventory = json.loads((bundle / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    if not isinstance(inventory, dict):
        raise ValueError('LCB 인벤토리 형식 오류')
    result = list(FILES) + [f'{LCB}/SHA256SUMS.json']
    for name, digest in inventory.items():
        path = PurePosixPath(name)
        # 이름만 검사하며 금지된 파일은 열지 않는다.
        if (path.is_absolute() or any(part in ('', '.', '..') for part in name.split('/'))
                or '\\' in name or ':' in name or any(part.startswith('.') and part != '.gitignore' for part in path.parts)
                or path.name.lower() in {'credentials.json', 'secrets.json', 'token.json', '.netrc', '.npmrc', '.pypirc', '.git-credentials'}
                or (path.suffix.lower() == '.json' and path.name.lower().startswith(
                    ('client-secret', 'client_secret', 'service-account', 'service_account')))
                or path.suffix.lower() in {'.key', '.p12', '.pfx', '.jks', '.keystore'}):
            raise ValueError(f'허용되지 않은 채점 묶음 경로: {name}')
        source = bundle.joinpath(*path.parts)
        if (any(bundle.joinpath(*path.parts[:i]).is_symlink() for i in range(1, len(path.parts) + 1))
                or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest):
            raise ValueError(f'채점 묶음 무결성 오류: {name}')
        result.append(f'{LCB}/{name}')
    if len(result) != len(set(result)):
        raise ValueError('중복 묶음 경로')
    return result


def make_archive():
    contents = {}
    for name in bundle_paths():
        path = ROOT / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'입력 파일 없음 또는 링크: {name}')
        contents[name] = path.read_bytes()
    inventory = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def put(name, data):
            item = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        put('_putare_inventory.json', json.dumps(inventory, sort_keys=True))
        for name, data in sorted(contents.items()):
            put(name, data)
    return out.getvalue()


def update_notebook():
    import ast
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    if not any(c.get('metadata', {}).get('putare_setup') == 'settings' for c in notebook['cells']):
        # 2026-11-09: 기존 실행 셀과 동작은 보존하고 준비 셀을 그 앞에 삽입한다.
        notebook['cells'][1:1] = [
            {'cell_type': 'markdown', 'metadata': {}, 'source': [
                '## Colab 환경 자동 준비\n',
                '`SETUP_COLAB = True`로 변경하고 위에서부터 실행하세요. 모델은 `MODEL_PATH`에 별도로 준비합니다.\n',
                '최초 준비는 의존성 설치와 CUDA 엔진 빌드를 수행합니다. 기존 파일을 덮어쓰지 않습니다.\n']},
            code_cell(SETTINGS_SOURCE, 'settings'),
            code_cell(RESTORE_SOURCE, 'restore'),
            code_cell('', 'payload'),
            code_cell(SETUP_SOURCE, 'setup'),
        ]
        for cell in notebook['cells']:
            if cell.get('cell_type') != 'code' or cell.get('metadata', {}).get('putare_setup'):
                continue
            source = ''.join(cell['source'])
            source = source.replace('MODEL = manual80.MODEL', 'MODEL = Path(MODEL_PATH) if MODEL_PATH else manual80.MODEL')
            source = source.replace("'--backend', 'cuda', '--gpu-target', '80'", "'--backend', 'cuda'")
            if source.startswith('if RUN_GENERATION:'):
                source = GENERATION_SOURCE
                cell.setdefault('metadata', {})['putare_setup'] = 'generation'
            if source.startswith('if RUN_GRADING:'):
                source = source.replace('if RUN_GRADING:\n', "if RUN_GRADING:\n    if not globals().get('_PUTARE_SETUP_READY', True):\n        raise RuntimeError('환경 준비가 완료되지 않았습니다.')\n", 1)
            cell['source'] = source.splitlines(keepends=True)
    # 초기 버전으로 생성된 셀을 새 태그·모델 경로 계약으로 이전한다.
    for cell in notebook['cells']:
        if cell.get('cell_type') == 'markdown' and ''.join(cell.get('source', [])).startswith('## Colab 환경 자동 준비'):
            cell.setdefault('id', 'putare-setup-guide')
            cell['source'] = [
                '## Colab 환경 자동 준비\n',
                '바로 아래 설정에서 `SETUP_COLAB = True`로 변경하고 위에서부터 실행하세요.\n',
                '모델은 별도로 준비합니다. 기본 경로: `/content/Putare/models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf`\n',
                '`MODEL_PATH`에 다른 모델 경로를 지정할 수 있습니다. 최초 실행은 의존성 설치와 CUDA 엔진 빌드를 수행합니다.\n',
                '소스와 채점 자료를 내장해 노트북 크기가 약 77MB이며 Colab 웹 편집기의 업로드·실행은 검증되지 않았습니다.\n',
            ]
        if cell.get('cell_type') != 'code' or cell.get('metadata', {}).get('putare_setup'):
            continue
        source = ''.join(cell['source'])
        source = source.replace('MODEL = Path(MODEL_PATH) if SETUP_COLAB else manual80.MODEL',
                                'MODEL = Path(MODEL_PATH) if MODEL_PATH else manual80.MODEL')
        if source.startswith('if RUN_GENERATION:'):
            cell.setdefault('metadata', {})['putare_setup'] = 'generation'
        cell['source'] = source.splitlines(keepends=True)
    raw = make_archive()
    encoded = base64.b64encode(raw).decode('ascii')
    chunks = [encoded[i:i + 32768] for i in range(0, len(encoded), 32768)]
    payload = "if SETUP_COLAB:\n    _BUNDLE_DATA = (\n" + ''.join(f'        {chunk!r}\n' for chunk in chunks) + ")\n    _BUNDLE_SHA256 = " + repr(hashlib.sha256(raw).hexdigest()) + '\n'
    for cell in notebook['cells']:
        tag = cell.get('metadata', {}).get('putare_setup')
        if tag:
            cell.setdefault('id', 'putare-' + tag)
        if tag in ('restore', 'payload'):
            cell.setdefault('metadata', {}).setdefault('jupyter', {})['source_hidden'] = True
        if tag == 'settings':
            old = ''.join(cell['source'])
            values = {}
            for item in ast.parse(old).body:
                if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name) and item.targets[0].id in ('SETUP_COLAB', 'MODEL_PATH'):
                    values[item.targets[0].id] = ast.literal_eval(item.value)
            updated = SETTINGS_SOURCE
            if values.get('MODEL_PATH') == '/content/Putare/models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf':
                values['MODEL_PATH'] = None
            for name in ('SETUP_COLAB', 'MODEL_PATH'):
                if name in values:
                    default = 'False' if name == 'SETUP_COLAB' else 'None'
                    updated = updated.replace(f'{name} = {default}', f'{name} = {values[name]!r}', 1)
            cell['source'] = updated.splitlines(keepends=True)
        elif tag == 'restore':
            cell['source'] = RESTORE_SOURCE.splitlines(keepends=True)
        elif tag == 'setup':
            cell['source'] = SETUP_SOURCE.splitlines(keepends=True)
        elif tag == 'generation':
            cell['source'] = GENERATION_SOURCE.splitlines(keepends=True)
        elif tag == 'payload':
            cell['source'] = payload.splitlines(keepends=True)
    if not all(any(c.get('metadata', {}).get('putare_setup') == tag for c in notebook['cells']) for tag in ('restore', 'payload')):
        raise ValueError('노트북의 복원·데이터 셀이 없습니다')
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(f'갱신 완료: {len(raw):,} bytes compressed, {len(encoded):,} base64 characters')


if __name__ == '__main__':
    update_notebook()
