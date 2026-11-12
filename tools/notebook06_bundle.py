"""06번 Colab용 소스·데이터 ZIP과 작은 부트스트랩을 갱신한다."""
from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import colab_archive


NOTEBOOK = ROOT / 'ipynb/06_expert80_benchmark.ipynb'
OUTPUT = ROOT / 'dist/colab'
DRIVE = '/content/drive/MyDrive/Putare'
LCB = 'download/gemma4-lcb-first100-local-grading'
SOURCE_FILES = [
    'requirements-research.txt',
    *['moe_prune/' + x for x in ('__init__.py', 'base_benchmark.py', 'benchmark_data.py', 'datasets.py',
                                  'execution_safety.py', 'gguf.py', 'grading.py', 'memory_safety.py',
                                  'research.py', 'resident.py')],
    *['tools/' + x for x in ('benchmark80.py', 'manual80.py', 'manual80_grade.py', 'manual104.py',
                              'manual104_grade.py', 'lcb_grade.py', 'lcb_prepare.py', 'colab_archive.py',
                              'colab_setup.py')],
    *['tools/native/' + x for x in ('CMakeLists.txt', 'expert_research.cpp', 'gpu_memory.py', 'prepare.py')],
]
DATA_FILES = ['runs/gemma4-ordered-e80-001/export-manifest.json',
              'runs/gemma4-lcb100-reordered-001/permutation-manifest.json']

SETTINGS_SOURCE = '''\
# 2026-11-09: 설치와 빌드는 명시적으로 켰을 때만 실행한다.
SETUP_COLAB = False
MODEL_PATH = None
if SETUP_COLAB:
    _PUTARE_SETUP_READY = False
'''

GENERATION_SOURCE = '''\
if RUN_GENERATION:
    if not globals().get('_PUTARE_SETUP_READY', True):
        raise RuntimeError('환경 준비가 완료되지 않았습니다. 준비 셀을 다시 실행하세요.')
    if not MODEL.is_file():
        raise FileNotFoundError(f'모델 파일을 준비하세요: {MODEL}')
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

BOOTSTRAP = '''\
import hashlib, json, os, shutil, sys, zipfile
from pathlib import Path, PurePosixPath

{archive_source}

if SETUP_COLAB:
    _PUTARE_SETUP_READY = False
    _source_sha256 = {source_sha!r}
    _data_sha256 = {data_sha!r}
    ROOT = Path('/content/Putare') / (_source_sha256[:12] + '-' + _data_sha256[:12])
    if not (3, 12) <= sys.version_info[:2] < (3, 13):
        raise RuntimeError('Python 3.12가 필요합니다. Colab 런타임 버전을 확인하세요.')
    if not os.path.ismount('/content/drive'):
        raise RuntimeError('Google Drive가 마운트되지 않았습니다. 설정 셀에서 직접 마운트하세요.')
    # 2026-11-09: 같은 커널의 이전 버전 모듈 혼용을 막는다.
    for _name, _module in list(sys.modules.items()):
        if _name not in ('tools', 'moe_prune') and not _name.startswith(('tools.', 'moe_prune.')):
            continue
        _file = getattr(_module, '__file__', None)
        if _file and not Path(_file).resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError('다른 Putare 버전이 이미 로드되었습니다. 커널을 다시 시작하세요: ' + _name)
        _paths = getattr(_module, '__path__', ()) if not _file else ()
        if any(not Path(p).resolve().is_relative_to(ROOT.resolve()) for p in _paths):
            raise RuntimeError('다른 Putare 버전이 이미 로드되었습니다. 커널을 다시 시작하세요: ' + _name)
    restore_archives(ROOT, [(SOURCE_ZIP, _source_sha256), (DATA_ZIP, _data_sha256)])
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.colab_setup import prepare_environment
    _PUTARE_SETUP_STATE = prepare_environment(ROOT, globals().get('_PUTARE_SETUP_STATE'))
    _expected_model = Path(MODEL_PATH) if MODEL_PATH else ROOT / 'models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf'
    print('환경 준비 완료. 모델 경로:', _expected_model, '| 존재:', _expected_model.is_file())
    if not _expected_model.is_file():
        print('추론 전 모델 파일을 위 경로에 준비하세요.')
    _PUTARE_SETUP_READY = True
'''


def data_paths(root=ROOT):
    bundle = root / LCB
    inventory = json.loads((bundle / 'SHA256SUMS.json').read_text(encoding='utf-8'))
    if not isinstance(inventory, dict):
        raise ValueError('LCB 인벤토리 형식 오류')
    result = list(DATA_FILES) + [f'{LCB}/SHA256SUMS.json']
    for name, digest in inventory.items():
        path = PurePosixPath(name)
        # 이름을 먼저 검사해 인증자료로 사용되는 파일을 열지 않는다.
        if (path.is_absolute() or any(part in ('', '.', '..') for part in name.split('/'))
                or '\\' in name or ':' in name or any(p.startswith('.') and p != '.gitignore' for p in path.parts)
                or path.name.lower() in {'credentials.json', 'secrets.json', 'token.json', '.netrc', '.npmrc', '.pypirc', '.git-credentials'}
                or (path.suffix.lower() == '.json' and path.name.lower().startswith(('client-secret', 'client_secret', 'service-account', 'service_account')))
                or path.suffix.lower() in {'.key', '.p12', '.pfx', '.jks', '.keystore'}):
            raise ValueError(f'허용되지 않은 채점 묶음 경로: {name}')
        source = bundle.joinpath(*path.parts)
        if (any(bundle.joinpath(*path.parts[:i]).is_symlink() for i in range(1, len(path.parts) + 1))
                or not source.is_file() or colab_archive.sha256_file(source) != digest):
            raise ValueError(f'채점 묶음 무결성 오류: {name}')
        result.append(f'{LCB}/{name}')
    if len(result) != len(set(result)):
        raise ValueError('중복 묶음 경로')
    return result


def archive_bytes(root, paths):
    inventory = {}
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        def put(name, data):
            item = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.external_attr = 0o100644 << 16
            archive.writestr(item, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        for name in sorted(paths):
            path = root / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f'입력 파일 없음 또는 링크: {name}')
            data = path.read_bytes()
            inventory[name] = hashlib.sha256(data).hexdigest()
            put(name, data)
        put('_putare_inventory.json', json.dumps(inventory, sort_keys=True))
    return out.getvalue()


def build_artifacts(root=None, output=None):
    root = ROOT if root is None else Path(root)
    output = OUTPUT if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for kind, paths in (('source', SOURCE_FILES), ('data', data_paths(root))):
        raw = archive_bytes(root, paths)
        digest = hashlib.sha256(raw).hexdigest()
        path = output / f'putare-{kind}-{digest[:12]}.zip'
        created = not path.exists()
        if created:
            with path.open('xb') as stream:
                stream.write(raw)
        elif colab_archive.sha256_file(path) != digest:
            raise FileExistsError(f'같은 이름의 ZIP 내용이 다릅니다: {path}')
        results[kind] = {'path': path, 'sha256': digest, 'created': created}
    return results


def _setting_values(source):
    values = {}
    for index, line in enumerate(source.splitlines(keepends=True)):
        match = re.match(r'^(SOURCE_ZIP|DATA_ZIP)\s*=', line)
        if match:
            name = match.group(1)
            try:
                node = ast.parse(line).body[0]
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError, AttributeError):
                value = None
            values[name] = (index, value)
    return values


def _update_settings(source, previous, artifacts):
    values = _setting_values(source)
    updates = []
    lines = source.splitlines(keepends=True)
    for kind, name in (('source', 'SOURCE_ZIP'), ('data', 'DATA_ZIP')):
        path = f"{DRIVE}/{artifacts[kind]['path'].name}"
        old = previous.get(kind)
        old_default = f'{DRIVE}/putare-{kind}-{old[:12]}.zip' if old else None
        if name not in values:
            updates.append(f'{name} = {path!r}\n')
        elif old_default is not None and values[name][1] == old_default:
            lines[values[name][0]] = f'{name} = {path!r}\n'
    return ''.join(lines).rstrip() + '\n' + ''.join(updates)


def _old_hashes(cells):
    setup = next((c for c in cells if c.get('metadata', {}).get('putare_setup') == 'setup'), None)
    if setup is None:
        return {}
    source = ''.join(setup['source'])
    # 이전 base64 셀을 구문 분석하지 않고 작은 생성 셀에만 SHA가 있을 때 읽는다.
    if len(source) > 100_000:
        return {}
    found = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('_source_sha256', '_data_sha256') and isinstance(node.value, ast.Constant):
                found[name[1:-7]] = node.value.value
    return found


def update_notebook():
    original = NOTEBOOK.read_text(encoding='utf-8')
    notebook = json.loads(original)
    previous = _old_hashes(notebook['cells'])
    artifacts = build_artifacts()
    archive_source = inspect.getsource(colab_archive.sha256_file) + '\n\n' + inspect.getsource(colab_archive.restore_archives)
    setup_source = BOOTSTRAP.format(archive_source=archive_source,
                                   source_sha=artifacts['source']['sha256'], data_sha=artifacts['data']['sha256'])
    for cell in notebook['cells']:
        tag = cell.get('metadata', {}).get('putare_setup')
        if cell.get('cell_type') == 'markdown' and ''.join(cell.get('source', [])).startswith('## Colab 환경 자동 준비'):
            cell['source'] = ['## Colab 환경 자동 준비\n',
                '소스 ZIP과 데이터 ZIP을 Google Drive의 `MyDrive/Putare` 폴더에 업로드하세요.\n',
                '설정 셀에서 Drive를 마운트하고 환경 준비 셀을 실행하세요. 모델은 `MODEL_PATH`에 별도로 준비합니다.\n',
                '준비 셀은 단독 재실행할 수 있습니다. 이전 분할 업로드 재조립 셀은 자동 실행하지 않습니다.\n']
        elif tag == 'settings':
            cell['source'] = _update_settings(''.join(cell['source']), previous, artifacts).splitlines(keepends=True)
        elif tag == 'setup':
            cell['source'] = setup_source.splitlines(keepends=True)
        elif tag == 'generation':
            cell['source'] = GENERATION_SOURCE.splitlines(keepends=True)
    if not all(any(c.get('metadata', {}).get('putare_setup') == t for c in notebook['cells']) for t in ('settings', 'setup')):
        raise ValueError('노트북의 설정 또는 환경 준비 셀이 없습니다')
    # 2026-11-09: 갱신 중 Windows 쓰기 오류가 나도 기존 노트북이 잘리지 않도록 원자적으로 교체한다.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='\n',
                                         dir=NOTEBOOK.parent, prefix='.putare-notebook-',
                                         suffix='.tmp', delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(notebook, stream, ensure_ascii=False, indent=1)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if NOTEBOOK.read_text(encoding='utf-8') != original:
            raise RuntimeError('노트북이 갱신 중 변경되어 덮어쓰지 않습니다.')
        os.replace(temp_path, NOTEBOOK)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    print('새로 생성된 ZIP (Drive에 없으면 업로드):')
    for item in artifacts.values():
        if item['created']:
            print(f"  {item['path']} SHA256={item['sha256']}")
    reused = [item['path'].name for item in artifacts.values() if not item['created']]
    print(f"기존 ZIP 재사용 {len(reused)}개: {', '.join(reused) or '없음'}")
    print('Drive에 실제로 업로드됐는지는 확인하지 않습니다. 처음이라면 소스와 데이터 ZIP 둘 다 올리세요.')
    return artifacts


if __name__ == '__main__':
    update_notebook()
