"""07 전용 allowlist ZIP과 안전한 Colab 노트북 생성."""
from __future__ import annotations
import hashlib
import inspect
import os
import tempfile
import json
from pathlib import Path

from tools import colab_archive
from tools.notebook06_bundle import archive_bytes, BOOTSTRAP

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'dist/colab'
SNAPSHOT = 'runs/datasets/aime2026-lcbv6-new-003'
SOURCE_FILES = ['requirements-research.txt',
    *['moe_prune/' + x for x in ('__init__.py', 'base_benchmark.py', 'benchmark_data.py', 'datasets.py',
      'execution_safety.py', 'gguf.py', 'grading.py', 'memory_safety.py', 'research.py', 'resident.py')],
    *['tools/' + x for x in ('benchmark175.py', 'lcb_grade.py', 'lcb_prepare.py', 'colab_archive.py', 'colab_setup.py')],
    *['tools/native/' + x for x in ('CMakeLists.txt', 'expert_research.cpp', 'gpu_memory.py', 'prepare.py')]]
DATA_FILES = [SNAPSHOT + '/manifest.json', SNAPSHOT + '/records.jsonl', SNAPSHOT + '-id-validation.json',
              'runs/gemma4-ordered-e80-001/export-manifest.json',
              'runs/gemma4-lcb100-reordered-001/permutation-manifest.json']


# 2026-11-09: 파일을 명시적으로 제한해 모델·인증자료·빌드 및 캐시를 묶지 않는다.
def build_artifacts():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {}
    for kind, paths in (('source', SOURCE_FILES), ('data', DATA_FILES)):
        raw = archive_bytes(ROOT, paths)
        digest = hashlib.sha256(raw).hexdigest()
        path = OUTPUT / f'putare07-{kind}-{digest[:12]}.zip'
        if not path.exists():
            with path.open('xb') as stream:
                stream.write(raw)
        elif colab_archive.sha256_file(path) != digest:
            raise FileExistsError(path)
        result[kind] = {'path': path, 'sha256': digest}
    return result


def write_notebook(artifacts):
    archive_source = inspect.getsource(colab_archive.sha256_file) + '\n\n' + inspect.getsource(colab_archive.restore_archives)
    setup = BOOTSTRAP.format(archive_source=archive_source,
                             source_sha=artifacts['source']['sha256'], data_sha=artifacts['data']['sha256'])
    def cell(kind, source):
        item = {'cell_type': kind, 'metadata': {}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            item.update({'execution_count': None, 'outputs': []})
        return item
    source_zip = '/content/drive/MyDrive/Putare/' + artifacts['source']['path'].name
    data_zip = '/content/drive/MyDrive/Putare/' + artifacts['data']['path'].name
    settings = f'''SETUP_COLAB = False
RUN_GENERATION = False
RUN_GRADING = False
SOURCE_ZIP = {source_zip!r}
DATA_ZIP = {data_zip!r}
MODEL_PATH = '/content/drive/MyDrive/models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf'
CHECKPOINT_DIR = None
if SETUP_COLAB:
    _PUTARE_SETUP_READY = False
    from google.colab import drive
    drive.mount('/content/drive')
'''
    run = '''if RUN_GENERATION or RUN_GRADING:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 SETUP_COLAB을 켜고 환경 준비를 완료하세요.')
    from pathlib import Path
    from tools import benchmark175
    MODEL = Path(MODEL_PATH)
    ENGINE = ROOT / '.cache/llama-build/bin/llama-expert-research'
    RUN_DIR = ROOT / 'runs/gemma4-e80-lcb175-001'
    if RUN_GENERATION:
        print(benchmark175.generate(RUN_DIR, model=MODEL, engine=ENGINE, checkpoint_dir=CHECKPOINT_DIR))
    if RUN_GRADING:
        print(benchmark175.grade(RUN_DIR, model=MODEL, engine=ENGINE, checkpoint_dir=CHECKPOINT_DIR))
else:
    print('생성 및 채점 대기 중입니다.')
'''
    note = '''# E80 LiveCodeBench 175문항

소스·데이터 ZIP을 `MyDrive/Putare`에, 모델을 `MyDrive/models`에 별도 업로드하세요. Drive를 직접 마운트하고 필요한 단계만 켜세요. 기본 Run all은 설치·추론·채점을 실행하지 않습니다.

pruning에 관측한 100문제를 포함한 175문항이므로 독립 held-out 평가는 아닙니다. context 262144에서 Q8 KV 약 29.22 GiB, 모델 파일과 합계 약 38.03 GiB는 작업 버퍼를 제외한 추정입니다. A100 80GB 또는 Blackwell 96GB를 권장하며 40GB는 여유가 부족할 수 있습니다. 실제 CUDA 동작은 미검증입니다.

모든 처리 토큰과 30개 층의 선택 Expert 8개 ID와 정규화 gate 가중치를 보존하며 weighted_output_norms는 계산하거나 저장하지 않습니다. 각 문제의 전체 routing 원본은 검증된 gzip으로 보존하며 크기가 매우 커질 수 있습니다. 결과는 런타임 디스크에 저장됩니다. CHECKPOINT_DIR에 Drive 경로를 지정하면 각 완료 문제를 검증 후 백업하며, 저장 공간이 부족하면 실행이 중단됩니다. 기본값 None은 백업하지 않습니다. 계측 callback 경계에서는 일부 CUDA fusion이 제한될 수 있습니다. 공식 채점은 비격리 코드 실행입니다.
'''
    notebook = {'cells': [cell('markdown', note), cell('code', settings), cell('code', setup), cell('code', run)],
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    path = ROOT / 'ipynb/07_expert80_lcb175.ipynb'
    original = path.read_text(encoding='utf-8') if path.exists() else None
    if original is not None:
        existing = json.loads(original)
        # 2026-11-09: 사용자 설정·출력·추가 셀은 보존하고 생성된 준비 코드만 교체한다.
        setup_cell = next((c for c in existing['cells'] if c.get('metadata', {}).get('putare07') == 'setup'), None)
        settings_cell = next((c for c in existing['cells'] if c.get('metadata', {}).get('putare07') == 'settings'), None)
        if setup_cell is None and len(existing['cells']) >= 4:
            settings_cell, setup_cell = existing['cells'][1:3]
            settings_cell.setdefault('metadata', {})['putare07'] = 'settings'
            setup_cell.setdefault('metadata', {})['putare07'] = 'setup'
        if setup_cell is None or settings_cell is None:
            raise ValueError('07 설정/준비 셀을 식별할 수 없습니다')
        from tools.notebook06_bundle import _old_hashes
        old_setup = dict(setup_cell)
        old_setup['metadata'] = {'putare_setup': 'setup'}
        previous = _old_hashes([old_setup])
        lines = ''.join(settings_cell['source']).splitlines(keepends=True)
        for kind, variable in (('source', 'SOURCE_ZIP'), ('data', 'DATA_ZIP')):
            old = previous.get(kind)
            old_path = f'/content/drive/MyDrive/Putare/putare07-{kind}-{old[:12]}.zip' if old else None
            new_path = '/content/drive/MyDrive/Putare/' + artifacts[kind]['path'].name
            for i, line in enumerate(lines):
                if old_path and line.strip() in (f"{variable} = {old_path!r}", f'{variable} = "{old_path}"'):
                    lines[i] = f'{variable} = {new_path!r}\n'
        text = ''.join(lines)
        if 'drive.mount(' not in text:
            text += "\nif SETUP_COLAB:\n    _PUTARE_SETUP_READY = False\n    from google.colab import drive\n    drive.mount('/content/drive')\n"
        if 'CHECKPOINT_DIR =' not in text:
            text += '\nCHECKPOINT_DIR = None\n'
        settings_cell['source'] = text.splitlines(keepends=True)
        for item in existing['cells']:
            if item.get('cell_type') != 'code':
                continue
            code = ''.join(item['source'])
            code = code.replace('benchmark175.generate(RUN_DIR, model=MODEL, engine=ENGINE)',
                                'benchmark175.generate(RUN_DIR, model=MODEL, engine=ENGINE, checkpoint_dir=CHECKPOINT_DIR)')
            code = code.replace('benchmark175.grade(RUN_DIR, model=MODEL, engine=ENGINE)',
                                'benchmark175.grade(RUN_DIR, model=MODEL, engine=ENGINE, checkpoint_dir=CHECKPOINT_DIR)')
            item['source'] = code.splitlines(keepends=True)
        setup_cell['source'] = setup.splitlines(keepends=True)
        notebook = existing
    else:
        notebook['cells'][1]['metadata']['putare07'] = 'settings'
        notebook['cells'][2]['metadata']['putare07'] = 'setup'
    for index, item in enumerate(notebook['cells'][:4]):
        item.setdefault('id', f'putare07-{index}')
    payload = json.dumps(notebook, ensure_ascii=False, indent=1) + '\n'
    temp = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='\n', dir=path.parent,
                                         prefix='.putare07-', suffix='.tmp', delete=False) as stream:
            temp = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if (path.read_text(encoding='utf-8') if path.exists() else None) != original:
            raise RuntimeError('노트북이 동시에 변경되어 덮어쓰지 않습니다')
        os.replace(temp, path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


if __name__ == '__main__':
    artifacts = build_artifacts()
    write_notebook(artifacts)
    for kind, item in artifacts.items():
        print(kind, item['path'], item['sha256'])
