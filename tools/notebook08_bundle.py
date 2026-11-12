"""원본 175문항 평가의 Colab ZIP과 노트북 생성."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import colab_archive
from tools.notebook06_bundle import archive_bytes
from tools.notebook07_bundle import SOURCE_FILES, SNAPSHOT

OUTPUT = ROOT / 'dist/colab'
DATA_FILES = [SNAPSHOT + '/manifest.json', SNAPSHOT + '/records.jsonl', SNAPSHOT + '-id-validation.json']
NOTEBOOK = ROOT / 'ipynb/08_base_model_lcb175.ipynb'


# 2026-11-10: 파일 허용 목록으로 모델과 사용자 인증자료가 묶이지 않게 한다.
def build_artifacts():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for kind, paths in (('source', SOURCE_FILES), ('data', DATA_FILES)):
        payload = archive_bytes(ROOT, paths)
        digest = hashlib.sha256(payload).hexdigest()
        path = OUTPUT / f'putare08-{kind}-{digest[:12]}.zip'
        if path.exists():
            if colab_archive.sha256_file(path) != digest:
                raise ValueError('기존 ZIP의 해시가 다릅니다')
        else:
            path.write_bytes(payload)
        artifacts[kind] = {'path': path, 'sha256': digest}
    return artifacts


def write_notebook(artifacts):
    def cell(kind, name, source):
        result = {'cell_type': kind, 'id': 'putare08-' + name,
                  'metadata': {'putare08': name}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            result.update(execution_count=None, outputs=[])
        return result

    source_sha, data_sha = (artifacts[k]['sha256'] for k in ('source', 'data'))
    source_zip, data_zip = ('/content/drive/MyDrive/Putare/' + artifacts[k]['path'].name for k in ('source', 'data'))
    settings = f'''SETUP_COLAB = False
DOWNLOAD_MODEL = False
RESTORE_CHECKPOINT = False
RUN_GENERATION = False
RUN_GRADING = False
SOURCE_ZIP = {source_zip!r}
DATA_ZIP = {data_zip!r}
MODEL_PATH = '/content/models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf'
CHECKPOINT_DIR = '/content/drive/MyDrive/Putare/runs/gemma4-base-lcb175-001'
'''
    archive_code = inspect.getsource(colab_archive.sha256_file) + '\n\n' + inspect.getsource(colab_archive.restore_archives)
    restore = f'''import hashlib, json, os, shutil, sys, zipfile
from pathlib import Path, PurePosixPath

{archive_code}

if SETUP_COLAB:
    _PUTARE_SETUP_READY = False
    from google.colab import drive
    drive.mount('/content/drive')
    if not os.path.ismount('/content/drive'):
        raise RuntimeError('Google Drive가 마운트되지 않았습니다')
    ROOT = Path('/content/Putare') / ({source_sha!r}[:12] + '-' + {data_sha!r}[:12])
    for _name, _module in list(sys.modules.items()):
        if _name not in ('tools', 'moe_prune') and not _name.startswith(('tools.', 'moe_prune.')):
            continue
        _file = getattr(_module, '__file__', None)
        _paths = getattr(_module, '__path__', ()) if not _file else ()
        if (_file and not Path(_file).resolve().is_relative_to(ROOT.resolve())) or any(not Path(p).resolve().is_relative_to(ROOT.resolve()) for p in _paths):
            raise RuntimeError('다른 Putare 버전이 로드되었습니다. 커널을 다시 시작하세요: ' + _name)
    restore_archives(ROOT, [(SOURCE_ZIP, {source_sha!r}), (DATA_ZIP, {data_sha!r})])
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
'''
    setup = '''if SETUP_COLAB:
    _PUTARE_SETUP_READY = False
    from tools.colab_setup import prepare_environment
    from tools import benchmark175
    _PUTARE_SETUP_STATE = prepare_environment(ROOT, globals().get('_PUTARE_SETUP_STATE'))
    def _validate_backup():
        if not os.path.ismount('/content/drive'):
            raise RuntimeError('Google Drive 연결을 확인하세요')
        if not Path(CHECKPOINT_DIR).resolve().is_relative_to(Path('/content/drive/MyDrive').resolve()):
            raise ValueError('백업 경로는 마운트된 MyDrive 아래여야 합니다')
        benchmark175.preflight_checkpoint(CHECKPOINT_DIR)
    _validate_backup()
    ENGINE = ROOT / '.cache/llama-build/bin/llama-expert-research'
    RUN_DIR = ROOT / 'runs/gemma4-base-lcb175-001'
    MODEL = Path(MODEL_PATH)
    _PUTARE_SETUP_READY = True
    print('준비 완료:', RUN_DIR)
'''
    download = '''if DOWNLOAD_MODEL:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    print('모델 준비:', benchmark175.download_base_model(Path(MODEL_PATH)))
'''
    resume = '''if RESTORE_CHECKPOINT:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _validate_backup()
    print('백업 복원:', benchmark175.restore_checkpoint(CHECKPOINT_DIR, RUN_DIR))
'''
    generation = '''if RUN_GENERATION:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _validate_backup()
    print(benchmark175.generate(RUN_DIR, model=Path(MODEL_PATH), engine=ENGINE, checkpoint_dir=CHECKPOINT_DIR, model_profile='base'))
'''
    grading = '''if RUN_GRADING:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _validate_backup()
    summary = benchmark175.grade(RUN_DIR, model=Path(MODEL_PATH), engine=ENGINE, checkpoint_dir=CHECKPOINT_DIR, model_profile='base')
    print(f"점수: {summary['passed']}/175" if summary.get('complete') else '채점 미완료: 인프라 오류 또는 시간 초과를 확인하세요')
'''
    note = '''# 원본 Gemma 4 LiveCodeBench 175문항

`putare08-source-*.zip`과 `putare08-data-*.zip`을 `MyDrive/Putare`에 올리세요. 설정 플래그를 필요한 단계만 켜고 위에서 아래로 실행합니다. 기본 Run all은 설치·다운로드·추론·채점을 실행하지 않습니다.

모델은 Hugging Face에서 런타임 로컬 디스크로 다운로드하며 128 experts 원본과 해시를 확인합니다. Google Drive에 각 완료 문제와 전체 라우팅 기록을 필수 백업합니다. 백업이 실패하면 중단합니다. 새 런타임에서 복원하려면 `RESTORE_CHECKPOINT=True`를 켜고 빈 로컬 실행 경로에 복원한 후 생성 셀을 다시 실행하세요.

재개에는 모델·소스·설정·엔진 파일의 해시와 경로가 기존 실행과 같아야 합니다. 새 런타임에서 엔진을 다시 빌드한 결과가 다르면 재개가 거부되며, 기존 백업을 보존한 채 별도 실행 경로와 백업 경로를 사용해야 합니다.

모든 토큰의 30개 층에서 선택된 Expert ID와 gate 가중치를 저장하므로 디스크와 Drive 사용량이 매우 커질 수 있습니다. 사용량 경고는 강제 메모리 제한이 아닙니다. 실제 RTX PRO 6000 CUDA 동작은 아직 검증되지 않았습니다. 공식 채점은 별도 프로세스에서 실행되지만 보안 격리가 없으므로 생성 코드에 현재 런타임 사용자 권한이 있습니다.
'''
    cells = [cell('markdown', 'intro', note), cell('code', 'settings', settings),
             cell('code', 'restore-archives', restore), cell('code', 'setup', setup),
             cell('code', 'download', download), cell('code', 'restore', resume),
             cell('code', 'generation', generation), cell('code', 'grading', grading)]
    notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    # 2026-11-10: 다시 묶을 때 사용자가 편집한 셀과 출력을 덮어쓰지 않는다.
    if NOTEBOOK.exists():
        old = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        old_cells = {c.get('metadata', {}).get('putare08'): c for c in old['cells']}
        for name in ('settings', 'restore-archives'):
            if name not in old_cells:
                raise ValueError('08 노트북의 설정 또는 복원 셀을 식별할 수 없습니다')
        previous = old_cells['restore-archives']
        if previous.get('outputs') or previous.get('execution_count') is not None:
            raise ValueError('실행한 노트북은 자동 갱신하지 않습니다. 백업 후 수동으로 ZIP 해시를 변경하세요')
        previous_code = ''.join(previous['source'])
        match = re.search(r"restore_archives\(ROOT, \[\(SOURCE_ZIP, '([0-9a-f]{64})'\), \(DATA_ZIP, '([0-9a-f]{64})'\)\]\)", previous_code)
        if not match:
            raise ValueError('기존 ZIP 해시를 식별할 수 없습니다')
        lines = old_cells['settings']['source']
        for kind, variable, previous_sha, path in (
                ('source', 'SOURCE_ZIP', match.group(1), source_zip),
                ('data', 'DATA_ZIP', match.group(2), data_zip)):
            old_default = f'/content/drive/MyDrive/Putare/putare08-{kind}-{previous_sha[:12]}.zip'
            for i, line in enumerate(lines):
                if line.strip() in (f"{variable} = {old_default!r}", f'{variable} = "{old_default}"'):
                    lines[i] = f'{variable} = {path!r}\n'
        for item in old['cells']:
            if item is previous:
                item['source'] = restore.splitlines(keepends=True)
        notebook = old
    NOTEBOOK.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


if __name__ == '__main__':
    built = build_artifacts()
    write_notebook(built)
    for name, item in built.items():
        print(name, item['path'], item['sha256'])
