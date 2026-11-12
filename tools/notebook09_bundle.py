"""2026-11-11: 비교 실험에 필요한 공개 소스와 고정 데이터만 Colab으로 묶는다."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import colab_archive
from tools.notebook06_bundle import archive_bytes
from tools.notebook07_bundle import DATA_FILES as ORIGINAL_DATA_FILES
from tools.notebook07_bundle import SOURCE_FILES as ORIGINAL_SOURCE_FILES

NOTEBOOK = ROOT / 'ipynb/09_comparison205.ipynb'
SINGLE_NOTEBOOKS = {
    'base': ROOT / 'ipynb/09_base_model_aime30_lcb175.ipynb',
    'e80': ROOT / 'ipynb/10_expert80_aime30_lcb175.ipynb',
}
OUTPUT = ROOT / 'dist/colab'
SOURCE_FILES = [*ORIGINAL_SOURCE_FILES, 'tools/comparison205.py']
DATA_FILES = list(ORIGINAL_DATA_FILES)


def build_artifacts(root=None, output=None):
    """명시된 파일만 재현 가능한 ZIP으로 만들고 동일 이름의 기존 결과를 검증한다."""
    root = ROOT if root is None else Path(root)
    output = OUTPUT if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for kind, paths in (('source', SOURCE_FILES), ('data', DATA_FILES)):
        payload = archive_bytes(root, paths)
        digest = hashlib.sha256(payload).hexdigest()
        target = output / f'putare09-{kind}-{digest[:12]}.zip'
        try:
            with target.open('xb') as stream:
                stream.write(payload)
        except FileExistsError:
            if colab_archive.sha256_file(target) != digest:
                raise ValueError(f'기존 ZIP 해시 불일치: {target}')
        result[kind] = {'path': target, 'sha256': digest}
    return result


def notebook_document(artifacts):
    """해시가 고정된 독립 부트스트랩과 실행 단계별 셀을 구성한다."""
    def cell(kind, name, source):
        value = {'cell_type': kind, 'id': 'putare09-' + name,
                 'metadata': {'putare09': name}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        return value

    for kind in ('source', 'data'):
        if not re.fullmatch(r'[0-9a-f]{64}', artifacts[kind]['sha256']):
            raise ValueError('ZIP SHA256이 올바르지 않습니다')
        expected = f"putare09-{kind}-{artifacts[kind]['sha256'][:12]}.zip"
        if artifacts[kind]['path'].name != expected:
            raise ValueError(f'{kind} ZIP 이름과 SHA256이 다릅니다')
    source_sha, data_sha = (artifacts[k]['sha256'] for k in ('source', 'data'))
    source_zip, data_zip = ('/content/drive/MyDrive/Putare/' + artifacts[k]['path'].name
                            for k in ('source', 'data'))
    settings = f'''# 2026-11-11: 각 비용 발생 단계를 명시적으로 켤 때만 실행한다.
SETUP_COLAB = False
DOWNLOAD_MODEL = False
RUN_PREFLIGHT = False
RUN_GENERATION = False
RUN_GRADING = False
EXPORT_RESULTS = False
DOWNLOAD_ARCHIVE = False
SOURCE_ZIP = {source_zip!r}
DATA_ZIP = {data_zip!r}
BASE_MODEL = '/content/models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf'
E80_MODEL = '/content/drive/MyDrive/models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf'
RUN_DIR = '/content/drive/MyDrive/Putare/results/09-comparison205-001'
ARCHIVE_PATH = '/content/putare09-results.zip'
'''
    archive_code = (inspect.getsource(colab_archive.sha256_file) + '\n\n' +
                    inspect.getsource(colab_archive.restore_archives))
    setup = f'''if SETUP_COLAB:
    _PUTARE_SETUP_READY = False
    import hashlib, json, os, shutil, sys, tempfile, zipfile
    from pathlib import Path, PurePosixPath
    from google.colab import drive
    drive.mount('/content/drive')
    if not os.path.ismount('/content/drive'):
        raise RuntimeError('Google Drive가 마운트되지 않았습니다')
    ROOT = Path('/content/Putare') / ({source_sha!r}[:12] + '-' + {data_sha!r}[:12])
    # 2026-11-11: 커널에 남은 다른 실험 버전의 모듈이 새 번들과 섞이지 않게 한다.
    for _name, _module in list(sys.modules.items()):
        if _name not in ('tools', 'moe_prune') and not _name.startswith(('tools.', 'moe_prune.')):
            continue
        _file = getattr(_module, '__file__', None)
        _paths = getattr(_module, '__path__', ()) if not _file else ()
        if (_file and not Path(_file).resolve().is_relative_to(ROOT.resolve())) or any(not Path(p).resolve().is_relative_to(ROOT.resolve()) for p in _paths):
            raise RuntimeError('다른 Putare 버전이 로드되었습니다. 커널을 다시 시작하세요: ' + _name)

{''.join('    ' + line if line.strip() else line for line in archive_code.splitlines(keepends=True))}
    restore_archives(ROOT, [(SOURCE_ZIP, {source_sha!r}), (DATA_ZIP, {data_sha!r})])
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    def _validate_run_directory():
        if not os.path.ismount('/content/drive'):
            raise RuntimeError('Google Drive가 연결되지 않았습니다')
        target = Path(RUN_DIR).resolve()
        drive_root = Path('/content/drive/MyDrive').resolve()
        if target == drive_root or not target.is_relative_to(drive_root):
            raise ValueError('실험 결과 경로는 MyDrive의 하위 폴더여야 합니다')
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=target) as probe:
            probe.write(b'putare-checkpoint')
            probe.flush()
            os.fsync(probe.fileno())
        return target
    _validate_run_directory()
    from tools.colab_setup import prepare_environment
    _PUTARE_SETUP_STATE = prepare_environment(ROOT, globals().get('_PUTARE_SETUP_STATE'))
    from tools import benchmark175, comparison205
    ENGINE = ROOT / '.cache/llama-build/bin/llama-expert-research'
    _PUTARE_SETUP_READY = True
    print('환경 준비 완료:', ROOT)
'''
    download_model = '''if DOWNLOAD_MODEL:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    print('원본 모델:', benchmark175.download_base_model(Path(BASE_MODEL)))
'''
    preflight = '''if RUN_PREFLIGHT:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _run = _validate_run_directory()
    print('입력 및 최대 컨텍스트 점검:', comparison205.prepare(
        _run, base_model=Path(BASE_MODEL), e80_model=Path(E80_MODEL), engine=ENGINE))
'''
    generation = '''if RUN_GENERATION:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _run = _validate_run_directory()
    print('생성:', comparison205.generate(
        _run, base_model=Path(BASE_MODEL), e80_model=Path(E80_MODEL), engine=ENGINE))
'''
    grading = '''if RUN_GRADING:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _run = _validate_run_directory()
    _report = comparison205.grade(
        _run, base_model=Path(BASE_MODEL), e80_model=Path(E80_MODEL), engine=ENGINE)
    print('완료:', _report.get('complete'), '영역별:', _report.get('domains'))
    print('문제별 비교:', _report.get('paired'), '효율:', _report.get('efficiency'))
    print('문제별 결과:', _run / 'comparison-report.json')
'''
    export = '''if EXPORT_RESULTS:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _run = _validate_run_directory()
    _archive = Path(ARCHIVE_PATH)
    _verified = comparison205.archive_results(_run, _archive)
    print('아카이브 검증:', _verified, '경로:', _archive)
'''
    download_archive = '''if DOWNLOAD_ARCHIVE:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _archive = Path(ARCHIVE_PATH)
    print('다운로드 전 검증:', comparison205.verify_archive(_archive))
    from google.colab import files
    files.download(str(_archive))
'''
    intro = '''# 베이스·80전문가 비교: AIME 30 + LiveCodeBench 175

`putare09-source-*.zip`과 `putare09-data-*.zip`을 `MyDrive/Putare`에 올리고,
80전문가 GGUF를 아래 설정의 Drive 경로에 준비합니다. 원본 GGUF는 별도로 준비하거나
설정의 다운로드 플래그를 켭니다. 단계별 플래그는 기본적으로 모두 꺼져 있습니다.

동일 RTX PRO 6000 GPU에서 205문제씩 한 번 실행합니다. 입력과 출력이 공유하는
컨텍스트 한도는 262,144토큰이며, 문제별 최대 생성량은 262,144에서 템플릿 포함
입력 토큰 수와 1토큰을 뺀 값입니다. Thinking 활성화, Q8 KV, Top-8, routing 계측을
사용합니다. 별도의 effort 단계는 지원되지 않아 설정하지 않습니다. 실제 생성량은
각 답변에서 확인합니다.

실행 폴더는 마운트된 MyDrive에 직접 생성되며 시작 마커도 추론 전에 기록합니다.
중단 시 같은 폴더·동일 GPU·동일 설정으로 재개하며, 이미 시작됐으나 답변이 없는
문제는 자동 재풀이하지 않습니다. Drive 용량 또는 쓰기에 문제가 있으면 실행을 중단합니다.
토큰별 routing 기록을 포함하므로 충분한 저장 공간이 필요합니다.

사전 점검을 통과한 후 생성, 채점, 아카이브 순서로 실행하세요. 채점한 생성 코드는
현재 런타임의 사용자 권한으로 실행됩니다. 보존된 ZIP을 PC로 내려받은 후 로컬에서
`tools.comparison205.verify_archive(경로)`를 실행해 파일별 해시를 확인해야 합니다.
'''
    cells = [cell('markdown', 'intro', intro), cell('code', 'settings', settings),
             cell('code', 'setup', setup), cell('code', 'download-model', download_model),
             cell('code', 'preflight', preflight), cell('code', 'generation', generation),
             cell('code', 'grading', grading), cell('code', 'export', export),
             cell('code', 'download-archive', download_archive)]
    return {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
            'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}


def _write_notebook(path, document_factory, metadata_key, artifacts):
    # 2026-11-11: 사용자가 바꾼 노트북은 보존하고 미수정 생성본만 새 번들로 갱신한다.
    path = Path(path)
    contents = json.dumps(document_factory(artifacts), ensure_ascii=False, indent=1) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FileExistsError('노트북이 심볼릭 링크입니다')
    try:
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(contents)
    except FileExistsError:
        original = path.read_text(encoding='utf-8')
        if original == contents:
            return path
        try:
            document = json.loads(original)
            setup = next(c for c in document['cells'] if c.get('metadata', {}).get(metadata_key) == 'setup')
            match = re.search(r"restore_archives\(ROOT, \[\(SOURCE_ZIP, '([0-9a-f]{64})'\), \(DATA_ZIP, '([0-9a-f]{64})'\)\]\)",
                              ''.join(setup['source']))
            if match is None:
                raise ValueError('이전 ZIP 해시 누락')
            previous = {kind: {'sha256': digest,
                        'path': Path(f'putare09-{kind}-{digest[:12]}.zip')}
                        for kind, digest in zip(('source', 'data'), match.groups())}
            if document != document_factory(previous):
                raise ValueError('기존 노트북이 수정되었습니다')
        except (KeyError, TypeError, ValueError, StopIteration) as error:
            raise FileExistsError('기존 09 노트북을 변경하지 않았습니다. 별도 경로에 생성하세요') from error
        temporary = None
        try:
            with tempfile.NamedTemporaryFile('w', encoding='utf-8', newline='\n', dir=path.parent,
                                             prefix='.putare09-', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(contents)
                stream.flush()
                os.fsync(stream.fileno())
            if path.is_symlink() or path.read_text(encoding='utf-8') != original:
                raise FileExistsError('노트북이 동시에 변경되어 갱신하지 않았습니다')
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return path


def write_notebook(artifacts):
    return _write_notebook(NOTEBOOK, notebook_document, 'putare09', artifacts)


def single_notebook_document(profile, artifacts):
    """2026-11-11: 각 세션이 상대 모델과 독립적으로 205문제를 완료하도록 구성한다."""
    if profile not in ('base', 'e80'):
        raise ValueError('알 수 없는 모델 프로필')
    original = notebook_document(artifacts)
    bootstrap = next(''.join(c['source']) for c in original['cells']
                     if c.get('metadata', {}).get('putare09') == 'setup')
    prefix = 'putare09' if profile == 'base' else 'putare10'

    def cell(kind, name, source):
        value = {'cell_type': kind, 'id': prefix + '-' + name,
                 'metadata': {'putare_single': name}, 'source': source.splitlines(keepends=True)}
        if kind == 'code':
            value.update(execution_count=None, outputs=[])
        return value

    path = ('/content/models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf' if profile == 'base' else
            '/content/drive/MyDrive/models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf')
    run_name = '09-base205-001' if profile == 'base' else '10-e80-205-001'
    archive_name = 'putare09-base-results.zip' if profile == 'base' else 'putare10-e80-results.zip'
    source_zip, data_zip = ('/content/drive/MyDrive/Putare/' + artifacts[k]['path'].name
                            for k in ('source', 'data'))
    settings = f'''# 2026-11-11: 모델별로 독립된 실행 경로와 명시적 단계 플래그를 사용한다.
SETUP_COLAB = False
DOWNLOAD_MODEL = False
RUN_PREFLIGHT = False
RUN_GENERATION = False
RUN_GRADING = False
EXPORT_RESULTS = False
DOWNLOAD_ARCHIVE = False
SOURCE_ZIP = {source_zip!r}
DATA_ZIP = {data_zip!r}
MODEL_PROFILE = {profile!r}
MODEL_PATH = {path!r}
RUN_DIR = '/content/drive/MyDrive/Putare/results/{run_name}'
ARCHIVE_PATH = '/content/{archive_name}'
'''
    if profile == 'e80':
        settings += '''RUN_COMPARE = False
BASE_RUN_DIR = '/content/drive/MyDrive/Putare/results/09-base205-001'
COMPARE_OUTPUT = '/content/drive/MyDrive/Putare/results/09-10-comparison-report.json'
'''
    download = ('''if DOWNLOAD_MODEL:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    print('원본 모델:', benchmark175.download_base_model(Path(MODEL_PATH)))
''' if profile == 'base' else '''if DOWNLOAD_MODEL:
    raise RuntimeError('80전문가 모델은 설정된 Drive 경로에 직접 준비하세요')
''')
    preflight = '''if RUN_PREFLIGHT:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _run = _validate_run_directory()
    print('입력 및 최대 컨텍스트 점검:', comparison205.prepare_single(
        _run, model=Path(MODEL_PATH), engine=ENGINE, model_profile=MODEL_PROFILE))
'''
    generation = '''if RUN_GENERATION:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _run = _validate_run_directory()
    print('생성:', comparison205.generate_single(
        _run, model=Path(MODEL_PATH), engine=ENGINE, model_profile=MODEL_PROFILE))
'''
    grading = '''if RUN_GRADING:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _run = _validate_run_directory()
    _report = comparison205.grade_single(
        _run, model=Path(MODEL_PATH), engine=ENGINE, model_profile=MODEL_PROFILE)
    print('완료:', _report.get('complete'), '영역별:', _report.get('domains'))
    print('문제별 결과는 실행 폴더에 보존됩니다:', _run)
'''
    export = '''if EXPORT_RESULTS:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _run = _validate_run_directory()
    _archive = Path(ARCHIVE_PATH)
    print('아카이브 검증:', comparison205.archive_single(_run, _archive), '경로:', _archive)
'''
    download_archive = next(''.join(c['source']) for c in original['cells']
                            if c.get('metadata', {}).get('putare09') == 'download-archive')
    comparison = '''if RUN_COMPARE:
    if not globals().get('_PUTARE_SETUP_READY', False):
        raise RuntimeError('먼저 환경 준비를 완료하세요')
    _run = _validate_run_directory()
    _report = comparison205.compare_runs(Path(BASE_RUN_DIR), _run, output_path=Path(COMPARE_OUTPUT))
    print('비교 완료:', _report.get('complete'), '영역별:', _report.get('domains'))
    print('문제별 비교:', _report.get('paired'), '효율:', _report.get('efficiency'))
    print('비교 결과:', COMPARE_OUTPUT)
'''
    name = '베이스 128전문가' if profile == 'base' else '80전문가'
    model_hint = ('베이스 모델은 선택적으로 `DOWNLOAD_MODEL`을 켜서 준비할 수 있습니다.' if profile == 'base'
                  else '80전문가 GGUF는 설정된 Drive 경로에 직접 준비하세요.')
    intro = f'''# {name} 모델: AIME 30 + LiveCodeBench 175

09·10번 노트북이 같은 소스·데이터 ZIP을 사용하지만, 이 노트북은 **{name} 모델만**
준비하고 별도 Colab 세션에서 205문제를 각각 한 번씩 풉니다. 상대 모델 파일이나 상대
실행 결과는 준비·생성·채점·보존에 필요하지 않습니다. 모든 실행 플래그는 기본적으로 꺼져 있습니다.

`putare09-source-*.zip`과 `putare09-data-*.zip`을 `MyDrive/Putare`에 올리고, 아래 모델
경로를 준비하세요. `SETUP_COLAB`을 켜 환경을 준비한 뒤 `RUN_PREFLIGHT`, `RUN_GENERATION`,
`RUN_GRADING`, `EXPORT_RESULTS`, `DOWNLOAD_ARCHIVE`를 필요한 순서대로 켜고 실행하세요.
{model_hint}

RTX PRO 6000에서 입력과 출력이 공유하는 컨텍스트는 262,144토큰이며,
문제별 최대 생성량은 262,144에서 템플릿 포함 입력 토큰 수와 1토큰을 뺀 값입니다.
Thinking 활성화, Q8 KV, Top-8, routing 계측을 사용합니다. 별도 effort 단계는 없습니다.

실행 폴더는 MyDrive에 직접 저장됩니다. 중단 시 이전 완료 답변을 보존하고 아직 시작하지
않은 문제만 재개합니다. 시작했으나 답변이 없는 문제는 자동으로 다시 풀지 않습니다.
Drive 저장 공간과 연결이 부족하면 실행을 중단합니다. 토큰별 routing 기록은 매우 클 수 있습니다.

각 모델의 본 실행을 마치고 나서만 별도 실행 결과를 비교합니다. 각 세션은 같은 종류의 GPU를
사용하지만 물리적 GPU UUID는 다를 수 있으며, 속도 비교에 이 사실을 명시합니다.
채점한 생성 코드는 현재 런타임의 사용자 권한으로 실행됩니다. ZIP을 PC로 내려받은 후
`tools.comparison205.verify_archive(경로)`로 파일별 해시를 확인하세요.
'''
    if profile == 'e80':
        intro += '''
비교 시 두 실행 폴더가 필요합니다. 실행 폴더 대신 ZIP만 있다면 검증한 ZIP을 풀어
각 `results` 폴더를 준비한 뒤 `comparison205.compare_runs(베이스폴더, 80전문가폴더)`를
호출할 수 있습니다. 비교할 Putare 작업 경로에도 같은 소스·데이터 ZIP이 복원되어 있어야
합니다. 이 비교 함수 자체는 GPU와 GGUF 없이 실행할 수 있습니다. 아래 노트북의 선택
비교 셀은 기존 Colab 환경 준비 후 사용합니다.
'''
    cells = [cell('markdown', 'intro', intro), cell('code', 'settings', settings),
             cell('code', 'setup', bootstrap), cell('code', 'download-model', download),
             cell('code', 'preflight', preflight), cell('code', 'generation', generation),
             cell('code', 'grading', grading), cell('code', 'export', export),
             cell('code', 'download-archive', download_archive)]
    if profile == 'e80':
        cells.append(cell('code', 'compare', comparison))
    return {'cells': cells, 'metadata': original['metadata'], 'nbformat': 4, 'nbformat_minor': 5}


def write_single_notebook(profile, artifacts):
    if profile not in SINGLE_NOTEBOOKS:
        raise ValueError('알 수 없는 모델 프로필')
    return _write_notebook(SINGLE_NOTEBOOKS[profile],
                           lambda data: single_notebook_document(profile, data),
                           'putare_single', artifacts)


if __name__ == '__main__':
    built = build_artifacts()
    for profile in ('base', 'e80'):
        print('노트북:', write_single_notebook(profile, built))
    for kind, item in built.items():
        print(kind, item['path'], item['sha256'])
