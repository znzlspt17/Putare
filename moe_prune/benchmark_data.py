"""2026-11-06: 공개 revision을 고정하고 답안과 정답을 분리한다."""
from __future__ import annotations

import hashlib
import ast
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import urllib.request

from .research import canonical_hash, read_jsonl, write_json, write_jsonl

LCB_COMMIT = '28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24'
AIME_REPO = 'MathArena/aime_2026'
LCB_REPO = 'livecodebench/code_generation_lite'
LCB_INDEX_SPACE = 'livecodebench/code_generation_samples'
ROOT = Path(__file__).resolve().parents[1]
GRADER_VENV = ROOT / '.cache/putare-lcb/venv'
GRADER_SOURCE = ROOT / '.cache/putare-lcb/livecodebench' / LCB_COMMIT


def select_v6_new(v5, v6, expected=175):
    """2026-11-06: release_v6의 누적 ID에서 v5를 빼고 중복·누락을 중단한다."""
    def ids(rows):
        result = [str(row['question_id']) for row in rows]
        if len(result) != len(set(result)):
            raise ValueError('LiveCodeBench question_id 중복')
        return set(result)
    old, current = ids(v5), ids(v6)
    if not old <= current:
        raise ValueError('v5가 v6에 포함되지 않습니다')
    selected = sorted((row for row in v6 if str(row['question_id']) not in old),
                      key=lambda row: str(row['question_id']))
    if len(selected) != expected:
        raise ValueError(f'v6 신규 문항 수 불일치: {len(selected)} != {expected}')
    return selected


def final_answer(text):
    # 2026-11-06: 추론 채널의 숫자나 코드 블록이 최종 답으로 채점되지 않게 한다.
    marker = '<|channel>thought'
    if marker in text:
        text = text.rsplit(marker, 1)[1]
        if '<channel|>' not in text:
            return ''
        text = text.split('<channel|>', 1)[1]
    if '<|channel>final' in text:
        text = text.rsplit('<|channel>final', 1)[1]
    return text.strip()


def aime_answer(text):
    found = re.findall(r'\\boxed\s*\{\s*(\d{1,3})\s*\}', text)
    return int(found[-1]) if found else None


def _open_public(url, timeout=60):
    # 인증 캐시와 프록시 인증정보를 자동 탐색하지 않는 공개 HTTPS 요청만 허용한다.
    request = urllib.request.Request(url, headers={'User-Agent': 'putare-benchmark/1'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def _public(url):
    with _open_public(url) as response:
        return response.read()


def _revision(repo):
    data = json.loads(_public('https://huggingface.co/api/datasets/' + repo))
    return data['sha'], data.get('cardData', {}).get('license')


def _lcb_rows(revision):
    """2026-11-06: v5 ID 메타데이터만 이용해 원본 비공개 테스트 4GB 중복 전송을 피한다."""
    base = f'https://huggingface.co/datasets/{LCB_REPO}/resolve/{revision}/'
    loader = _public(base + 'code_generation_lite.py')
    tree = ast.parse(loader.decode('utf-8'))
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'ALLOWED_FILES' for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError('LCB 공식 dataset loader의 release 정의가 다릅니다')
    allowed = ast.literal_eval(assignments[0].value)
    old_files = ['test.jsonl'] + [f'test{i}.jsonl' for i in range(2, 6)]
    if allowed.get('release_v5') != old_files or allowed.get('release_v6') != old_files + ['test6.jsonl']:
        raise ValueError('LCB v5/v6 파일 집합 정의가 다릅니다')
    index_revision = json.loads(_public('https://huggingface.co/api/spaces/' + LCB_INDEX_SPACE))['sha']
    index_url = f'https://huggingface.co/spaces/{LCB_INDEX_SPACE}/resolve/{index_revision}/problems.json'
    index_data = _public(index_url)
    index = json.loads(index_data)
    if not isinstance(index, list) or len(index) != 880:
        raise ValueError('공식 LiveCodeBench Explorer v5 ID index는 880개여야 합니다')
    ids = [str(row['question_id']) for row in index]
    if len(set(ids)) != 880:
        raise ValueError('LiveCodeBench Explorer v5 ID index 중복')
    print('LiveCodeBench v6 신규 데이터 다운로드 중 (약 134 MB)', flush=True)
    digest, added = hashlib.sha256(), []
    with _open_public(base + 'test6.jsonl', timeout=120) as response:
        for line in response:
            digest.update(line)
            if line.strip():
                added.append(json.loads(line))
    previous = [{'question_id': value} for value in ids]
    current = previous + added
    # 데이터 자체의 파일 정의와 독립 index를 둘 다 확인하고 출처 차이를 manifest에 남긴다.
    select_v6_new(previous, current)
    hashes = {'dataset_loader_sha256': hashlib.sha256(loader).hexdigest(),
              'v5_ids_source_space': LCB_INDEX_SPACE, 'v5_ids_source_revision': index_revision,
              'v5_ids_source_file': 'problems.json', 'v5_ids_source_sha256': hashlib.sha256(index_data).hexdigest(),
              'v5_ids_count': len(index), 'test6.jsonl': digest.hexdigest(),
              'release_v5_files': old_files, 'release_v6_files': old_files + ['test6.jsonl']}
    return previous, current, hashes


def _aime_rows(revision):
    import pyarrow.parquet as parquet
    url = f'https://huggingface.co/datasets/{AIME_REPO}/resolve/{revision}/data/train-00000-of-00001.parquet'
    raw = _public(url)
    rows = parquet.read_table(io.BytesIO(raw)).to_pylist()
    if len(rows) != 30:
        raise ValueError(f'AIME 2026 문항 수 불일치: {len(rows)}')
    return rows, hashlib.sha256(raw).hexdigest()


def grader_paths(root):
    root = Path(root).resolve()
    return root / '.cache/putare-lcb/venv', root / '.cache/putare-lcb/livecodebench' / LCB_COMMIT


def prepare_grader(root):
    """2026-11-09: 공개 채점기와 전용 환경은 사용자가 준비 셀을 켤 때만 만든다."""
    from tools.lcb_prepare import prepare
    return prepare(root)


def _grader_tool(root, *args, capture_output=False, timeout_s=None):
    from .grading import worker_environment
    venv, source = grader_paths(root)
    return subprocess.run([str(venv / 'bin/python'), '-I', '-B', str(Path(root).resolve() / 'tools/lcb_grade.py'),
                           *map(str, args), '--official-source', str(source)], env=worker_environment(),
                          check=True, capture_output=capture_output, text=True, encoding='utf-8', timeout=timeout_s)


def _official_prompts(root, rows):
    # 테스트와 정답을 공식 formatter에 전달하지 않는다.
    root = Path(root).resolve()
    (root / 'runs').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root / 'runs') as folder:
        folder = Path(folder)
        source, dest = folder / 'prompts-in.json', folder / 'prompts-out.json'
        write_json(source, [{'question_content': r['question_content'], 'starter_code': r['starter_code']} for r in rows])
        _grader_tool(root, '--prompts', '--input', source, '--output', dest)
        return json.loads(dest.read_text(encoding='utf-8'))


def official_identity(root, *, timeout_s=None):
    checked = _grader_tool(root, '--verify', capture_output=True, timeout_s=timeout_s)
    return json.loads(checked.stdout)


def grader_identity(root):
    root = Path(root).resolve()
    names = ('tools/lcb_grade.py', 'moe_prune/grading.py')
    return {**{name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names},
            'execution_mode': 'unisolated'}


def prepare_snapshot(destination, root):
    """AIME30과 v6 신규175를 변조 탐지 가능한 immutable snapshot으로 준비한다."""
    destination, root = Path(destination), Path(root)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('snapshot이 이미 있습니다. 새 경로를 사용하세요')
    aime_revision, aime_license = _revision(AIME_REPO)
    lcb_revision, lcb_license = _revision(LCB_REPO)
    official = official_identity(root)
    grader = grader_identity(root)
    aime, aime_hash = _aime_rows(aime_revision)
    v5, v6, files = _lcb_rows(lcb_revision)
    code = select_v6_new(v5, v6)
    prompts = _official_prompts(root, code)
    if len(prompts) != len(code):
        raise ValueError('공식 prompt 수가 문제 수와 다릅니다')
    records = []
    for row in aime:
        records.append({'id': f'aime2026:{row["problem_idx"]}', 'domain': 'math', 'dataset': 'aime2026',
                        'prompt': row['problem'] + '\n\nSolve the problem without tools. Give the final integer answer in \\boxed{...}.',
                        'reference': str(row['answer'])})
    for row, item in zip(code, prompts):
        records.append({'id': f'livecodebench:{row["question_id"]}', 'domain': 'code',
                        'dataset': 'livecodebench_v6_new', 'prompt': item['prompt'],
                        'system_prompt': item['system_prompt'], 'reference': '', 'lcb_problem': row})
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / 'records.jsonl'
    write_jsonl(path, records)
    manifest = {'schema_version': 1, 'official_lcb_commit': LCB_COMMIT, 'official_identity': official,
                'grader_identity': grader,
                'datasets': {'aime': {'repo': AIME_REPO, 'revision': aime_revision, 'license': aime_license,
                                      'source_sha256': aime_hash},
                             'lcb': {'repo': LCB_REPO, 'revision': lcb_revision, 'license': lcb_license,
                                     'source_sha256': files}},
                'counts': {'math': len(aime), 'code': len(code)},
                'record_ids': [row['id'] for row in records], 'records_hash': canonical_hash(records),
                'file_sha256': {'records': hashlib.sha256(path.read_bytes()).hexdigest()}}
    write_json(destination / 'manifest.json', manifest)
    return manifest


def load_snapshot(destination):
    destination = Path(destination)
    manifest = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
    path = destination / 'records.jsonl'
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['file_sha256']['records']:
        raise ValueError('benchmark snapshot 파일 해시 불일치')
    records = list(read_jsonl(path))
    if canonical_hash(records) != manifest['records_hash'] or [r['id'] for r in records] != manifest['record_ids']:
        raise ValueError('benchmark snapshot 내용 불일치')
    if (manifest.get('official_lcb_commit') != LCB_COMMIT or
            not isinstance(manifest.get('official_identity'), dict) or
            not isinstance(manifest.get('grader_identity'), dict)):
        raise ValueError('공식 채점기 식별 정보가 올바르지 않습니다')
    if len({r.get('id') for r in records}) != len(records):
        raise ValueError('snapshot 문제 ID 중복')
    counts = {name: sum(r.get('domain') == name for r in records) for name in ('math', 'code')}
    if counts != {'math': 30, 'code': 175} or manifest.get('counts') != counts or len(records) != 205:
        raise ValueError('benchmark snapshot에는 AIME 30 및 LiveCodeBench 175문항이 필요합니다')
    for row in records:
        if not all(isinstance(row.get(k), str) and row[k] for k in ('id', 'domain', 'dataset', 'prompt')):
            raise ValueError('문제의 필수 문자열 필드가 누락되었습니다')
        if row['domain'] == 'math':
            if not str(row.get('reference', '')).isdigit() or not 0 <= int(row['reference']) <= 999:
                raise ValueError('AIME 정답은 0–999 정수여야 합니다')
        elif not isinstance(row.get('lcb_problem'), dict) or not row.get('system_prompt'):
            raise ValueError('LiveCodeBench 공식 문제 또는 system prompt가 누락되었습니다')
    return records, manifest


def validate_id_proof(destination):
    """2026-11-06: 단축된 Explorer index가 같은 revision v5와 일치한 증거를 요구한다."""
    destination = Path(destination).resolve()
    manifest_path = destination / 'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    proof_path = destination.parent / (destination.name + '-id-validation.json')
    if not proof_path.is_file():
        raise FileNotFoundError(f'v5 원본 ID 검증이 필요합니다: {proof_path}')
    proof = json.loads(proof_path.read_text(encoding='utf-8'))
    lcb = manifest['datasets']['lcb']
    source = lcb['source_sha256']
    files = proof.get('v5_files', {})
    required_files = ['test.jsonl'] + [f'test{i}.jsonl' for i in range(2, 6)]
    with (destination / 'records.jsonl').open('rb') as stream:
        records_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    if (proof.get('status') != 'verified' or
            proof.get('snapshot_manifest_sha256') != hashlib.sha256(manifest_bytes).hexdigest() or
            proof.get('snapshot_records_sha256') != manifest['file_sha256']['records'] or
            records_hash != manifest['file_sha256']['records'] or
            proof.get('dataset_revision') != lcb['revision'] or
            proof.get('v5_index_revision') != source['v5_ids_source_revision'] or
            proof.get('v5_count') != 880 or proof.get('v6_new_count') != 175 or
            source['release_v5_files'] != required_files or set(files) != set(required_files) or
            any(type(files[name].get('rows')) is not int or files[name]['rows'] <= 0 or
                not re.fullmatch(r'[a-f0-9]{64}', files[name].get('sha256', '')) for name in required_files) or
            sum(files[name]['rows'] for name in required_files) != 880):
        raise ValueError('v5 원본 ID 검증 결과가 현재 snapshot과 다릅니다')
    return proof


def grading_cache_key(payload, official, grader, proof):
    return canonical_hash({'payload': payload, 'official_commit': LCB_COMMIT,
                           'official_identity': official, 'grader_identity': grader,
                           'id_validation_proof': canonical_hash(proof)})


def grade_snapshot(destination, generations, run_dir, root):
    from .execution_safety import execution_guard
    from .grading import GradeDeadline
    run_dir = Path(run_dir).resolve()
    if Path(generations).resolve() != run_dir / 'generations.jsonl':
        raise ValueError('생성 결과는 실행 폴더의 generations.jsonl이어야 합니다')
    identity = json.loads((run_dir / 'benchmark-identity.json').read_text(encoding='utf-8'))
    with execution_guard('grading', memory_config=identity.get('config')) as guard:
        return _grade_snapshot(destination, generations, run_dir, root, guard, GradeDeadline())


def _grade_snapshot(destination, generations, run_dir, root, guard, deadline):
    """2026-11-06: 채점기 실패를 모델 오답으로 바꾸지 않고 완료 결과만 요약한다."""
    from .base_benchmark import load_completed
    from .gguf import sha256
    records, manifest = load_snapshot(destination)
    root, run_dir, generations = Path(root).resolve(), Path(run_dir).resolve(), Path(generations).resolve()
    if generations != run_dir / 'generations.jsonl':
        raise ValueError('생성 결과는 실행 폴더의 generations.jsonl이어야 합니다')
    proof = validate_id_proof(destination)
    identity = json.loads((run_dir / 'benchmark-identity.json').read_text(encoding='utf-8'))
    status = json.loads((run_dir / 'benchmark-status.json').read_text(encoding='utf-8'))
    if identity.get('scope') != 'benchmark' or not status.get('complete') or status.get('scope') != 'benchmark':
        raise ValueError('완료된 전체 benchmark 실행만 채점할 수 있습니다')
    if (identity.get('snapshot_manifest') != sha256(Path(destination) / 'manifest.json') or
            identity.get('snapshot_records') != sha256(Path(destination) / 'records.jsonl')):
        raise ValueError('생성 당시 데이터 snapshot과 현재 데이터가 다릅니다')
    identity_hash = canonical_hash(identity)
    generated = list(read_jsonl(generations))
    by_id = {}
    for row in generated:
        if row['id'] in by_id:
            raise ValueError('generation ID 중복')
        key = hashlib.sha256(row['id'].encode('utf-8')).hexdigest()
        saved = load_completed(run_dir / 'answers' / f'{key}.json', row['id'], identity_hash, run_dir)
        if saved != row:
            raise ValueError('생성 목록과 무결성 검증된 답안이 다릅니다')
        by_id[row['id']] = row
    if set(by_id) != {row['id'] for row in records}:
        raise ValueError('생성이 완료되지 않아 전체 점수를 산출할 수 없습니다')
    official = official_identity(root, timeout_s=deadline.remaining())
    if official.get('source_digest') != manifest.get('official_identity', {}).get('source_digest'):
        raise ValueError('공식 채점기 source가 snapshot과 다릅니다')
    grader = grader_identity(root)
    if (manifest.get('grader_identity', {}).get('tools/lcb_grade.py') != grader['tools/lcb_grade.py'] or
            identity.get('grader_identity') != grader or identity.get('official_identity') != official):
        raise ValueError('현재 채점기 identity가 생성 당시와 다릅니다')
    from .grading import run as run_grade
    venv, official_source = grader_paths(root)
    results = []
    grade_dir = run_dir / 'grade-cache'
    grade_dir.mkdir(parents=True, exist_ok=True)
    for number, row in enumerate(records, 1):
        try:
            deadline.remaining()
        except TimeoutError as error:
            write_json(run_dir / 'grading-status.json', {'complete': False, 'id': row['id'],
                       'status': 'infrastructure_error', 'error_type': type(error).__name__, 'completed': len(results)})
            raise
        generation = by_id[row['id']]
        print(f'채점 {number}/{len(records)}: {row["id"]}', flush=True)
        if row['domain'] == 'math':
            answer = aime_answer(final_answer(generation.get('text', '')))
            results.append({'id': row['id'], 'domain': 'math', 'answer': answer,
                            'passed': answer is not None and answer == int(row['reference']),
                            'stop_reason': generation.get('stop_reason')})
            continue
        if row['domain'] != 'code':
            raise ValueError('알 수 없는 평가 도메인')
        payload = {'id': row['id'], 'problem': row['lcb_problem'], 'text': generation.get('text', '')}
        digest = grading_cache_key(payload, official, grader, proof)
        key = hashlib.sha256(row['id'].encode('utf-8')).hexdigest()
        cached = grade_dir / f'{key}.json'
        was_cached = cached.exists()
        if was_cached:
            stored = json.loads(cached.read_text(encoding='utf-8'))
            if stored['input_hash'] != digest:
                raise ValueError('기존 채점 결과의 입력 또는 공식 채점기 버전이 다릅니다')
            graded = stored['result']
        else:
            with tempfile.TemporaryDirectory(dir=run_dir) as temp:
                temporary = Path(temp)
                input_path, output_path = temporary / 'input.json', temporary / 'output.json'
                write_json(input_path, payload)
                try:
                    run_grade(official_source, venv, input_path, output_path,
                              memory_config=identity.get('config'), guard=guard, timeout_s=deadline.remaining())
                except Exception as error:
                    write_json(run_dir / 'grading-status.json', {'complete': False, 'id': row['id'],
                               'status': 'infrastructure_error', 'error_type': type(error).__name__})
                    raise
                if not output_path.is_file():
                    raise RuntimeError('공식 채점기가 결과를 생성하지 않았습니다')
                graded = json.loads(output_path.read_text(encoding='utf-8'))
        if graded.get('id') != row['id'] or type(graded.get('passed')) is not bool:
            raise ValueError(f'공식 채점 실패 또는 결과 불일치: {graded.get("error_type", "invalid_result")} {graded.get("location", [])}')
        if graded.get('pass_at_1') != float(graded['passed']):
            raise ValueError('공식 pass@1과 문제별 성공 여부가 다릅니다')
        if not was_cached:
            write_json(cached, {'input_hash': digest, 'result': graded})
        results.append({'id': row['id'], 'domain': 'code', 'passed': graded['passed'],
                        'pass_at_1': graded['pass_at_1'], 'test_count': graded['test_count'],
                        'details': graded['details'], 'stop_reason': generation.get('stop_reason')})
    expected = manifest['counts']
    summary = {'complete': True, 'counts': {}, 'limit_reached': sum(r['stop_reason'] == 'length' for r in results),
               'official_lcb_commit': LCB_COMMIT, 'official_identity': official, 'grader_identity': grader,
               'benchmark_identity_hash': identity_hash, 'id_validation_proof_hash': canonical_hash(proof)}
    for domain in ('math', 'code'):
        subset = [row for row in results if row['domain'] == domain]
        if len(subset) != expected[domain]:
            raise ValueError('도메인별 채점 개수가 snapshot과 다릅니다')
        passes = sum(row['passed'] for row in subset)
        summary['counts'][domain] = {'correct': passes, 'total': len(subset), 'accuracy': passes / len(subset)}
    write_jsonl(run_dir / 'grades.jsonl', results)
    write_json(run_dir / 'grades-summary.json', summary)
    return summary
