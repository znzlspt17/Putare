"""2026-11-11: 두 모델의 문제별 단일 시도를 보존하는 비교 실행기."""
from pathlib import Path
import json
import os
import subprocess
import tempfile
import hashlib
import platform
import re
import sys
import time
import zipfile
import uuid

from moe_prune.base_benchmark import build_request, validate_routing, verify_runtime
from moe_prune.benchmark_data import aime_answer, final_answer, load_snapshot, validate_id_proof
from moe_prune.gguf import GGUF, sha256
from moe_prune.research import canonical_hash, read_jsonl, spawn_native, write_json
from tools import benchmark175

CONTEXT_SIZE = 262144
SESSION_ID = uuid.uuid4().hex


def generation_budget(prompt_tokens):
    if type(prompt_tokens) is not int or prompt_tokens < 1 or prompt_tokens >= CONTEXT_SIZE - 1:
        raise ValueError('입력 토큰이 context 범위를 벗어났습니다')
    return CONTEXT_SIZE - prompt_tokens - 1


def reserve_case(directory, number, record_id):
    # 2026-11-11: 시작 기록을 먼저 영구화해 중단된 추론의 자동 재시도를 막는다.
    path = Path(directory) / 'started' / f'{number:03d}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open('x', encoding='utf-8') as stream:
            json.dump({'id': record_id, 'number': number}, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise RuntimeError(f'이미 시작된 문제입니다: {record_id}') from error
    return path


def compare_preflight(base, e80):
    if len(base) != len(e80):
        raise ValueError('문제 개수 불일치')
    for left, right in zip(base, e80):
        if (left['id'] != right['id'] or left['rendered_prompt'] != right['rendered_prompt']
                or left['tokens'] != right['tokens']):
            raise ValueError(f'입력 토큰 또는 템플릿 불일치: {left["id"]}')
        generation_budget(len(left['tokens']))
    return True


def config(engine):
    return {**benchmark175.config(engine), 'context_size': CONTEXT_SIZE, 'max_tokens': CONTEXT_SIZE - 1}


def make_request(record, model, engine, token_ids):
    c = config(engine)
    c['max_tokens'] = generation_budget(len(token_ids))
    request = build_request(record, model, (31, True), c)
    request['expected_prompt_tokens'] = token_ids
    return request


def tokenize_records(records, model, engine):
    """실제 native tokenizer만 실행해 모든 입력의 채팅 템플릿과 토큰을 얻는다."""
    # 2026-11-11: 별도 vocab-only 세션을 사용해 사전 검증이 평가 추론이 되지 않게 한다.
    values = []
    with tempfile.TemporaryDirectory(prefix='putare-tokenize-') as tmp:
        root = Path(tmp)
        with (root / 'stderr.log').open('wb') as log:
            proc = spawn_native([str(Path(engine).resolve()), '--session'], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=log, text=True, encoding='utf-8', bufsize=1)
            try:
                for number, record in enumerate(records):
                    request = {'model': str(Path(model).resolve()), 'mode': 'tokenize',
                               'prompt': record['prompt'], 'context_size': CONTEXT_SIZE,
                               'max_tokens': 0, 'enable_thinking': True, 'apply_chat_template': True,
                               'n_gpu_layers': 0, 'instrumentation': False}
                    if record.get('system_prompt'):
                        request['system_prompt'] = record['system_prompt']
                    req, out = root / f'{number}.request.json', root / f'{number}.out.jsonl'
                    req.write_text(json.dumps(request, ensure_ascii=False), encoding='utf-8')
                    proc.stdin.write(json.dumps({'request': str(req.resolve()), 'output': str(out.resolve())}) + '\n')
                    proc.stdin.flush()
                    response = proc.stdout.readline()
                    if not response or json.loads(response) != {'status': 'ok'}:
                        raise RuntimeError('native tokenizer 사전 검증 실패')
                    rows = out.read_text(encoding='utf-8').splitlines()
                    if len(rows) != 1:
                        raise ValueError('native tokenizer 결과 형식 오류')
                    row = json.loads(rows[0])
                    if row.get('type') != 'result' or not isinstance(row.get('tokens'), list):
                        raise ValueError('native tokenizer 결과 누락')
                    values.append({'id': record['id'], 'tokens': row['tokens'],
                                   'rendered_prompt': row['rendered_prompt']})
            finally:
                if proc.stdin:
                    proc.stdin.close()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=15)
                if proc.stdout:
                    proc.stdout.close()
            if proc.returncode:
                raise RuntimeError('native tokenizer 종료 실패')
    return values


def dry_run(records, base_model, e80_model, engine):
    """양 모델의 실제 토큰을 비교하고 410개 생성 요청을 추론 없이 구성한다."""
    if len(records) != 205 or len({r['id'] for r in records}) != 205:
        raise ValueError('205개 고유 문제가 필요합니다')
    if sum(r['domain'] == 'math' for r in records) != 30 or sum(r['domain'] == 'code' for r in records) != 175:
        raise ValueError('AIME 30과 LCB 175가 필요합니다')
    base = tokenize_records(records, base_model, engine)
    e80 = tokenize_records(records, e80_model, engine)
    compare_preflight(base, e80)
    requests = {}
    for profile, model, tokens in (('base', base_model, base), ('e80', e80_model, e80)):
        requests[profile] = [make_request(record, model, engine, row['tokens'])
                             for record, row in zip(records, tokens)]
    for a, b in zip(requests['base'], requests['e80']):
        if {k: v for k, v in a.items() if k != 'model'} != {k: v for k, v in b.items() if k != 'model'}:
            raise ValueError('양 모델의 공통 생성 요청이 다릅니다')
    return {'base_tokens': base, 'e80_tokens': e80, 'requests': requests}


def checked_gpu():
    from tools.native.gpu_memory import NvmlDevice
    device = NvmlDevice()
    try:
        total, _ = device.sample()
        if 'RTX PRO 6000' not in device.name or 'Blackwell' not in device.name:
            raise RuntimeError(f'RTX PRO 6000 Blackwell이 아닙니다: {device.name}')
        major, minor = device.nvml.nvmlDeviceGetCudaComputeCapability(device.handle)
        driver = device.nvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        return {'name': device.name, 'uuid': device.uuid, 'total_bytes': total,
                'compute_capability': f'{major}.{minor}', 'driver': str(driver)}
    finally:
        device.close()


def ensure_gpu(expected):
    actual = checked_gpu()
    if actual != expected:
        raise RuntimeError('GPU 또는 드라이버가 실험 도중 변경되었습니다')
    return actual


def source_identity(records, manifest, base_model, e80_model, engine, gpu):
    # 2026-11-11: 모델, 데이터, 빌드 및 프롬프트의 출처를 재개 시에도 동일하게 묶는다.
    _, base_id = benchmark175.source_identity(base_model, engine, model_profile='base')
    _, e80_id = benchmark175.source_identity(e80_model, engine, model_profile='e80')
    template = [GGUF(path).metadata.get('tokenizer.chat_template') for path in (base_model, e80_model)]
    if not template[0] or template[0] != template[1]:
        raise ValueError('모델 chat template 불일치')
    provenance = json.loads(benchmark175.PERMUTATION.read_text(encoding='utf-8'))
    selection = [row['id'] for row in provenance['profile']['problems']]
    if len(selection) != 100 or len(set(selection)) != 100:
        raise ValueError('선정 프로파일 100문제 ID 오류')
    return {'base': base_id, 'e80': e80_id, 'gpu': gpu, 'record_ids': [r['id'] for r in records],
            'snapshot_manifest': sha256(benchmark175.SNAPSHOT / 'manifest.json'),
            'snapshot_id_proof': canonical_hash(validate_id_proof(benchmark175.SNAPSHOT)),
            'chat_template_sha256': hashlib.sha256(template[0].encode('utf-8')).hexdigest(),
            'comparison_source_sha256': sha256(Path(__file__)), 'python': sys.version,
            'platform': platform.platform(), 'selection_ids': selection,
            'official_identity': manifest['official_identity'],
            'config': config(engine), 'effort': None,
            'limitations': ['routing 계측 포함 속도', 'GPU 메모리는 표본이며 세션 누적 최대값',
                            'CUDA fusion 요청은 개별 커널 fusion 증명이 아님']}


def _write_once(path, content):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != content:
            raise ValueError(f'기존 실험 기록 불일치: {path}')
        return
    write_json(path, content)


def _durable_answer(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('기존 답안을 덮어쓰지 않습니다')
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            raise FileExistsError('기존 답안을 덮어쓰지 않습니다')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _validate_result(directory, job, request, result, experts):
    folder = Path(directory) / 'jobs' / job
    if (result.get('actual_context_size') != CONTEXT_SIZE
            or result.get('enable_thinking') is not True
            or result.get('apply_chat_template') is not True
            or result.get('instrumentation_level') != 'routing'
            or result.get('masked_layers') or result.get('profiled_layers') != list(range(30))
            or result.get('stop_reason') not in ('eog', 'length')
            or result.get('prompt_tokens') != len(request['expected_prompt_tokens'])
            or result.get('rendered_prompt') != request['expected_rendered_prompt']
            or (result.get('stop_reason') == 'length' and len(result.get('token_ids', [])) != request['max_tokens'])):
        raise ValueError('실제 생성 결과와 사전 검증 조건 불일치')
    observed = verify_runtime(result, (folder / 'stderr.log').read_text(encoding='utf-8', errors='replace'), request)
    if observed.get('offloaded_layers') != {'gpu': 31, 'total': 31}:
        raise ValueError('GPU 전체 적재 미확인')
    buffers = observed.get('kv_buffers_mib') or []
    if not buffers or any(not re.fullmatch(r'CUDA\d+', row['backend'], re.IGNORECASE) for row in buffers):
        raise ValueError('KV 캐시의 전체 GPU 적재 미확인')
    rows = validate_routing(read_jsonl(folder / 'events.jsonl'), result, experts=experts)
    if rows != result.get('routing_records'):
        raise ValueError('라우팅 관측 개수 불일치')
    return observed


def _native_request(request):
    # expected_rendered_prompt는 Python 검증 전용이며 엔진 입력에는 필요하지 않다.
    return {k: v for k, v in request.items() if k != 'expected_rendered_prompt'}


def prepare(directory, *, base_model, e80_model, engine):
    """토크나이저 사전 검사와 양 모델 최대 context 할당을 수행한다."""
    from moe_prune.resident import ResidentExperiment
    directory, engine = Path(directory), Path(engine).resolve()
    models = {'base': Path(base_model).resolve(), 'e80': Path(e80_model).resolve()}
    records, manifest = load_snapshot(benchmark175.SNAPSHOT)
    gpu = checked_gpu()
    identity = source_identity(records, manifest, models['base'], models['e80'], engine, gpu)
    directory.mkdir(parents=True, exist_ok=True)
    _write_once(directory / 'identity.json', identity)
    dry = dry_run(records, models['base'], models['e80'], engine)
    _write_once(directory / 'dry-run.json', dry)
    warm_record = {'id': 'warmup', 'prompt': 'Reply with the word ready.'}
    warm = tokenize_records([warm_record], models['base'], engine)
    warm_e80 = tokenize_records([warm_record], models['e80'], engine)
    compare_preflight(warm, warm_e80)
    for profile, experts in (('base', 128), ('e80', 80)):
        ensure_gpu(gpu)
        root = directory / 'preflight' / profile
        done = root / 'verified.json'
        if done.exists():
            saved = json.loads(done.read_text(encoding='utf-8'))
            if saved.get('identity_hash') != canonical_hash(identity):
                raise ValueError('기존 사전 점검 신원 불일치')
            continue
        request = make_request(warm_record, models[profile], engine, warm[0]['tokens'])
        request['max_tokens'] = 0
        request['expected_rendered_prompt'] = warm[0]['rendered_prompt']
        with ResidentExperiment(root, config(engine)) as exp:
            result = exp.run('allocation-check', _native_request(request))
            observed = _validate_result(root, 'allocation-check', request, result, experts)
        ensure_gpu(gpu)
        _write_once(done, {'identity_hash': canonical_hash(identity), 'gpu': gpu, 'observed': observed,
                           'measurement': result['measurement'], 'input': 'non-benchmark input'})
    _write_once(directory / 'prepared.json', {'identity_hash': canonical_hash(identity),
                                              'dry_run_hash': canonical_hash(dry), 'requests': 410})
    return {'ready': True, 'requests': 410, 'gpu': gpu}


def case_state(directory, number, record_id):
    root = Path(directory)
    started = root / 'started' / f'{number:03d}.json'
    answer = root / 'answers' / f'{number:03d}.json'
    if answer.exists() and not started.exists():
        raise ValueError('시작 기록 없는 답안입니다')
    if not started.exists():
        return 'unstarted'
    marker = json.loads(started.read_text(encoding='utf-8'))
    if marker != {'id': record_id, 'number': number}:
        raise ValueError('시작 기록의 문제 ID 불일치')
    return 'complete' if answer.exists() else 'interrupted'


def grade_math(text, reference):
    answer = aime_answer(final_answer(text))
    return answer is not None and answer == int(reference)


def grade_cached(path, answer, compute):
    path = Path(path)
    if path.exists():
        verdict = json.loads(path.read_text(encoding='utf-8'))
        if verdict.get('answer_sha256') != answer['answer_sha256']:
            raise ValueError('기존 채점의 답안 해시가 다릅니다')
        if type(verdict.get('passed')) is not bool:
            raise ValueError('기존 채점 결과 형식 오류')
        if verdict.get('verdict_sha256') != canonical_hash({k: v for k, v in verdict.items() if k != 'verdict_sha256'}):
            raise ValueError('기존 채점 결과 해시 불일치')
        return verdict
    passed = compute()
    if type(passed) is not bool:
        raise ValueError('채점 결과는 boolean이어야 합니다')
    verdict = {'answer_sha256': answer['answer_sha256'], 'passed': passed}
    verdict['verdict_sha256'] = canonical_hash(verdict)
    write_json(path, verdict)
    return verdict


def efficiency_summary(answers):
    # 2026-11-11: 채널 분리 불확실성은 추정하지 않고 전체 토큰을 미분류로 남긴다.
    channels = {'thinking': 0, 'final': 0, 'control': 0, 'unclassified': 0}
    durations = {'prefill_s': 0.0, 'decode_s': 0.0, 'wall_s': 0.0}
    gpu_peaks = []
    for answer in answers:
        generated = answer['generated_tokens']
        counts = answer.get('channel_tokens') or {}
        if (all(type(counts.get(key)) is int and counts[key] >= 0 for key in channels)
                and sum(counts[key] for key in channels) == generated):
            for key in channels:
                channels[key] += counts[key]
        else:
            channels['unclassified'] += generated
        for key in ('prefill_s', 'decode_s'):
            value = answer['timings'].get(key)
            if value is None:
                raise ValueError(f'시간 측정 누락: {key}')
            durations[key] += value
        durations['wall_s'] += answer['measurement']['wall_s']
        gpu = answer['measurement'].get('os_gpu_memory', {})
        if gpu.get('scope') == 'session_to_date' and gpu.get('peak_bytes') is not None:
            gpu_peaks.append(gpu['peak_bytes'])
    generated = sum(row['generated_tokens'] for row in answers)
    return {'count': len(answers), 'generated_tokens': generated, 'channel_tokens': channels,
            'length_stops': sum(row['stop_reason'] == 'length' for row in answers),
            'missing_final_answers': sum(not row['final_answer_present'] for row in answers),
            **durations, 'mean_decode_tokens_per_second': generated / durations['decode_s'] if durations['decode_s'] else None,
            'gpu_memory_session_peak_bytes': max(gpu_peaks) if gpu_peaks else None,
            'gpu_memory_scope': 'session_to_date; sampled, not per-problem or exact peak'}


def comparison_report(records, base, e80, selection_ids):
    pairs = {'both_pass': 0, 'base_only': 0, 'e80_only': 0, 'both_fail': 0}
    domains = {domain: {'total': 0, 'base_pass': 0, 'e80_pass': 0} for domain in ('math', 'code')}
    overlap = {k: {'total': 0, 'base_pass': 0, 'e80_pass': 0} for k in ('used', 'unused')}
    selected = set(selection_ids)
    cases = []
    for record in records:
        rid, domain = record['id'], record['domain']
        a, b = base[rid], e80[rid]
        av, bv = a['passed'], b['passed']
        if type(av) is not bool or type(bv) is not bool:
            raise ValueError('정답 여부 누락')
        pair = 'both_pass' if av and bv else 'base_only' if av else 'e80_only' if bv else 'both_fail'
        pairs[pair] += 1
        row = domains[domain]
        row['total'] += 1
        row['base_pass'] += av
        row['e80_pass'] += bv
        if domain == 'code':
            row = overlap['used' if rid in selected else 'unused']
            row['total'] += 1
            row['base_pass'] += av
            row['e80_pass'] += bv
        cases.append({'id': rid, 'domain': domain, 'selection_used': rid in selected,
                      'base': a, 'e80': b, 'pair': pair})
    for category in (*domains.values(), *overlap.values()):
        for profile in ('base', 'e80'):
            category[profile + '_percentage'] = (100 * category[profile + '_pass'] / category['total']
                                                  if category['total'] else None)
    efficiency = {}
    by_domain = {domain: {} for domain in domains}
    for profile, values in (('base', base), ('e80', e80)):
        rows = [values[r['id']] for r in records]
        if all('generated_tokens' in row for row in rows):
            efficiency[profile] = efficiency_summary(rows)
            for domain in domains:
                subset = [values[r['id']] for r in records if r['domain'] == domain]
                by_domain[domain][profile] = efficiency_summary(subset)
    return {'complete': True, 'domains': domains, 'paired': pairs, 'selection_overlap': overlap,
            'efficiency': efficiency, 'efficiency_by_domain': by_domain,
            'cases': cases, 'limitations': ['문제당 단일 시도', '라우팅 계측 포함 속도',
                                             '세션 누적 GPU 메모리 표본 최대값']}


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or 'manifest.json' not in names:
            raise ValueError('아카이브 목록 오류')
        manifest = json.loads(archive.read('manifest.json'))
        if set(names) != set(manifest) | {'manifest.json'}:
            raise ValueError('아카이브 목록 불일치')
        for name, expected in manifest.items():
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError(f'아카이브 해시 불일치: {name}')
    return {'files_verified': len(manifest), 'sha256': sha256(path)}


def verified_answer(directory, number, record_id, identity_hash):
    directory = Path(directory)
    if case_state(directory, number, record_id) != 'complete':
        raise ValueError('완료되지 않은 답안입니다')
    answer = json.loads((directory / 'answers' / f'{number:03d}.json').read_text(encoding='utf-8'))
    content = {k: v for k, v in answer.items() if k != 'answer_sha256'}
    if (answer.get('id') != record_id or answer.get('identity_hash') != identity_hash
            or answer.get('answer_sha256') != canonical_hash(content)):
        raise ValueError('답안 해시 또는 실행 신원 불일치')
    path = directory / 'routing' / f'{number:03d}.jsonl.gz'
    if path.is_symlink() or sha256(path) != answer['routing_integrity']['compressed_sha256']:
        raise ValueError('라우팅 압축 해시 불일치')
    import gzip
    with gzip.open(path, 'rb') as stream:
        if benchmark175._digest_stream(stream) != answer['routing_integrity']['uncompressed_sha256']:
            raise ValueError('라우팅 원본 해시 불일치')
    native = benchmark175.native_result(path)
    if (native != {k: v for k, v in answer['result'].items() if k != 'measurement'}
            or native.get('text') != answer['text']):
        raise ValueError('라우팅 원본의 native 결과와 답안이 다릅니다')
    return answer


def _answer_entry(answer, verdict):
    result = answer['result']
    return {'passed': verdict['passed'], 'generated_tokens': len(result['token_ids']),
            'channel_tokens': result.get('channel_tokens'),
            'final_answer_present': bool(final_answer(answer['text'])),
            'stop_reason': result['stop_reason'], 'timings': result['timings'],
            'measurement': result['measurement'], 'answer_sha256': answer['answer_sha256']}


def _load_timings(directory):
    result = {}
    for profile in ('base', 'e80'):
        paths = sorted((Path(directory) / profile / 'jobs').glob('warmup-*/events.jsonl'))
        values = []
        for path in paths:
            rows = [row for row in read_jsonl(path) if row.get('type') == 'result']
            if len(rows) != 1 or 'load_s' not in rows[0].get('timings', {}):
                raise ValueError('워밍업 모델 로딩 시간 누락')
            values.append(rows[0]['timings']['load_s'])
        result[profile] = values
    return result


def verified_report(directory):
    """아카이브하기 전에 410개 답안 및 채점 결속과 집계를 재검증한다."""
    directory = Path(directory)
    status = json.loads((directory / 'grading-status.json').read_text(encoding='utf-8'))
    if status != {'complete': True, 'completed': 410, 'total': 410, 'status': 'graded'}:
        raise ValueError('410개 채점이 완료되지 않았습니다')
    identity = json.loads((directory / 'identity.json').read_text(encoding='utf-8'))
    records, _ = load_snapshot(benchmark175.SNAPSHOT)
    if (identity.get('record_ids') != [r['id'] for r in records]
            or identity.get('snapshot_manifest') != sha256(benchmark175.SNAPSHOT / 'manifest.json')):
        raise ValueError('결과 데이터 신원 불일치')
    values = {}
    for profile in ('base', 'e80'):
        root = directory / profile
        values[profile] = {}
        for number, record in enumerate(records, 1):
            answer = verified_answer(root, number, record['id'], canonical_hash(identity))
            verdict_path = root / 'grading' / f'{number:03d}.json'
            verdict = grade_cached(verdict_path, answer, lambda: (_ for _ in ()).throw(ValueError('채점 누락')))
            values[profile][record['id']] = _answer_entry(answer, verdict)
    expected = comparison_report(records, values['base'], values['e80'], identity['selection_ids'])
    expected['identity_hash'] = canonical_hash(identity)
    expected['model_load_s'] = _load_timings(directory)
    stored = json.loads((directory / 'comparison-report.json').read_text(encoding='utf-8'))
    if stored != expected:
        raise ValueError('비교 보고서가 답안 및 채점 결과와 다릅니다')
    return stored


def _check_prepared(directory, base_model, e80_model, engine):
    directory = Path(directory)
    records, manifest = load_snapshot(benchmark175.SNAPSHOT)
    gpu = checked_gpu()
    identity = source_identity(records, manifest, base_model, e80_model, engine, gpu)
    if json.loads((directory / 'identity.json').read_text(encoding='utf-8')) != identity:
        raise ValueError('현재 모델·엔진·GPU 또는 데이터가 사전 점검과 다릅니다')
    dry = json.loads((directory / 'dry-run.json').read_text(encoding='utf-8'))
    prepared = json.loads((directory / 'prepared.json').read_text(encoding='utf-8'))
    if prepared != {'identity_hash': canonical_hash(identity), 'dry_run_hash': canonical_hash(dry), 'requests': 410}:
        raise ValueError('410개 요청 사전 점검이 완전하지 않습니다')
    compare_preflight(dry['base_tokens'], dry['e80_tokens'])
    return records, identity, dry


def generate(directory, *, base_model, e80_model, engine):
    """최대 한 번 시작하며 불완전 시도는 보존하고 미시작 문제만 실행한다."""
    from moe_prune.resident import ResidentExperiment
    directory, engine = Path(directory), Path(engine).resolve()
    models = {'base': Path(base_model).resolve(), 'e80': Path(e80_model).resolve()}
    records, identity, dry = _check_prepared(directory, models['base'], models['e80'], engine)
    identity_hash, gpu = canonical_hash(identity), identity['gpu']
    for profile, experts in (('base', 128), ('e80', 80)):
        root = directory / profile
        states = [case_state(root, i, r['id']) for i, r in enumerate(records, 1)]
        for i, (record, state) in enumerate(zip(records, states), 1):
            if state == 'complete':
                verified_answer(root, i, record['id'], identity_hash)
        pending = [i for i, state in enumerate(states, 1) if state == 'unstarted']
        if pending:
            ensure_gpu(gpu)
            warm_record = {'id': 'warmup', 'prompt': 'Reply with the word ready.'}
            warm = tokenize_records([warm_record], models[profile], engine)[0]
            request = make_request(warm_record, models[profile], engine, warm['tokens'])
            request['max_tokens'] = 1
            request['expected_rendered_prompt'] = warm['rendered_prompt']
            with ResidentExperiment(root, config(engine)) as exp:
                name = 'warmup-' + uuid.uuid4().hex
                result = exp.run(name, _native_request(request))
                _validate_result(root, name, request, result, experts)
                for number in pending:
                    record = records[number - 1]
                    ensure_gpu(gpu)
                    # 2026-11-11: dispatch 전에 같은 영구 디렉터리에 시작을 기록해 재풀이를 봉쇄한다.
                    reserve_case(root, number, record['id'])
                    job = f'problem-{number:03d}'
                    request = dict(dry['requests'][profile][number - 1])
                    request['expected_rendered_prompt'] = dry[profile + '_tokens'][number - 1]['rendered_prompt']
                    try:
                        result = exp.run(job, _native_request(request))
                        observed = _validate_result(root, job, request, result, experts)
                        ensure_gpu(gpu)
                        raw = root / 'jobs' / job / 'events.jsonl'
                        target = root / 'routing' / f'{number:03d}.jsonl.gz'
                        integrity = benchmark175.compress_events(raw, target)
                        with target.open('r+b') as stream:
                            stream.flush()
                            os.fsync(stream.fileno())
                        exp.check()
                        answer = {'id': record['id'], 'identity_hash': identity_hash, 'text': result['text'],
                                  'result': result, 'gpu': gpu, 'observed_runtime': observed,
                                  'routing_integrity': integrity, 'job': job}
                        answer['answer_sha256'] = canonical_hash(answer)
                        _durable_answer(root / 'answers' / f'{number:03d}.json', answer)
                        verified_answer(root, number, record['id'], identity_hash)
                        raw.unlink()
                        print(f'{profile} {number}/205 {record["id"]}', flush=True)
                    except BaseException as error:
                        write_json(root / 'errors' / f'{number:03d}.json',
                                   {'id': record['id'], 'status': 'infrastructure_error',
                                    'error_type': type(error).__name__, 'message': str(error)})
                        raise
        states = [case_state(root, i, r['id']) for i, r in enumerate(records, 1)]
        write_json(root / 'status.json', {'complete': all(s == 'complete' for s in states),
                   'completed': states.count('complete'), 'interrupted': states.count('interrupted'), 'total': 205})
        if any(s != 'complete' for s in states):
            raise RuntimeError(f'{profile} 미완료: 중단된 문제는 자동 재풀이하지 않습니다')
    return {'complete': True, 'answers': 410}


def _grade_impl(directory, *, base_model, e80_model, engine):
    """보존된 답안만 고정된 채점기로 평가한다. 인프라 실패는 오답으로 처리하지 않는다."""
    from moe_prune.execution_safety import execution_guard
    from moe_prune.grading import GradeDeadline, run, worker_environment
    from tools.lcb_prepare import COMMIT
    directory, engine = Path(directory), Path(engine).resolve()
    records, identity, _ = _check_prepared(directory, Path(base_model).resolve(), Path(e80_model).resolve(), engine)
    source = benchmark175.ROOT / '.cache/putare-lcb/livecodebench' / COMMIT
    venv = benchmark175.ROOT / '.cache/putare-lcb/venv'
    adapter = benchmark175.ROOT / 'tools/lcb_grade.py'
    snapshot_manifest = json.loads((benchmark175.SNAPSHOT / 'manifest.json').read_text(encoding='utf-8'))
    if sha256(adapter) != snapshot_manifest['grader_identity']['tools/lcb_grade.py']:
        raise ValueError('채점 adapter 해시 불일치')
    with execution_guard('grading', memory_config=config(engine)) as guard:
        deadline = GradeDeadline()
        probe = subprocess.run([str(venv / 'bin/python'), '-I', '-B', str(adapter), '--verify',
                                '--official-source', str(source)], env=worker_environment(),
                               check=True, capture_output=True, text=True, timeout=deadline.remaining())
        if json.loads(probe.stdout)['source_digest'] != identity['official_identity']['source_digest']:
            raise ValueError('공식 채점기 해시 불일치')
        data = {}
        completed = 0
        for profile in ('base', 'e80'):
            root = directory / profile
            data[profile] = {}
            for number, record in enumerate(records, 1):
                answer = verified_answer(root, number, record['id'], canonical_hash(identity))
                def compute():
                    if record['domain'] == 'math':
                        return grade_math(answer['text'], record['reference'])
                    payload = {'id': record['id'], 'problem': record['lcb_problem'], 'text': answer['text']}
                    input_path = root / 'grade-inputs' / f'{number:03d}.json'
                    _write_once(input_path, payload)
                    target = root / 'grading-raw' / f'{number:03d}-{uuid.uuid4().hex}.json'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    graded = run(source, venv, input_path, target, adapter=adapter,
                                 guard=guard, timeout_s=deadline.remaining())
                    if graded.get('id') != record['id'] or type(graded.get('passed')) is not bool:
                        raise ValueError('공식 채점 결과 불일치')
                    if graded.get('pass_at_1') != float(graded['passed']):
                        raise ValueError('공식 채점 pass@1 불일치')
                    return graded['passed']
                verdict = grade_cached(root / 'grading' / f'{number:03d}.json', answer, compute)
                data[profile][record['id']] = _answer_entry(answer, verdict)
                completed += 1
                write_json(directory / 'grading-status.json', {'complete': False, 'completed': completed,
                           'total': 410, 'status': 'grading', 'last_id': record['id'], 'profile': profile})
        report = comparison_report(records, data['base'], data['e80'], identity['selection_ids'])
        report['identity_hash'] = canonical_hash(identity)
        report['model_load_s'] = _load_timings(directory)
        write_json(directory / 'comparison-report.json', report)
        write_json(directory / 'grading-status.json', {'complete': True, 'completed': 410,
                   'total': 410, 'status': 'graded'})
    return report


def grade(directory, *, base_model, e80_model, engine):
    directory = Path(directory)
    try:
        return _grade_impl(directory, base_model=base_model, e80_model=e80_model, engine=engine)
    except BaseException as error:
        if directory.is_dir():
            status_path = directory / 'grading-status.json'
            previous = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
            write_json(status_path, {'complete': False, 'completed': previous.get('completed', 0),
                       'total': 410, 'status': 'infrastructure_error', 'error_type': type(error).__name__,
                       'message': str(error)})
        raise


def _artifact_paths(directory, *, single=False):
    # 2026-11-11: 사용자 폴더를 무차별 보관하지 않도록 생성 경로만 허용하고 먼저 전체 이름을 검사한다.
    root_files = {'identity.json', 'dry-run.json', 'prepared.json', 'grading-status.json', 'comparison-report.json'}
    fixed_profile = {'status.json', 'manifest.json'}
    number_json = re.compile(r'^\d{3}\.json$')
    job = re.compile(r'^(?:problem-\d{3}|warmup-[0-9a-f]{32})$')
    session = re.compile(r'^native-[A-Za-z0-9_-]+\.log$')
    job_files = {'request.json', 'identity.json', 'measurement.json', 'stderr.log', 'stderr-failure.log', 'events.jsonl'}

    def allowed(parts, directory_entry):
        if single:
            if len(parts) == 1:
                if directory_entry and parts[0] in {'checks', 'session-checks'}:
                    return True
                if not directory_entry and parts[0] in root_files | {'single-report.json', 'status.json', 'manifest.json'}:
                    return True
            if parts[0] == 'session-checks' and len(parts) == 2:
                return not directory_entry and bool(re.fullmatch(r'[0-9a-f]{32}\.json', parts[1]))
            if parts[0] == 'checks':
                if len(parts) == 2:
                    return directory_entry and bool(re.fullmatch(r'[0-9a-f]{32}', parts[1]))
                if len(parts) >= 3 and re.fullmatch(r'[0-9a-f]{32}', parts[1]):
                    return allowed_combined(('preflight', 'base', *parts[2:]), directory_entry)
                return False
            return allowed_profile(parts, directory_entry)
        return allowed_combined(parts, directory_entry)

    def allowed_profile(parts, directory_entry):
        return allowed_combined(('base', *parts), directory_entry)

    def allowed_combined(parts, directory_entry):
        if len(parts) == 1:
            return parts[0] in ({'base', 'e80', 'preflight'} if directory_entry else root_files)
        if parts[0] == 'preflight':
            if len(parts) == 2:
                return directory_entry and parts[1] in {'base', 'e80'}
            if parts[1] not in {'base', 'e80'}:
                return False
            tail = parts[2:]
            if len(tail) == 1:
                return tail[0] in ({'jobs', 'sessions'} if directory_entry else {'manifest.json', 'verified.json'})
            if tail[0] == 'sessions' and len(tail) == 2:
                return not directory_entry and bool(session.fullmatch(tail[1]))
            if tail[0] == 'jobs' and len(tail) == 2:
                return directory_entry and tail[1] == 'allocation-check'
            if tail[0] == 'jobs' and len(tail) == 3:
                return not directory_entry and tail[1] == 'allocation-check' and tail[2] in job_files
            return False
        if parts[0] not in {'base', 'e80'}:
            return False
        tail = parts[1:]
        groups = {'started', 'answers', 'routing', 'jobs', 'sessions', 'errors',
                  'grade-inputs', 'grading', 'grading-raw'}
        if len(tail) == 1:
            return tail[0] in (groups if directory_entry else fixed_profile)
        if len(tail) == 2 and tail[0] in groups - {'jobs'}:
            if directory_entry:
                return False
            if tail[0] == 'sessions':
                return bool(session.fullmatch(tail[1]))
            if tail[0] == 'routing':
                return bool(re.fullmatch(r'\d{3}\.jsonl\.gz', tail[1]))
            if tail[0] == 'grading-raw':
                return bool(re.fullmatch(r'\d{3}-[0-9a-f]{32}\.json', tail[1]))
            return bool(number_json.fullmatch(tail[1]))
        if tail[0] == 'jobs' and len(tail) == 2:
            return directory_entry and bool(job.fullmatch(tail[1]))
        if tail[0] == 'jobs' and len(tail) == 3:
            return not directory_entry and bool(job.fullmatch(tail[1])) and tail[2] in job_files
        return False

    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError('실험 디렉터리가 심볼릭 링크입니다')
    files = []
    for path in directory.rglob('*'):
        if path.is_symlink():
            raise ValueError('실험 결과에 심볼릭 링크가 있습니다')
        parts = path.relative_to(directory).parts
        is_dir = path.is_dir()
        if not allowed(parts, is_dir):
            raise ValueError(f'허용되지 않은 실험 경로: {path.relative_to(directory)}')
        if not is_dir:
            if not path.is_file():
                raise ValueError('일반 파일이 아닙니다')
            files.append(path)
    return sorted(files)


def _archive_files(directory, archive_path, files):
    if archive_path.exists() or archive_path.is_relative_to(directory):
        raise ValueError('새 아카이브 경로는 실험 디렉터리 밖이어야 합니다')
    manifest = {}
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        fd, name = tempfile.mkstemp(prefix='.putare-comparison-', suffix='.zip', dir=archive_path.parent)
        os.close(fd)
        temporary = Path(name)
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for path in files:
                if path.is_symlink():
                    raise ValueError('실험 결과에 심볼릭 링크가 있습니다')
                name = 'results/' + path.relative_to(directory).as_posix()
                manifest[name] = sha256(path)
                archive.write(path, name)
            archive.writestr('manifest.json', json.dumps(manifest, sort_keys=True))
        with temporary.open('r+b') as stream:
            os.fsync(stream.fileno())
        result = verify_archive(temporary)
        if archive_path.exists():
            raise FileExistsError('기존 아카이브는 덮어쓰지 않습니다')
        # 2026-11-11: 링크 생성은 원자적이며 동시 생성된 기존 결과를 덮어쓰지 않는다.
        os.link(temporary, archive_path)
        return result
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def archive_results(directory, archive_path):
    """결과의 전체 파일을 ZIP과 해시 목록으로 보존하고 복원 해시를 검증한다."""
    directory, archive_path = Path(directory).absolute(), Path(archive_path).resolve()
    files = _artifact_paths(directory)
    verified_report(directory)
    return _archive_files(directory, archive_path, files)


def single_gpu_contract(gpu):
    return {key: gpu[key] for key in ('name', 'total_bytes', 'compute_capability', 'driver')}


def validate_single_merge(base, e80, base_tokens, e80_tokens):
    if base.get('model_profile') != 'base' or e80.get('model_profile') != 'e80':
        raise ValueError('두 모델 프로필의 순서가 다릅니다')
    if base.get('shared') != e80.get('shared'):
        raise ValueError('모델별 공통 실험 조건이 다릅니다')
    compare_preflight(base_tokens, e80_tokens)
    return True


def single_source_identity(records, manifest, model, engine, model_profile, gpu):
    # 2026-11-11: 모델 파일·장치 UUID·경로는 독립 보존하고 비교 가능한 조건만 공통 계약으로 묶는다.
    if model_profile not in ('base', 'e80'):
        raise ValueError('모델 프로필은 base 또는 e80이어야 합니다')
    _, provenance = benchmark175.source_identity(model, engine, model_profile=model_profile)
    template = GGUF(model).metadata.get('tokenizer.chat_template')
    if not isinstance(template, str) or not template:
        raise ValueError('모델 chat template 누락')
    shared_config = {k: v for k, v in config(engine).items() if k != 'engine'}
    shared = {'engine_files': provenance['engine_files'], 'source_hashes': provenance['source_hashes'],
              'comparison_source_sha256': sha256(Path(__file__)),
              'snapshot_records': provenance['snapshot_records'],
              'snapshot_manifest': provenance['snapshot_manifest'], 'id_proof': provenance['id_proof'],
              'record_ids': [r['id'] for r in records], 'official_identity': provenance['official_identity'],
              'template_sha256': hashlib.sha256(template.encode('utf-8')).hexdigest(),
              'gpu': single_gpu_contract(gpu), 'config': shared_config,
              'python': sys.version, 'platform': platform.platform(), 'effort': None}
    result = {'model_profile': model_profile, 'model_sha256': provenance['model_sha256'], 'shared': shared}
    if model_profile == 'e80':
        selection = json.loads(benchmark175.PERMUTATION.read_text(encoding='utf-8'))['profile']['problems']
        ids = [row['id'] for row in selection]
        if len(ids) != 100 or len(set(ids)) != 100:
            raise ValueError('선정 프로파일 100문제 ID 오류')
        result['selection_ids'] = ids
        result['export_sha256'] = provenance['export_sha256']
        result['permutation_sha256'] = provenance['permutation_sha256']
    return result


def _single_records():
    records, manifest = load_snapshot(benchmark175.SNAPSHOT)
    if (len(records) != 205 or len({r['id'] for r in records}) != 205
            or sum(r['domain'] == 'math' for r in records) != 30
            or sum(r['domain'] == 'code' for r in records) != 175):
        raise ValueError('고유한 AIME 30문제와 LCB 175문제가 필요합니다')
    return records, manifest


def _single_dry_run(records, model, engine):
    tokens = tokenize_records(records, model, engine)
    if len(tokens) != 205 or [r['id'] for r in tokens] != [r['id'] for r in records]:
        raise ValueError('205문제 토큰 사전 점검 누락')
    requests = []
    for record, row in zip(records, tokens):
        request = make_request(record, model, engine, row['tokens'])
        requests.append({k: v for k, v in request.items() if k != 'model'})
    return {'tokens': tokens, 'requests': requests}


def prepare_single(directory, *, model, engine, model_profile):
    """한 모델과 현재 Colab 세션에서만 입력 및 최대 context를 검증한다."""
    from moe_prune.resident import ResidentExperiment
    directory, model, engine = Path(directory), Path(model).resolve(), Path(engine).resolve()
    records, manifest = _single_records()
    gpu = checked_gpu()
    identity = single_source_identity(records, manifest, model, engine, model_profile, gpu)
    directory.mkdir(parents=True, exist_ok=True)
    _write_once(directory / 'identity.json', identity)
    dry = _single_dry_run(records, model, engine)
    _write_once(directory / 'dry-run.json', dry)
    prepared = {'identity_hash': canonical_hash(identity), 'dry_run_hash': canonical_hash(dry), 'requests': 205}
    _write_once(directory / 'prepared.json', prepared)
    session_check = directory / 'session-checks' / (SESSION_ID + '.json')
    if session_check.exists():
        existing = json.loads(session_check.read_text(encoding='utf-8'))
        if (existing.get('gpu') != gpu or existing.get('identity_hash') != canonical_hash(identity)
                or existing.get('dry_run_hash') != canonical_hash(dry)):
            raise ValueError('같은 세션의 GPU 또는 사전 점검 신원이 변경되었습니다')
        return {'ready': True, 'requests': 205, 'gpu': gpu, 'session_id': SESSION_ID}
    warm_record = {'id': 'allocation-check', 'prompt': 'Reply with the word ready.'}
    warm = tokenize_records([warm_record], model, engine)[0]
    request = make_request(warm_record, model, engine, warm['tokens'])
    request['max_tokens'] = 0
    request['expected_rendered_prompt'] = warm['rendered_prompt']
    attempt = uuid.uuid4().hex
    check_dir = directory / 'checks' / attempt
    with ResidentExperiment(check_dir, config(engine)) as exp:
        result = exp.run('allocation-check', _native_request(request))
        observed = _validate_result(check_dir, 'allocation-check', request, result,
                                    128 if model_profile == 'base' else 80)
    ensure_gpu(gpu)
    _write_once(session_check, {'session_id': SESSION_ID, 'gpu': gpu, 'check': attempt,
                                'identity_hash': canonical_hash(identity), 'dry_run_hash': canonical_hash(dry),
                                'observed': observed, 'measurement': result['measurement']})
    return {'ready': True, 'requests': 205, 'gpu': gpu, 'session_id': SESSION_ID}


def _check_single(directory, model, engine, model_profile):
    directory = Path(directory)
    records, manifest = _single_records()
    gpu = checked_gpu()
    identity = single_source_identity(records, manifest, model, engine, model_profile, gpu)
    if json.loads((directory / 'identity.json').read_text(encoding='utf-8')) != identity:
        raise ValueError('현재 모델·GPU 클래스·엔진 또는 데이터가 실험 조건과 다릅니다')
    dry = json.loads((directory / 'dry-run.json').read_text(encoding='utf-8'))
    prepared = json.loads((directory / 'prepared.json').read_text(encoding='utf-8'))
    if prepared != {'identity_hash': canonical_hash(identity), 'dry_run_hash': canonical_hash(dry), 'requests': 205}:
        raise ValueError('205개 요청 사전 점검이 완전하지 않습니다')
    session = json.loads((directory / 'session-checks' / (SESSION_ID + '.json')).read_text(encoding='utf-8'))
    if (session.get('gpu') != gpu or session.get('identity_hash') != canonical_hash(identity)
            or session.get('dry_run_hash') != canonical_hash(dry)):
        raise ValueError('현재 세션 사전 점검 또는 GPU UUID가 다릅니다')
    if ([row['id'] for row in dry['tokens']] != [r['id'] for r in records]
            or len(dry['requests']) != 205):
        raise ValueError('사전 점검 문제 순서 또는 요청 수가 다릅니다')
    return records, identity, dry, gpu


def _load_stored_single(directory):
    """GPU와 GGUF 없이 저장된 프로토콜과 입력 사전 검증을 다시 확인한다."""
    directory = Path(directory)
    records, manifest = _single_records()
    identity = json.loads((directory / 'identity.json').read_text(encoding='utf-8'))
    dry = json.loads((directory / 'dry-run.json').read_text(encoding='utf-8'))
    prepared = json.loads((directory / 'prepared.json').read_text(encoding='utf-8'))
    if prepared != {'identity_hash': canonical_hash(identity), 'dry_run_hash': canonical_hash(dry), 'requests': 205}:
        raise ValueError('저장된 사전 검증의 해시 불일치')
    profile, shared = identity.get('model_profile'), identity.get('shared', {})
    expected_sha = {'base': benchmark175.BASE_MODEL_SHA, 'e80': benchmark175.MODEL_SHA}.get(profile)
    if (not expected_sha or identity.get('model_sha256') != expected_sha
            or shared.get('record_ids') != [r['id'] for r in records]
            or shared.get('snapshot_records') != benchmark175.RECORDS_SHA
            or shared.get('snapshot_manifest') != sha256(benchmark175.SNAPSHOT / 'manifest.json')
            or shared.get('id_proof') != canonical_hash(validate_id_proof(benchmark175.SNAPSHOT))
            or shared.get('official_identity') != manifest['official_identity']
            or shared.get('config') != {k: v for k, v in config(None).items() if k != 'engine'}):
        raise ValueError('저장된 데이터 또는 모델·실험 조건 불일치')
    tokens, requests = dry.get('tokens', []), dry.get('requests', [])
    if len(tokens) != 205 or len(requests) != 205:
        raise ValueError('저장된 205개 요청 누락')
    for record, row, request in zip(records, tokens, requests):
        if row['id'] != record['id'] or not isinstance(row.get('rendered_prompt'), str):
            raise ValueError('저장된 토큰의 문제 ID 불일치')
        expected = make_request(record, 'unused', None, row['tokens'])
        if {k: v for k, v in expected.items() if k != 'model'} != request:
            raise ValueError('저장된 생성 조건 불일치')
    return records, identity, dry


def generate_single(directory, *, model, engine, model_profile):
    """단일 모델의 205문제를 실행하고 새 세션에서는 미시작 문제만 재개한다."""
    from moe_prune.resident import ResidentExperiment
    directory, model, engine = Path(directory), Path(model).resolve(), Path(engine).resolve()
    records, identity, dry, gpu = _check_single(directory, model, engine, model_profile)
    identity_hash = canonical_hash(identity)
    states = [case_state(directory, i, row['id']) for i, row in enumerate(records, 1)]
    for i, (record, state) in enumerate(zip(records, states), 1):
        if state == 'complete':
            verified_single_answer(directory, i, record, identity, dry)
    pending = [i for i, state in enumerate(states, 1) if state == 'unstarted']
    if pending:
        ensure_gpu(gpu)
        warm_record = {'id': 'warmup', 'prompt': 'Reply with the word ready.'}
        warm = tokenize_records([warm_record], model, engine)[0]
        warm_request = make_request(warm_record, model, engine, warm['tokens'])
        warm_request.update(max_tokens=1, expected_rendered_prompt=warm['rendered_prompt'])
        experts = 128 if model_profile == 'base' else 80
        with ResidentExperiment(directory, config(engine)) as exp:
            job = 'warmup-' + uuid.uuid4().hex
            warm_result = exp.run(job, _native_request(warm_request))
            _validate_result(directory, job, warm_request, warm_result, experts)
            for number in pending:
                record, tokens = records[number - 1], dry['tokens'][number - 1]
                ensure_gpu(gpu)
                request = make_request(record, model, engine, tokens['tokens'])
                if {k: v for k, v in request.items() if k != 'model'} != dry['requests'][number - 1]:
                    raise ValueError('현재 생성 요청이 사전 검증과 다릅니다')
                request['expected_rendered_prompt'] = tokens['rendered_prompt']
                reserve_case(directory, number, record['id'])
                job = f'problem-{number:03d}'
                try:
                    result = exp.run(job, _native_request(request))
                    observed = _validate_result(directory, job, request, result, experts)
                    ensure_gpu(gpu)
                    raw = directory / 'jobs' / job / 'events.jsonl'
                    target = directory / 'routing' / f'{number:03d}.jsonl.gz'
                    integrity = benchmark175.compress_events(raw, target)
                    with target.open('r+b') as stream:
                        os.fsync(stream.fileno())
                    exp.check()
                    answer = {'id': record['id'], 'identity_hash': identity_hash, 'text': result['text'],
                              'result': result, 'gpu': gpu, 'session_id': SESSION_ID,
                              'observed_runtime': observed, 'routing_integrity': integrity, 'job': job}
                    answer['answer_sha256'] = canonical_hash(answer)
                    _durable_answer(directory / 'answers' / f'{number:03d}.json', answer)
                    verified_single_answer(directory, number, record, identity, dry)
                    raw.unlink()
                    print(f'{model_profile} {number}/205 {record["id"]}', flush=True)
                except BaseException as error:
                    write_json(directory / 'errors' / f'{number:03d}.json',
                               {'id': record['id'], 'status': 'infrastructure_error',
                                'error_type': type(error).__name__, 'message': str(error), 'session_id': SESSION_ID})
                    raise
    states = [case_state(directory, i, row['id']) for i, row in enumerate(records, 1)]
    status = {'complete': all(s == 'complete' for s in states), 'completed': states.count('complete'),
              'interrupted': states.count('interrupted'), 'total': 205}
    write_json(directory / 'status.json', status)
    if not status['complete']:
        raise RuntimeError('미완료: 중단된 문제는 자동 재풀이하지 않습니다')
    return status


def verified_single_answer(directory, number, record, identity, dry):
    directory = Path(directory)
    answer = verified_answer(directory, number, record['id'], canonical_hash(identity))
    tokens = dry['tokens'][number - 1]
    result = answer['result']
    if (answer.get('job') != f'problem-{number:03d}' or tokens['id'] != record['id']
            or result.get('prompt_tokens') != len(tokens['tokens'])
            or result.get('rendered_prompt') != tokens['rendered_prompt']
            or result.get('actual_context_size') != CONTEXT_SIZE):
        raise ValueError('답안의 문제 또는 토큰 사전 점검 불일치')
    request = json.loads((directory / 'jobs' / answer['job'] / 'request.json').read_text(encoding='utf-8'))
    if {k: v for k, v in request.items() if k != 'model'} != dry['requests'][number - 1]:
        raise ValueError('답안의 실제 생성 요청 불일치')
    job_identity = json.loads((directory / 'jobs' / answer['job'] / 'identity.json').read_text(encoding='utf-8'))
    if (job_identity['model']['sha256'] != identity['model_sha256']
            or job_identity['engine']['sha256'] not in identity['shared']['engine_files'].values()):
        raise ValueError('답안의 모델 또는 엔진 해시 불일치')
    session_id = answer.get('session_id', '')
    if not re.fullmatch(r'[0-9a-f]{32}', session_id):
        raise ValueError('답안 세션 ID 오류')
    session = json.loads((directory / 'session-checks' / (session_id + '.json')).read_text(encoding='utf-8'))
    if (session.get('gpu') != answer.get('gpu') or session.get('identity_hash') != canonical_hash(identity)
            or session.get('dry_run_hash') != canonical_hash(dry)
            or single_gpu_contract(session['gpu']) != identity['shared']['gpu']):
        raise ValueError('답안의 세션 GPU 또는 실험 조건 불일치')
    return answer


def _single_manifest():
    return json.loads((benchmark175.SNAPSHOT / 'manifest.json').read_text(encoding='utf-8'))


def _single_load_timings(directory):
    values = []
    for folder in sorted((Path(directory) / 'jobs').glob('warmup-*')):
        path = folder / 'events.jsonl'
        try:
            rows = [row for row in read_jsonl(path) if row.get('type') == 'result']
            values.append(rows[0].get('timings', {}).get('load_s') if len(rows) == 1 else None)
        except (OSError, ValueError):
            values.append(None)
    return values


def _single_report(records, entries, identity, directory):
    domains = {}
    by_domain = {}
    cases = []
    for domain in ('math', 'code'):
        rows = [entries[r['id']] for r in records if r['domain'] == domain]
        count = len(rows)
        passed = sum(row['passed'] for row in rows)
        domains[domain] = {'total': count, 'passed': passed, 'percentage': 100 * passed / count if count else None}
        by_domain[domain] = efficiency_summary(rows)
    for record in records:
        cases.append({'id': record['id'], 'domain': record['domain'], **entries[record['id']]})
    return {'complete': True, 'model_profile': identity['model_profile'],
            'identity_hash': canonical_hash(identity), 'domains': domains,
            'efficiency': efficiency_summary([entries[r['id']] for r in records]),
            'efficiency_by_domain': by_domain, 'model_load_s': _single_load_timings(directory),
            'cases': cases, 'limitations': ['문제당 단일 시도', '라우팅 계측 포함 속도',
                'GPU 메모리는 세션별 누적 표본 최대값', '세션이 다르면 GPU UUID가 다를 수 있음']}


def _single_entry(answer, verdict):
    return {**_answer_entry(answer, verdict), 'gpu': answer['gpu'], 'session_id': answer['session_id']}


def _grade_single_impl(directory, *, model=None, engine=None, model_profile=None):
    from moe_prune.execution_safety import execution_guard
    from moe_prune.grading import GradeDeadline, run, worker_environment
    from tools.lcb_prepare import COMMIT
    directory = Path(directory)
    records, identity, dry = _load_stored_single(directory)
    if model_profile is not None and identity['model_profile'] != model_profile:
        raise ValueError('요청한 모델 프로필이 보존된 답안과 다릅니다')
    if any(case_state(directory, i, row['id']) != 'complete' for i, row in enumerate(records, 1)):
        raise ValueError('205개 답안이 완료되지 않았습니다')
    source = benchmark175.ROOT / '.cache/putare-lcb/livecodebench' / COMMIT
    venv = benchmark175.ROOT / '.cache/putare-lcb/venv'
    adapter = benchmark175.ROOT / 'tools/lcb_grade.py'
    if sha256(adapter) != _single_manifest()['grader_identity']['tools/lcb_grade.py']:
        raise ValueError('채점 adapter 해시 불일치')
    entries = {}
    with execution_guard('grading', memory_config=config(engine), monitor_gpu=False) as guard:
        deadline = GradeDeadline()
        probe = subprocess.run([str(venv / 'bin/python'), '-I', '-B', str(adapter), '--verify',
                                '--official-source', str(source)], env=worker_environment(),
                               check=True, capture_output=True, text=True, timeout=deadline.remaining())
        if json.loads(probe.stdout)['source_digest'] != identity['shared']['official_identity']['source_digest']:
            raise ValueError('공식 채점기 해시 불일치')
        for number, record in enumerate(records, 1):
            answer = verified_single_answer(directory, number, record, identity, dry)
            def compute():
                if record['domain'] == 'math':
                    return grade_math(answer['text'], record['reference'])
                payload = {'id': record['id'], 'problem': record['lcb_problem'], 'text': answer['text']}
                input_path = directory / 'grade-inputs' / f'{number:03d}.json'
                _write_once(input_path, payload)
                target = directory / 'grading-raw' / f'{number:03d}-{uuid.uuid4().hex}.json'
                target.parent.mkdir(parents=True, exist_ok=True)
                graded = run(source, venv, input_path, target, adapter=adapter,
                             guard=guard, timeout_s=deadline.remaining())
                if (graded.get('id') != record['id'] or type(graded.get('passed')) is not bool
                        or graded.get('pass_at_1') != float(graded['passed'])):
                    raise ValueError('공식 채점 결과 불일치')
                return graded['passed']
            verdict = grade_cached(directory / 'grading' / f'{number:03d}.json', answer, compute)
            entries[record['id']] = _single_entry(answer, verdict)
            write_json(directory / 'grading-status.json', {'complete': False, 'completed': number,
                       'total': 205, 'status': 'grading', 'last_id': record['id']})
    report = _single_report(records, entries, identity, directory)
    write_json(directory / 'single-report.json', report)
    write_json(directory / 'grading-status.json', {'complete': True, 'completed': 205,
               'total': 205, 'status': 'graded'})
    return report


def grade_single(directory, *, model=None, engine=None, model_profile=None):
    directory = Path(directory)
    try:
        return _grade_single_impl(directory, model=model, engine=engine, model_profile=model_profile)
    except BaseException as error:
        if directory.is_dir():
            path = directory / 'grading-status.json'
            previous = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            write_json(path, {'complete': False, 'completed': previous.get('completed', 0), 'total': 205,
                              'status': 'infrastructure_error', 'error_type': type(error).__name__,
                              'message': str(error)})
        raise


def verified_single_report(directory):
    """단일 실험의 답안·세션·채점 및 집계 205개를 GPU 없이 재검증한다."""
    directory = Path(directory)
    records, identity, dry = _load_stored_single(directory)
    status = json.loads((directory / 'grading-status.json').read_text(encoding='utf-8'))
    if status != {'complete': True, 'completed': 205, 'total': 205, 'status': 'graded'}:
        raise ValueError('205개 채점이 완료되지 않았습니다')
    entries = {}
    for number, record in enumerate(records, 1):
        answer = verified_single_answer(directory, number, record, identity, dry)
        path = directory / 'grading' / f'{number:03d}.json'
        def missing_grade():
            raise ValueError('채점 결과가 누락되었습니다')
        verdict = grade_cached(path, answer, missing_grade)
        entries[record['id']] = _single_entry(answer, verdict)
    expected = _single_report(records, entries, identity, directory)
    stored = json.loads((directory / 'single-report.json').read_text(encoding='utf-8'))
    if stored != expected:
        raise ValueError('단일 모델 보고서가 답안 및 채점과 다릅니다')
    return stored


def archive_single(directory, archive_path):
    """단일 모델 결과만 저장하며 다른 GGUF, GPU 또는 다른 실행을 요구하지 않는다."""
    directory, archive_path = Path(directory).absolute(), Path(archive_path).resolve()
    files = _artifact_paths(directory, single=True)
    verified_single_report(directory)
    return _archive_files(directory, archive_path, files)


def compare_runs(base_dir, e80_dir, output_path=None):
    """독립 Colab 세션의 완료된 결과를 원본 모델과 GPU 없이 비교한다."""
    base_records, base_id, base_dry = _load_stored_single(base_dir)
    e80_records, e80_id, e80_dry = _load_stored_single(e80_dir)
    if [r['id'] for r in base_records] != [r['id'] for r in e80_records]:
        raise ValueError('두 실험의 문제 목록이 다릅니다')
    validate_single_merge(base_id, e80_id, base_dry['tokens'], e80_dry['tokens'])
    if base_dry['requests'] != e80_dry['requests']:
        raise ValueError('두 실험의 문제별 생성 조건이 다릅니다')
    base_report, e80_report = verified_single_report(base_dir), verified_single_report(e80_dir)
    base = {row['id']: {k: v for k, v in row.items() if k not in ('id', 'domain')}
            for row in base_report['cases']}
    e80 = {row['id']: {k: v for k, v in row.items() if k not in ('id', 'domain')}
           for row in e80_report['cases']}
    report = comparison_report(base_records, base, e80, e80_id['selection_ids'])
    report['identity_hashes'] = {'base': canonical_hash(base_id), 'e80': canonical_hash(e80_id)}
    report['gpu_uuids'] = {profile: sorted({entry['gpu']['uuid'] for entry in values.values()})
                           for profile, values in (('base', base), ('e80', e80))}
    report['model_load_s'] = {'base': base_report.get('model_load_s', []),
                              'e80': e80_report.get('model_load_s', [])}
    report['limitations'].append('별도 Colab 세션의 GPU UUID는 다를 수 있으며 같은 물리 GPU를 뜻하지 않음')
    if output_path is not None:
        _write_once(output_path, report)
    return report
