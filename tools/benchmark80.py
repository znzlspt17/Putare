"""80 Expert 모델의 수학 1문항과 LCB 2문항 시험 실행."""
from __future__ import annotations

import gzip
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import re
import subprocess

from moe_prune.gguf import sha256
from moe_prune.research import canonical_hash, read_jsonl, write_json, write_jsonl
from tools import manual80

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'runs/gemma4-ordered-e80-trial3-colab-a100-001'
MATH_RECORD = {
    'id': 'math500-test-11', 'domain': 'math', 'dataset': 'MATH-500',
    'prompt': 'Let $p(x)$ be a polynomial of degree 5 such that\n'
              '\\[p(n) = \\frac{n}{n^2 - 1}\\]for $n = 2,$ 3, 4, $\\dots,$ 7.  Find $p(8).$',
    'reference': r'\frac{3}{56}',
}
IDS = [MATH_RECORD['id'], *manual80.IDS]


# 2026-11-09: 이전 문제 원문과 동일한 입력을 고정하고 LCB 묶음의 출처 검증을 재사용한다.
def source_identity(model, engine):
    code_records, identity = manual80.source_identity(model=Path(model), engine=Path(engine))
    if [r['id'] for r in code_records] != manual80.IDS:
        raise ValueError('LCB 문제 ID 불일치')
    records = [dict(MATH_RECORD), *code_records]
    return records, identity


def _boxed(text):
    from moe_prune.benchmark_data import final_answer
    text = final_answer(text)
    matches = list(re.finditer(r'\\boxed\s*\{', text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
    return None


# 2026-11-09: 문제의 분수 정답만 정확하게 비교하며 임의 숫자를 정답으로 추출하지 않는다.
def grade_math(text):
    answer = _boxed(text)
    value = None
    if answer is not None:
        compact = re.sub(r'\s+', '', answer)
        match = re.fullmatch(r'\\(?:d?frac)\{([+-]?\d+)\}\{([+-]?\d+)\}', compact)
        if not match:
            match = re.fullmatch(r'([+-]?\d+)/([+-]?\d+)', compact)
        if match and int(match.group(2)) != 0:
            value = Fraction(int(match.group(1)), int(match.group(2)))
    return {'status': 'graded' if value is not None else 'unparsed',
            'passed': value == Fraction(3, 56) if value is not None else None,
            'answer': answer, 'reference': MATH_RECORD['reference'],
            'method': 'exact rational comparison; not official MATH-500 grading'}


def _check_records(records):
    if [r['id'] for r in records] != IDS or records[0] != MATH_RECORD:
        raise ValueError('선택한 세 문제의 ID 또는 수학 문제가 다릅니다')


# 2026-11-09: 세 문제는 하나의 상주 모델을 공유하고 정상 종료 후에만 채점한다.
def generate(directory=DIRECTORY, *, model=manual80.MODEL, engine=manual80.ENGINE):
    from moe_prune.base_benchmark import build_request
    from moe_prune.resident import ResidentExperiment
    from tools.manual104 import config03, validate_job, compress_events

    directory, model, engine = Path(directory), Path(model), Path(engine)
    if directory.exists() or directory.is_symlink():
        raise FileExistsError('기존 실행 폴더 보존')
    records, source = source_identity(model, engine)
    _check_records(records)
    config = {**config03(), 'n_gpu_layers': 31, 'engine': str(engine)}
    identity = {**source, 'ids': IDS, 'records_hash': canonical_hash(records),
                'trial_script_sha256': sha256(Path(__file__)), 'config': config,
                'model_path': str(model.resolve()), 'engine_path': str(engine.resolve()),
                'placement': [31, True]}
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'identity.json', identity)
    answers = []
    try:
        with ResidentExperiment(directory, config) as exp:
            for number, record in enumerate(records, 1):
                name = f'problem-{number:03d}'
                folder = directory / 'jobs' / name
                request = build_request(record, model, (31, True), config)
                print(f'{number}/3 {record["id"]} 생성 시작', flush=True)
                result = exp.run(name, request)
                if result.get('stop_reason') not in ('eog', 'length') or (
                    result.get('stop_reason') == 'length' and len(result.get('token_ids', [])) != 32768
                ):
                    raise ValueError('생성 종료 사유 오류')
                if result.get('masked_layers'):
                    raise ValueError('물리 모델에 마스크가 적용됨')
                observed = validate_job(directory, name, request, result, 80)
                events = folder / 'events.jsonl'
                native = manual80.native_result(events)
                if native != {key: value for key, value in result.items() if key != 'measurement'}:
                    raise ValueError('native 최종 결과 불일치')
                compressed = directory / 'routing' / f'{number:03d}.jsonl.gz'
                integrity = compress_events(events, compressed)
                exp.check()
                answer = {'id': record['id'], 'number': number, **result,
                          'native_result_sha256': canonical_hash(native), 'observed_runtime': observed,
                          'routing_path': str(compressed.relative_to(directory)), 'routing_integrity': integrity,
                          'raw_events_path': str(events.relative_to(directory))}
                write_json(directory / 'answers' / f'{number:03d}.json', answer)
                payload = {'id': record['id'], 'text': native['text']}
                if record['domain'] == 'code':
                    payload['problem'] = record['lcb_problem']
                write_json(directory / 'grade-inputs' / f'{number:03d}.json', payload)
                answers.append(answer)
                write_jsonl(directory / 'generations.jsonl', answers)
                write_json(directory / 'status.json', {'complete': False, 'completed': len(answers), 'total': 3})
        write_json(directory / 'status.json', {'complete': True, 'completed': 3, 'total': 3, 'grading': 'pending'})
    except BaseException:
        write_json(directory / 'status.json', {'complete': False, 'completed': len(answers), 'total': 3})
        raise
    return directory


# 2026-11-09: 저장된 답안과 압축 계측이 해당 모델·문제·원본 native 결과에 결속되는지 재검증한다.
def verify_inputs(directory, *, model=manual80.MODEL, engine=manual80.ENGINE):
    from moe_prune.base_benchmark import build_request
    from tools.manual104 import config03

    directory = Path(directory).resolve()
    status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    if status.get('complete') is not True or status.get('completed') != 3 or status.get('total') != 3:
        raise ValueError('세 문제 생성 미완료')
    records, source = source_identity(model, engine)
    _check_records(records)
    identity = json.loads((directory / 'identity.json').read_text(encoding='utf-8'))
    if (identity.get('ids') != IDS or identity.get('records_hash') != canonical_hash(records) or
            identity.get('trial_script_sha256') != sha256(Path(__file__)) or
            identity.get('model_path') != str(Path(model).resolve()) or
            identity.get('engine_path') != str(Path(engine).resolve())):
        raise ValueError('실행 입력 출처 불일치')
    expected_config = {**config03(), 'n_gpu_layers': 31, 'engine': str(Path(engine))}
    if identity.get('config') != expected_config or identity.get('placement') != [31, True]:
        raise ValueError('실행 설정 불일치')
    for key, value in source.items():
        if key != 'ids' and identity.get(key) != value:
            raise ValueError('모델·엔진·문제 출처 불일치')
    items = []
    for number, record in enumerate(records, 1):
        folder = directory / 'jobs' / f'problem-{number:03d}'
        job = json.loads((folder / 'identity.json').read_text(encoding='utf-8'))
        if job['model']['sha256'] != identity['model_sha256'] or job['engine']['sha256'] != identity['engine_sha256']:
            raise ValueError('native 작업 출처 불일치')
        request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
        if request != build_request(record, model, (31, True), expected_config):
            raise ValueError('native 요청과 선택한 문제가 다릅니다')
        answer = json.loads((directory / 'answers' / f'{number:03d}.json').read_text(encoding='utf-8'))
        payload_path = directory / 'grade-inputs' / f'{number:03d}.json'
        payload = json.loads(payload_path.read_text(encoding='utf-8'))
        if (answer.get('routing_path', '').replace('\\', '/') != f'routing/{number:03d}.jsonl.gz' or
                answer.get('raw_events_path', '').replace('\\', '/') != f'jobs/problem-{number:03d}/events.jsonl'):
            raise ValueError('계측 경로 불일치')
        raw = (folder / 'events.jsonl').resolve()
        compressed = (directory / 'routing' / f'{number:03d}.jsonl.gz').resolve()
        if not raw.is_relative_to(directory) or not compressed.is_relative_to(directory):
            raise ValueError('계측 경로 범위 오류')
        integrity = answer['routing_integrity']
        if sha256(raw) != integrity['uncompressed_sha256'] or sha256(compressed) != integrity['compressed_sha256']:
            raise ValueError('계측 파일 해시 불일치')
        digest = hashlib.sha256()
        with gzip.open(compressed, 'rb') as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        if digest.hexdigest() != integrity['uncompressed_sha256']:
            raise ValueError('압축 계측 내용 불일치')
        native = manual80.native_result(raw)
        if (answer.get('id') != record['id'] or answer.get('text') != native.get('text') or
                answer.get('token_ids') != native.get('token_ids') or
                answer.get('native_result_sha256') != canonical_hash(native)):
            raise ValueError('답안과 native 결과 불일치')
        expected = {'id': record['id'], 'text': native['text']}
        if record['domain'] == 'code':
            expected['problem'] = record['lcb_problem']
        if payload != expected:
            raise ValueError('채점 입력이 native 결과와 다릅니다')
        items.append({'id': record['id'], 'number': number, 'record': record,
                      'answer': answer, 'payload': payload_path})
    return identity, items


def _verify_official(venv, deadline):
    from moe_prune.grading import worker_environment
    bundle = manual80.BUNDLE
    probe = subprocess.run([str(Path(venv) / 'bin/python'), '-I', '-B', str(bundle / 'lcb_grade.py'),
                            '--verify', '--official-source', str(bundle / 'official-source')],
                           env=worker_environment(), check=True, capture_output=True, text=True,
                           timeout=deadline.remaining())
    official = json.loads(probe.stdout)
    original = json.loads((bundle / 'original-manifest.json').read_text(encoding='utf-8'))
    if official.get('source_digest') != original['official_identity'].get('source_digest'):
        raise ValueError('공식 채점기 source 불일치')
    return official


# 2026-11-09: 추론 종료 뒤 하나의 잠금과 제한 시간으로 두 공식 코드 채점을 순차 실행한다.
def grade(directory=DIRECTORY, *, model=manual80.MODEL, engine=manual80.ENGINE,
          venv=ROOT / '.cache/putare-lcb/venv'):
    from moe_prune.execution_safety import execution_guard
    from moe_prune.grading import GradeDeadline, GradingCleanupError, identity as grader_identity, run as run_grade
    from tools.manual104 import config03

    directory = Path(directory).resolve()
    identity, items = verify_inputs(directory, model=model, engine=engine)
    deadline = GradeDeadline()
    with execution_guard('grading', memory_config=config03()) as guard:
        official = _verify_official(venv, deadline)
        out = directory / 'grading'
        out.mkdir(exist_ok=False)
        write_json(out / 'identity.json', {'official': official, 'run_identity_sha256': sha256(directory / 'identity.json'),
                                          'inventory_sha256': identity['bundle_inventory_sha256'], **grader_identity()})
        rows = []
        for item in items:
            record, answer = item['record'], item['answer']
            if record['domain'] == 'math':
                verdict = grade_math(answer['text'])
            else:
                target = out / f"{item['number']:03d}.json"
                try:
                    result = run_grade(manual80.BUNDLE / 'official-source', venv, item['payload'], target,
                                       adapter=manual80.BUNDLE / 'lcb_grade.py', guard=guard,
                                       timeout_s=deadline.remaining())
                    verdict = {'status': 'graded', 'passed': result['passed'], 'test_count': result.get('test_count')}
                except GradingCleanupError:
                    raise
                except Exception as error:
                    verdict = {'status': 'infrastructure_error', 'passed': None, 'error_type': type(error).__name__}
            rows.append({'number': item['number'], 'id': item['id'], 'domain': record['domain'], **verdict,
                         'generated_tokens': len(answer['token_ids']), 'stop_reason': answer['stop_reason'],
                         'timings': answer.get('timings'), 'measurement': answer.get('measurement')})
            write_json(out / 'summary.json', {'complete': False, 'scope': 'MATH-500 1 + LCB 2', 'cases': rows})
            if verdict.get('error_type') == 'TimeoutError':
                break
        summary = {'complete': len(rows) == 3 and all(r['status'] in ('graded', 'unparsed') for r in rows),
                   'scope': 'MATH-500 1 + LCB 2', 'model_sha256': identity['model_sha256'], 'cases': rows,
                   'limitations': ['수학 채점은 3/56의 정확한 분수 비교이며 공식 MATH-500 채점이 아닙니다.',
                                   'LCB 코드는 비격리 프로세스에서 실행됩니다.',
                                   '속도와 GPU 메모리는 계측 영향과 표본 한계가 있습니다.']}
        write_json(out / 'summary.json', summary)
        return summary
