"""2026-11-06: 고정된 공식 LiveCodeBench formatter·실행 채점 어댑터."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import zlib

SOURCE_DIGEST = 'cff1236a4979e47edd9abd87cdd3b43b381165f96777f69fd89c3d7b8a99b469'


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 공개 데이터의 pickle은 문자열만 포함한다. 임의 코드 실행을 허용하지 않는다.
        raise pickle.UnpicklingError('pickle globals are not allowed')


def decode_cases(value):
    try:
        result = json.loads(value)
    except json.JSONDecodeError:
        raw = zlib.decompress(base64.b64decode(value, validate=True))
        unpacked = _DataOnlyUnpickler(io.BytesIO(raw)).load()
        if not isinstance(unpacked, str):
            raise ValueError('private tests pickle must contain a JSON string')
        result = json.loads(unpacked)
    if not isinstance(result, list) or not result:
        raise ValueError('tests must be a non-empty list')
    for item in result:
        if not isinstance(item, dict) or not {'input', 'output'} <= set(item):
            raise ValueError('invalid test case')
    return result


def _source(path):
    source = Path(path).resolve()
    if not (source / 'lcb_runner/evaluation/testing_util.py').is_file():
        raise RuntimeError('고정된 공식 source가 없습니다')
    package = source / 'lcb_runner'
    files = sorted(package.rglob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(package)).encode())
        digest.update(b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    if len(files) != 40 or digest.hexdigest() != SOURCE_DIGEST:
        raise RuntimeError('고정된 공식 source 해시가 일치하지 않습니다')
    sys.path.insert(0, str(source))


def source_identity(source):
    _source(source)
    return {'source_digest': SOURCE_DIGEST, 'python': sys.version.split()[0],
            'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'tqdm', 'datasets')}}


def prompts(items, source):
    _source(source)
    # 공식 formatter는 저장소 기준 상대경로에서 few-shot 자료를 읽는다.
    os.chdir(source)
    # prompts/__init__는 사용하지 않는 클라우드 SDK를 강제하므로 공식 모듈만 직접 로드한다.
    path = Path(source) / 'lcb_runner/prompts/code_generation.py'
    spec = importlib.util.spec_from_file_location('_lcb_official_codegen', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = []
    for row in items:
        question = SimpleNamespace(question_content=row['question_content'], starter_code=row['starter_code'])
        result.append({'prompt': module.get_generic_question_template_answer(question),
                       'system_prompt': module.PromptConstants.SYSTEM_MESSAGE_GENERIC})
    return result


def grade_one(row, source):
    """2026-11-06: 공식 추출·테스트·pass@1을 실행하며 프레임워크 예외는 전파한다."""
    _source(source)
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness
    from lcb_runner.evaluation.pass_k_utils import estimate_pass_at_k
    problem = row['problem']
    public = decode_cases(problem['public_test_cases'])
    private = decode_cases(problem['private_test_cases'])
    tests = public + private
    metadata = json.loads(problem['metadata'])
    sample = {'input_output': json.dumps({'inputs': [x['input'] for x in tests],
                                          'outputs': [x['output'] for x in tests],
                                          'fn_name': metadata.get('func_name')})}
    text = row['text']
    if '<|channel>thought' in text:
        thought_tail = text.rsplit('<|channel>thought', 1)[1]
        text = thought_tail.split('<channel|>', 1)[1] if '<channel|>' in thought_tail else ''
    if '<|channel>final' in text:
        text = text.rsplit('<|channel>final', 1)[1]
    code = extract_code(text, LMStyle.OpenAIChat)
    results, details = check_correctness(sample, code, timeout=6, debug=False)
    if not isinstance(results, list) or not results or not isinstance(details, dict):
        raise RuntimeError('공식 채점기 결과 계약 위반')
    # numpy.bool_도 True로 처리하지만 음수 error code는 성공으로 간주하지 않는다.
    passed = all(value is True or (type(value).__name__ == 'bool_' and bool(value)) for value in results)
    score = float(estimate_pass_at_k([1], [int(passed)], 1)[0])
    return {'id': row['id'], 'passed': passed, 'pass_at_1': score,
            'code': code, 'test_count': len(tests), 'details': details}


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--worker', action='store_true')
    mode.add_argument('--prompts', action='store_true')
    mode.add_argument('--verify', action='store_true')
    parser.add_argument('--official-source', required=True)
    parser.add_argument('--input')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(source_identity(args.official_source)))
        return
    if not args.input or not args.output:
        parser.error('input and output are required')
    data = json.loads(Path(args.input).read_text(encoding='utf-8'))
    if args.worker:
        try:
            result = grade_one(data, args.official_source)
        except Exception as error:
            # 생성 코드의 출력이나 예외 문자열은 격리 경계 밖으로 내보내지 않는다.
            import traceback
            frames = traceback.extract_tb(error.__traceback__)
            result = {'status': 'infrastructure_error', 'error_type': type(error).__name__,
                      'location': [[Path(f.filename).name, f.name, f.lineno] for f in frames[-4:]]}
    else:
        result = prompts(data, args.official_source)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
