"""LCB 175문항의 E80 상주 추론과 공식 채점."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from pathlib import Path
import subprocess

from moe_prune.gguf import GGUF, sha256
from moe_prune.research import canonical_hash, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / 'runs/datasets/aime2026-lcbv6-new-003'
RECORDS_SHA = 'd402380ad1e4c104143721427fe99d07e0cabb0389f0c28f080991d71f7f3ca6'
MODEL_SHA = '760a302c4f67cb99dc28b6d195ae0ec6a623d8c37614fedb20df555b92180321'
BASE_MODEL_SHA = 'eeb867f279ea5a3d52a0dc15fe8ada677b3328a328530957f3f9a5da93cb10b8'
BASE_REPO = 'unsloth/gemma-4-26B-A4B-it-GGUF'
BASE_REVISION = 'c099eb48e663fd284577b04978a94ffccb261841'
BASE_FILENAME = 'gemma-4-26B-A4B-it-UD-IQ4_NL.gguf'
MODEL = ROOT / 'models/gemma-4-26B-A4B-it-UD-IQ4_NL-lcb100-activity-ordered-e80.gguf'
ENGINE = ROOT / '.cache/llama-build/bin/llama-expert-research'
RUN = ROOT / 'runs/gemma4-e80-lcb175-001'
EXPORT = ROOT / 'runs/gemma4-ordered-e80-001/export-manifest.json'
PERMUTATION = ROOT / 'runs/gemma4-lcb100-reordered-001/permutation-manifest.json'


def config(engine=ENGINE):
    from moe_prune.base_benchmark import DEFAULT_CONFIG
    return {**DEFAULT_CONFIG, 'context_size': 262144, 'max_tokens': 65536,
            'n_gpu_layers': 31, 'engine': str(engine), 'fusion_enabled': True,
            'instrumentation_level': 'routing'}


def make_request(record, model, engine=ENGINE):
    from moe_prune.base_benchmark import build_request
    return {**build_request(record, model, (31, True), config(engine)),
            'fusion_enabled': True, 'instrumentation_level': 'routing'}


def _profile(model_profile):
    if model_profile not in ('e80', 'base'):
        raise ValueError('알 수 없는 모델 프로필')
    return (MODEL_SHA, 80) if model_profile == 'e80' else (BASE_MODEL_SHA, 128)


# 2026-11-10: 원본과 축소 모델을 서로 혼동하지 않도록 해시와 구조를 함께 고정한다.
def check_model(model, *, model_profile='e80'):
    expected_sha, experts = _profile(model_profile)
    model = Path(model)
    if not model.is_file():
        raise FileNotFoundError(f'모델 파일 없음: {model}')
    if sha256(model) != expected_sha or GGUF(model).architecture() != (list(range(30)), experts, 8):
        raise ValueError(f'{model_profile} 모델 해시 또는 구조 불일치')


def download_base_model(target):
    target = Path(target)
    if target.name != BASE_FILENAME and not target.exists():
        raise ValueError('다운로드 대상 파일명이 원본 모델과 다릅니다')
    if not target.exists():
        from huggingface_hub import hf_hub_download
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(hf_hub_download(repo_id=BASE_REPO, filename=BASE_FILENAME,
                                          revision=BASE_REVISION, local_dir=str(target.parent), token=False))
        if downloaded.resolve() != target.resolve():
            raise ValueError('다운로드 경로가 예상과 다릅니다')
    check_model(target, model_profile='base')
    return target


# 2026-11-09: 175문항 데이터와 모델 pruning에 사용한 별도 프로파일의 출처를 혼동하지 않는다.
def source_identity(model, engine, *, model_profile='e80'):
    from moe_prune.benchmark_data import load_snapshot, validate_id_proof
    expected_sha, experts = _profile(model_profile)
    check_model(model, model_profile=model_profile)
    engine = Path(engine)
    if not engine.is_file():
        raise FileNotFoundError(f'엔진 파일 없음: {engine}')
    records, manifest = load_snapshot(SNAPSHOT)
    proof = validate_id_proof(SNAPSHOT)
    if sha256(SNAPSHOT / 'records.jsonl') != RECORDS_SHA:
        raise ValueError('고정 175문항 snapshot 해시 불일치')
    records = [r for r in records if r['domain'] == 'code']
    if (len(records) != 175 or len({r['id'] for r in records}) != 175
            or len({str(r['lcb_problem']['question_id']) for r in records}) != 175
            or any(r['id'] != 'livecodebench:' + str(r['lcb_problem']['question_id']) for r in records)):
        raise ValueError('코드 175문항 ID 불일치')
    provenance = {}
    if model_profile == 'e80':
        export = json.loads(EXPORT.read_text(encoding='utf-8'))
        permutation = json.loads(PERMUTATION.read_text(encoding='utf-8'))
        if (export.get('output_sha256') != MODEL_SHA or export.get('file_validation_status') != 'passed'
                or export.get('reorder_manifest_sha256') != sha256(PERMUTATION)):
            raise ValueError('모델 export/permutation 출처 불일치')
        provenance = {'export_sha256': sha256(EXPORT), 'permutation_sha256': sha256(PERMUTATION),
                      'pruning_profile_records_sha256': permutation['profile']['records_sha256']}
    else:
        provenance = {'model_profile': 'base', 'experts': experts,
                      'huggingface': {'repo': BASE_REPO, 'revision': BASE_REVISION, 'filename': BASE_FILENAME}}
    source_paths = ['moe_prune/base_benchmark.py', 'moe_prune/benchmark_data.py',
                    'moe_prune/gguf.py', 'moe_prune/grading.py', 'moe_prune/research.py',
                    'moe_prune/resident.py', 'moe_prune/execution_safety.py',
                    'moe_prune/memory_safety.py', 'tools/benchmark175.py',
                    'tools/lcb_grade.py', 'tools/lcb_prepare.py', 'tools/colab_setup.py',
                    'tools/colab_archive.py', 'tools/native/expert_research.cpp',
                    'tools/native/CMakeLists.txt', 'tools/native/prepare.py',
                    'tools/native/gpu_memory.py', 'moe_prune/datasets.py',
                    'moe_prune/__init__.py', 'requirements-research.txt']
    engine_files = [engine, *sorted(p for p in engine.parent.glob('*.so*')
                                     if p.name.lower().startswith(('libllama', 'libggml')))]
    identity = {'model_sha256': expected_sha, 'model_path': str(Path(model).resolve()),
                'engine_files': {p.name: sha256(p) for p in engine_files},
                'engine_path': str(engine.resolve()), 'snapshot_records': RECORDS_SHA,
                'snapshot_manifest': sha256(SNAPSHOT / 'manifest.json'),
                'id_proof': canonical_hash(proof), 'ids': [r['id'] for r in records],
                **provenance,
                'source_hashes': {p: sha256(ROOT / p) for p in source_paths},
                'config': config(engine), 'placement': [31, True],
                'official_identity': manifest['official_identity'],
                'limitations': ['계측 callback 경계에서는 일부 CUDA fusion이 제한됩니다.',
                                '계측은 속도에 영향을 주며 GPU 메모리는 표본 관측입니다.']}
    return records, identity


def _digest_stream(stream):
    h = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        h.update(chunk)
    return h.hexdigest()


# 2026-11-09: 대용량 원본을 스트리밍 압축하고 복원 해시를 확인한 뒤 중복 원본만 제거한다.
def compress_events(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open('rb') as inp, target.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=3) as out:
        while chunk := inp.read(1024 * 1024):
            digest.update(chunk)
            out.write(chunk)
    with gzip.open(target, 'rb') as inp:
        if _digest_stream(inp) != digest.hexdigest():
            raise ValueError('압축 원본 복원 해시 불일치')
    return {'uncompressed_sha256': digest.hexdigest(), 'compressed_sha256': sha256(target)}


def _verified_answer(directory, number, record, identity_hash):
    directory = Path(directory)
    path = directory / 'answers' / f'{number:03d}.json'
    answer = json.loads(path.read_text(encoding='utf-8'))
    content = {k: v for k, v in answer.items() if k != 'answer_sha256'}
    if (answer.get('answer_sha256') != canonical_hash(content) or answer.get('id') != record['id']
            or answer.get('identity_hash') != identity_hash):
        raise ValueError('완료 답안의 해시 또는 실행 신원 불일치')
    job = answer.get('job', '')
    if not re.fullmatch(r'problem-\d{3}-attempt-\d{3}', job) or not job.startswith(f'problem-{number:03d}-'):
        raise ValueError('작업 경로 오류')
    routing = directory / 'routing' / (job + '.jsonl.gz')
    if routing.is_symlink() or not routing.resolve().is_relative_to(directory.resolve()):
        raise ValueError('계측 경로 범위 오류')
    integrity = answer['routing_integrity']
    if sha256(routing) != integrity['compressed_sha256']:
        raise ValueError('압축 계측 해시 불일치')
    with gzip.open(routing, 'rb') as stream:
        if _digest_stream(stream) != integrity['uncompressed_sha256']:
            raise ValueError('압축 계측 복원 해시 불일치')
    native = native_result(routing)
    if (native != {k: v for k, v in answer['result'].items() if k != 'measurement'}
            or answer['text'] != native.get('text')):
        raise ValueError('압축 원본과 답안의 native 결과 불일치')
    identity = json.loads((directory / 'identity.json').read_text(encoding='utf-8'))
    if canonical_hash(identity) != identity_hash:
        raise ValueError('실행 신원 불일치')
    folder = directory / 'jobs' / job
    if folder.is_symlink() or not folder.resolve().is_relative_to(directory.resolve()):
        raise ValueError('작업 경로 범위 오류')
    request = json.loads((folder / 'request.json').read_text(encoding='utf-8'))
    if request != make_request(record, identity['model_path'], identity['engine_path']):
        raise ValueError('기존 작업 요청 불일치')
    job_identity = json.loads((folder / 'identity.json').read_text(encoding='utf-8'))
    if (job_identity['model']['sha256'] != identity['model_sha256'] or
            job_identity['engine']['sha256'] != identity['engine_files'][Path(identity['engine_path']).name]):
        raise ValueError('기존 작업 모델 또는 엔진 불일치')
    payload = json.loads((directory / 'grade-inputs' / f'{number:03d}.json').read_text(encoding='utf-8'))
    if payload != {'id': record['id'], 'problem': record['lcb_problem'], 'text': answer['text']}:
        raise ValueError('채점 입력 불일치')
    return answer


def _validate_job(directory, name, request, result, *, model_profile='e80'):
    from moe_prune.base_benchmark import validate_routing, verify_runtime
    folder = Path(directory) / 'jobs' / name
    if (result.get('fusion_enabled') is not True or result.get('gpu_fusion_disabled') is not False
            or result.get('instrumentation_level') != 'routing' or result.get('masked_layers')
            or result.get('profiled_layers') != list(range(30)) or result.get('enable_thinking') is not True
            or result.get('stop_reason') not in ('eog', 'length')
            or (result.get('stop_reason') == 'length' and len(result.get('token_ids', [])) != 65536)):
        raise ValueError('실제 생성 또는 fusion/계측 설정 불일치')
    observed = verify_runtime(result, (folder / 'stderr.log').read_text(encoding='utf-8', errors='replace'), request)
    if observed.get('offloaded_layers') != {'gpu': 31, 'total': 31}:
        raise ValueError('실제 GPU 전체 레이어 배치 미확인')
    count = validate_routing(read_jsonl(folder / 'events.jsonl'), result, experts=_profile(model_profile)[1])
    if result.get('routing_records') != count:
        raise ValueError('라우팅 이벤트 수 불일치')
    return observed


def gpu_info():
    from tools.native.gpu_memory import NvmlDevice
    device = NvmlDevice()
    try:
        major, minor = device.nvml.nvmlDeviceGetCudaComputeCapability(device.handle)
        return {'name': device.name, 'uuid': device.uuid, 'compute_capability': f'{major}.{minor}'}
    finally:
        device.close()


def _write_status(directory, completed):
    write_json(Path(directory) / 'status.json', {'complete': completed == 175, 'completed': completed, 'total': 175})


# 2026-11-10: 비용이 큰 모델을 적재하기 전에 백업 경로의 실제 쓰기 실패를 확인한다.
def preflight_checkpoint(destination):
    import os
    import tempfile
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(dir=destination) as probe:
        probe.write(b'putare-checkpoint')
        probe.flush()
        os.fsync(probe.fileno())


# 2026-11-09: 실패한 job을 덮어쓰지 않고 새로운 시도로 보존하며 완결 답안만 재사용한다.
def generate(directory=RUN, *, model=MODEL, engine=ENGINE, checkpoint_dir=None, model_profile='e80'):
    from moe_prune.resident import ResidentExperiment
    _profile(model_profile)
    if model_profile == 'base' and checkpoint_dir is None:
        raise ValueError('원본 모델 평가는 백업 경로가 필수입니다')
    if model_profile == 'base':
        preflight_checkpoint(checkpoint_dir)
    directory, model, engine = Path(directory), Path(model), Path(engine)
    records, identity = (source_identity(model, engine) if model_profile == 'e80'
                         else source_identity(model, engine, model_profile=model_profile))
    identity_hash = canonical_hash(identity)
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / 'identity.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('기존 실행과 모델·데이터·설정·엔진·소스가 다릅니다')
    else:
        if any(directory.iterdir()):
            raise ValueError('신원 없는 기존 실행 디렉터리')
        write_json(identity_path, identity)
    gpu = gpu_info()
    observations = directory / 'gpu-observations'
    observations.mkdir(exist_ok=True)
    observation_number = 1
    while (observations / f'{observation_number:03d}.json').exists():
        observation_number += 1
    observation_path = f'gpu-observations/{observation_number:03d}.json'
    write_json(directory / observation_path, gpu)
    completed = 0
    try:
        with ResidentExperiment(directory, config(engine)) as exp:
            for number, record in enumerate(records, 1):
                answer_path = directory / 'answers' / f'{number:03d}.json'
                if answer_path.exists():
                    answer = _verified_answer(directory, number, record, identity_hash)
                    completed += 1
                    if checkpoint_dir is not None:
                        checkpoint_problem(directory, checkpoint_dir, number, answer)
                    continue
                attempt = 1
                while (directory / 'jobs' / f'problem-{number:03d}-attempt-{attempt:03d}').exists():
                    attempt += 1
                name = f'problem-{number:03d}-attempt-{attempt:03d}'
                request = make_request(record, model, engine)
                print(f'{number}/175 {record["id"]}', flush=True)
                result = exp.run(name, request)
                observed = (_validate_job(directory, name, request, result) if model_profile == 'e80'
                            else _validate_job(directory, name, request, result, model_profile=model_profile))
                raw = directory / 'jobs' / name / 'events.jsonl'
                target = directory / 'routing' / (name + '.jsonl.gz')
                integrity = compress_events(raw, target)
                exp.check()
                payload = {'id': record['id'], 'problem': record['lcb_problem'], 'text': result['text']}
                write_json(directory / 'grade-inputs' / f'{number:03d}.json', payload)
                answer = {'id': record['id'], 'identity_hash': identity_hash, 'text': result['text'],
                          'result': result, 'gpu': gpu, 'gpu_observation': observation_path,
                          'observed_runtime': observed, 'routing_integrity': integrity,
                          'job': name}
                answer['answer_sha256'] = canonical_hash(answer)
                write_json(answer_path, answer)
                _verified_answer(directory, number, record, identity_hash)
                raw.unlink()
                completed += 1
                _write_status(directory, completed)
                if checkpoint_dir is not None:
                    checkpoint_problem(directory, checkpoint_dir, number, answer)
        _write_status(directory, completed)
    except BaseException:
        _write_status(directory, completed)
        raise
    return directory




# 2026-11-09: 부분 실행이나 채점 인프라 실패를 전체 점수 또는 모델 오답으로 표시하지 않는다.
def grade(directory=RUN, *, model=MODEL, engine=ENGINE, checkpoint_dir=None, model_profile='e80'):
    from moe_prune.execution_safety import execution_guard
    from moe_prune.grading import GradeDeadline, GradingCleanupError, run, worker_environment
    from tools.lcb_prepare import COMMIT
    _profile(model_profile)
    if model_profile == 'base':
        if checkpoint_dir is None:
            raise ValueError('원본 모델 평가는 백업 경로가 필수입니다')
        preflight_checkpoint(checkpoint_dir)
    directory = Path(directory)
    status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    if status != {'complete': True, 'completed': 175, 'total': 175}:
        raise ValueError('175문항 생성 미완료')
    records, identity = (source_identity(model, engine) if model_profile == 'e80'
                         else source_identity(model, engine, model_profile=model_profile))
    if json.loads((directory / 'identity.json').read_text(encoding='utf-8')) != identity:
        raise ValueError('채점 시 실행 신원 불일치')
    for number, record in enumerate(records, 1):
        answer = _verified_answer(directory, number, record, canonical_hash(identity))
        if model_profile == 'base':
            checkpoint_problem(directory, checkpoint_dir, number, answer)
    manifest = json.loads((SNAPSHOT / 'manifest.json').read_text(encoding='utf-8'))
    if sha256(ROOT / 'tools/lcb_grade.py') != manifest['grader_identity']['tools/lcb_grade.py']:
        raise ValueError('snapshot 채점 어댑터 해시 불일치')
    source = ROOT / '.cache/putare-lcb/livecodebench' / COMMIT
    venv = ROOT / '.cache/putare-lcb/venv'
    adapter = ROOT / 'tools/lcb_grade.py'
    deadline = GradeDeadline()
    grading_root = directory / 'grading'
    grading_root.mkdir(exist_ok=True)
    attempt = 1
    while (grading_root / f'attempt-{attempt:03d}').exists():
        attempt += 1
    out = grading_root / f'attempt-{attempt:03d}'
    out.mkdir()
    with execution_guard('grading', memory_config=config(engine)) as guard:
        probe = subprocess.run([str(venv / 'bin/python'), '-I', '-B', str(adapter), '--verify',
                                '--official-source', str(source)], env=worker_environment(),
                               check=True, capture_output=True, text=True, timeout=deadline.remaining())
        if json.loads(probe.stdout)['source_digest'] != identity['official_identity']['source_digest']:
            raise ValueError('공식 채점기 source 해시 불일치')
        write_json(out / 'identity.json', {'run_identity_sha256': sha256(directory / 'identity.json'),
                                            'official_source_digest': identity['official_identity']['source_digest']})
        rows = []
        for number, record in enumerate(records, 1):
            target = out / f'{number:03d}.json'
            if target.exists():
                raise FileExistsError('기존 채점 결과를 보존합니다: ' + str(target))
            try:
                result = run(source, venv, directory / 'grade-inputs' / f'{number:03d}.json', target,
                             adapter=adapter, guard=guard, timeout_s=deadline.remaining())
                verdict = {'status': 'graded', 'passed': result['passed']}
            except GradingCleanupError:
                raise
            except Exception as error:
                verdict = {'status': 'infrastructure_error', 'passed': None, 'error_type': type(error).__name__}
            rows.append({'id': record['id'], **verdict})
            verdict_path = out / f'verdict-{number:03d}.json'
            write_json(verdict_path, rows[-1])
            write_json(out / 'summary.json', {'complete': False, 'cases': rows})
            if checkpoint_dir is not None:
                paths = ['identity.json', str((out / 'identity.json').relative_to(directory)),
                         str(verdict_path.relative_to(directory))]
                if target.is_file():
                    paths.append(str(target.relative_to(directory)))
                checkpoint_files(directory, checkpoint_dir, paths,
                                 str((out / f'checkpoint-{number:03d}.json').relative_to(directory)))
            if verdict['status'] != 'graded':
                break
    complete = len(rows) == 175 and all(row['status'] == 'graded' for row in rows)
    summary = {'complete': complete, 'cases': rows}
    if complete:
        summary['passed'] = sum(row['passed'] is True for row in rows)
        summary['total'] = 175
    write_json(out / 'summary.json', summary)
    if checkpoint_dir is not None:
        checkpoint_files(directory, checkpoint_dir, ['identity.json',
                         str((out / 'summary.json').relative_to(directory))],
                         str((out / 'checkpoint-summary.json').relative_to(directory)))
    return summary


# 2026-11-09: 저장한 답안이 압축 원본의 마지막 native 결과와 같은지 재확인한다.
def native_result(path):
    result = None
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            if result is not None:
                raise ValueError('native 결과 뒤에 추가 이벤트가 있습니다')
            if row.get('type') == 'result':
                result = row
            elif row.get('type') != 'routing':
                raise ValueError('알 수 없는 native 이벤트')
    if result is None:
        raise ValueError('native 결과 누락')
    return result








# 2026-11-09: 불완전한 Drive 복사본을 완료로 오인하지 않도록 파일별 해시를 확인한 뒤 marker를 게시한다.
def checkpoint_files(source, destination, files, marker):
    import os
    import shutil
    import tempfile
    source, destination = Path(source).resolve(), Path(destination).resolve()
    def checked(root, name):
        path = root / name
        if Path(name).is_absolute() or not path.resolve().is_relative_to(root) or path.is_symlink():
            raise ValueError('checkpoint 경로 오류')
        return path
    inventory = {}
    for name in files:
        src, dst = checked(source, name), checked(destination, name)
        if not src.is_file():
            raise FileNotFoundError(src)
        expected = sha256(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            if not dst.is_file() or sha256(dst) != expected:
                raise ValueError(f'checkpoint 파일 충돌: {name}')
        else:
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(dir=dst.parent, prefix='.putare-copy-', delete=False) as out:
                    tmp = Path(out.name)
                    with src.open('rb') as inp:
                        shutil.copyfileobj(inp, out, length=1024 * 1024)
                    out.flush()
                    os.fsync(out.fileno())
                if sha256(tmp) != expected:
                    raise ValueError(f'checkpoint 복사 해시 불일치: {name}')
                if dst.exists():
                    raise FileExistsError(dst)
                os.replace(tmp, dst)
            finally:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
        inventory[name] = expected
    target = checked(destination, marker)
    if target.exists():
        if json.loads(target.read_text(encoding='utf-8')) != inventory:
            raise ValueError('checkpoint 완료 marker 불일치')
    else:
        write_json(target, inventory)


def checkpoint_problem(directory, destination, number, answer):
    observation = Path(directory) / answer['gpu_observation']
    if (observation.is_symlink() or not observation.resolve().is_relative_to(Path(directory).resolve())
            or json.loads(observation.read_text(encoding='utf-8')) != answer['gpu']):
        raise ValueError('GPU 관측 기록 불일치')
    job = answer['job']
    files = ['identity.json', f'answers/{number:03d}.json', f'grade-inputs/{number:03d}.json',
             f'routing/{job}.jsonl.gz', answer['gpu_observation'],
             *[f'jobs/{job}/{name}' for name in ('request.json', 'identity.json', 'measurement.json', 'stderr.log')]]
    checkpoint_files(directory, destination, files, f'problems/{number:03d}.json')




# 2026-11-09: 완료 marker에 결속된 파일만 검증해 빈 로컬 실행 폴더로 복원한다.
def restore_checkpoint(checkpoint_dir, directory):
    backup, directory = Path(checkpoint_dir).resolve(), Path(directory).resolve()
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError('복원 대상은 빈 디렉터리여야 합니다')
    markers = sorted((backup / 'problems').glob('*.json'))
    markers += sorted((backup / 'grading').glob('attempt-*/checkpoint-*.json'))
    if not markers:
        raise ValueError('복원할 완료 checkpoint가 없습니다')
    inventories = []
    for marker in markers:
        if marker.is_symlink() or not marker.resolve().is_relative_to(backup):
            raise ValueError('checkpoint marker 경로 오류')
        inventory = json.loads(marker.read_text(encoding='utf-8'))
        if not isinstance(inventory, dict) or 'identity.json' not in inventory:
            raise ValueError('checkpoint marker 형식 오류')
        for name, digest in inventory.items():
            path = backup / name
            if (Path(name).is_absolute() or path.is_symlink() or not path.resolve().is_relative_to(backup)
                    or not path.is_file() or sha256(path) != digest):
                raise ValueError(f'checkpoint 복원 해시 오류: {name}')
        inventories.append((marker.relative_to(backup), inventory))
    for marker, inventory in inventories:
        checkpoint_files(backup, directory, list(inventory), str(marker))
    return directory
