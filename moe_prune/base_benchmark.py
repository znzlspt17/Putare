"""02번 벤치마크의 생성, 필수 계측, 재개 및 요약."""
from pathlib import Path
from collections import defaultdict
import gzip
import hashlib
import html
import json
import math
import os
import re
import shutil

from .gguf import GGUF, sha256
from .research import Experiment, canonical_hash, read_jsonl, write_json, write_jsonl
from .resident import ResidentExperiment, SessionCleanupError


DEFAULT_CONFIG = {'seed': 42, 'context_size': 65535, 'max_tokens': 32768,
                  'n_gpu_layers': 31, 'ram_usage_warn_fraction': .80,
                  'vram_usage_warn_fraction': .97}
PLACEMENTS = ((31, True),)


def run_with_offload(invoke, start=0):
    """2026-11-09: Colab 비교 실험은 전체 GPU 배치를 유지하고 실패를 전파한다."""
    if start != 0:
        raise ValueError('전체 GPU 배치만 지원합니다')
    return invoke(PLACEMENTS[0]), 0


def load_completed(path, record_id, identity_hash, run_dir):
    answer = json.loads(Path(path).read_text(encoding='utf-8'))
    content = {key: value for key, value in answer.items() if key != 'answer_sha256'}
    if answer.get('answer_sha256') != canonical_hash(content):
        raise ValueError('기존 답안 내용이 변경되었습니다')
    if answer.get('id') != record_id or answer.get('identity_hash') != identity_hash:
        raise ValueError('기존 답안의 문제 ID 또는 실행 조건이 다릅니다')
    root = Path(run_dir).resolve()
    routing = (root / answer['routing_path']).resolve()
    try:
        routing.relative_to(root)
    except ValueError as error:
        raise ValueError('계측 경로가 실행 폴더를 벗어났습니다') from error
    if not routing.is_file() or sha256(routing) != answer.get('routing_sha256'):
        raise ValueError('기존 계측 파일이 없거나 변경되었습니다')
    return answer


def verify_runtime(result, stderr, request):
    """2026-11-06: 요청 필드만이 아니라 실제 llama backend 로그의 cache/FA를 확인한다."""
    if not result.get('instrumentation') or not result.get('flash_attention_requested'):
        raise ValueError('계측 또는 Flash Attention 요청 누락')
    if result.get('offload_kqv_requested') != request['offload_kqv']:
        raise ValueError('KV 배치 요청 불일치')
    if result.get('actual_context_size', 0) < request['context_size']:
        raise ValueError('실제 context가 요청보다 작습니다')
    sampling = result.get('sampling', {})
    if (sampling.get('strategy') != 'sampled' or sampling.get('top_k') != 64
            or not math.isclose(sampling.get('temperature', -1), 1.0, abs_tol=1e-6)
            or not math.isclose(sampling.get('top_p', -1), .95, abs_tol=1e-6)
            or result.get('seed') != request.get('seed', 42)):
        raise ValueError('요청한 샘플링이 적용되지 않았습니다')
    if not re.search(r'llama_context:\s*flash_attn\s*=\s*enabled', stderr, re.I):
        raise ValueError('실제 Flash Attention 활성화 로그가 없습니다')
    pairs = re.findall(r'K\s*\(([^)]+)\)[^\n]*V\s*\(([^)]+)\)', stderr, re.I)
    if not pairs or any(k.lower() != 'q8_0' or v.lower() != 'q8_0' for k, v in pairs):
        raise ValueError('실제 Q8/Q8 KV cache 로그가 없거나 형식이 다릅니다')
    buffers = [{'backend': name.strip(), 'mib': float(size)} for name, size in
               re.findall(r'llama_kv_cache:\s*([^\n:]+?)\s+KV buffer size\s*=\s*([\d.]+)\s*MiB', stderr, re.I)]
    weights = [{'backend': name.strip(), 'mib': float(size)} for name, size in
               re.findall(r'load_tensors:\s*([^\n:]+?)\s+model buffer size\s*=\s*([\d.]+)\s*MiB', stderr, re.I)]
    offloaded = re.search(r'load_tensors:\s*offloaded\s+(\d+)/(\d+)\s+layers to GPU', stderr, re.I)
    return {'flash_attention': True, 'kv_cache_types': pairs, 'kv_buffers_mib': buffers,
            'model_buffers_mib': weights,
            'offloaded_layers': {'gpu': int(offloaded.group(1)), 'total': int(offloaded.group(2))} if offloaded else None,
            'actual_context_size': result['actual_context_size']}


def build_request(record, model, placement, config):
    """2026-11-06: 정답/테스트는 본체 추론 요청에 포함하지 않는다."""
    c = {**DEFAULT_CONFIG, **config}
    layers, offload_kqv = placement
    request = {'model': str(model), 'prompt': record['prompt'], 'mode': 'profile',
               'instrumentation': True, 'max_tokens': c['max_tokens'],
               'context_size': c['context_size'], 'n_gpu_layers': layers,
               'offload_kqv': offload_kqv, 'cache_type_k': 'q8_0', 'cache_type_v': 'q8_0',
               'flash_attention': True, 'enable_thinking': True, 'apply_chat_template': True,
               'all_token_logits': True,
               'seed': c['seed'], 'sampling': {'temperature': 1.0, 'top_p': .95, 'top_k': 64},
               'max_routing_records': c['context_size'] * 30}
    if record.get('system_prompt'):
        request['system_prompt'] = record['system_prompt']
    return request


def validate_routing(events, result, layers=30, experts=128, top_k=8):
    """2026-11-06: 모든 forward 토큰의 각 레이어를 한 번씩 확인한다."""
    prompt = int(result['prompt_tokens'])
    decode = int(result['processed_decode_tokens'])
    if prompt <= 0 or decode < 0:
        raise ValueError('잘못된 처리 토큰 수')
    if 'token_ids' in result and result.get('stop_reason') in ('length', 'eog'):
        generated = len(result['token_ids'])
        expected_decode = generated - int(result['stop_reason'] == 'length' and generated > 0)
        if decode != expected_decode:
            raise ValueError('생성 토큰과 계측된 decode 토큰 수 불일치')
    expected = {(phase, layer): (0 if phase == 'prefill' else prompt)
                for phase in ('prefill', 'decode') for layer in range(layers)}
    count = 0
    for row in events:
        if row.get('type') != 'routing':
            continue
        phase, layer = row.get('phase'), row.get('layer')
        key = (phase, layer)
        if key not in expected or type(layer) is not int or row.get('token_index') != expected[key]:
            raise ValueError(f'계측 위치 누락 또는 중복: {key}')
        ids, weights, norms = row.get('expert_ids'), row.get('weights'), row.get('weighted_output_norms')
        if not all(isinstance(x, list) and len(x) == top_k for x in (ids, weights, norms)):
            raise ValueError('계측 벡터 형상 오류')
        if len(set(ids)) != top_k or any(type(x) is not int or not 0 <= x < experts for x in ids):
            raise ValueError('Expert ID 오류')
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) or not math.isfinite(x) or x < 0
               for x in weights + norms):
            raise ValueError('계측값 오류')
        expected[key] += 1
        count += 1
    for (phase, layer), end in expected.items():
        target = prompt if phase == 'prefill' else prompt + decode
        if end != target:
            raise ValueError(f'계측 누락: {phase}/{layer} {end} != {target}')
    if count != layers * (prompt + decode):
        raise ValueError('계측 개수 불일치')
    return count


def aggregate_profiles(samples, layers=30, experts=128):
    """2026-11-06: 문제별 평균과 토큰 평균을 따로 계산해 긴 답안의 쏠림을 드러낸다."""
    totals = {}
    def blank():
        return {'tokens': 0, 'counts': [0] * experts, 'gate_sum': [0.] * experts,
                'norm_sum': [0.] * experts, 'problem_rate_sum': [0.] * experts, 'problems': 0}
    for domain, events in samples:
        per_sample = {}
        for row in events:
            if row.get('type') != 'routing':
                continue
            key = (domain, row['phase'], str(row['layer']))
            item = per_sample.setdefault(key, blank())
            item['tokens'] += 1
            for expert, weight, norm in zip(row['expert_ids'], row['weights'], row['weighted_output_norms']):
                item['counts'][expert] += 1
                item['gate_sum'][expert] += weight
                item['norm_sum'][expert] += norm
        for key, item in per_sample.items():
            target = totals.setdefault(key, blank())
            target['tokens'] += item['tokens']
            target['problems'] += 1
            for expert in range(experts):
                target['counts'][expert] += item['counts'][expert]
                target['gate_sum'][expert] += item['gate_sum'][expert]
                target['norm_sum'][expert] += item['norm_sum'][expert]
                target['problem_rate_sum'][expert] += item['counts'][expert] / item['tokens']
    result = {}
    for (domain, phase, layer), item in totals.items():
        counts, tokens, n = item['counts'], item['tokens'], item.pop('problems')
        item['problem_count'] = n
        item['token_rate'] = [v / tokens for v in counts]
        item['problem_mean_rate'] = [v / n for v in item.pop('problem_rate_sum')]
        item['mean_gate_per_selection'] = [v / count if count else None for v, count in zip(item['gate_sum'], counts)]
        item['mean_norm_per_selection'] = [v / count if count else None for v, count in zip(item['norm_sum'], counts)]
        result.setdefault(domain, {}).setdefault(phase, {})[layer] = item
    return result


class BenchmarkRun:
    def __init__(self, config, model, engine, snapshot, run_dir):
        self.config, self.model, self.engine = {**DEFAULT_CONFIG, **config}, Path(model), Path(engine)
        self.snapshot, self.run_dir = Path(snapshot), Path(run_dir)

    def run(self, smoke=False):
        """2026-11-06: 완료된 문제만 재사용하며 미완료는 점수에 포함하지 않는다."""
        records_path = self.snapshot / 'records.jsonl'
        manifest_path = self.snapshot / 'manifest.json'
        from .benchmark_data import grader_identity, load_snapshot, validate_id_proof
        records, manifest = load_snapshot(self.snapshot)
        if len({r['id'] for r in records}) != len(records):
            raise ValueError('중복 문제 ID')
        counts = {domain: sum(r['domain'] == domain for r in records) for domain in ('math', 'code')}
        if counts != {'math': 30, 'code': 175} or len(records) != 205:
            raise ValueError(f'평가 snapshot 문항 수 불일치: {counts}')
        if smoke:
            records = [next(r for r in records if r['domain'] == d) for d in ('math', 'code')]
        else:
            try:
                proof = validate_id_proof(self.snapshot)
            except FileNotFoundError as error:
                raise FileNotFoundError(f'{error} — 노트북의 v5 원본 ID 검증 준비 셀을 먼저 실행하세요') from error
            # 2026-11-09: snapshot의 공식 formatter는 유지하면서 새 실행기의 identity는 별도로 결속한다.
            root = Path(__file__).resolve().parents[1]
            current_grader = grader_identity(root)
            if manifest.get('grader_identity', {}).get('tools/lcb_grade.py') != current_grader['tools/lcb_grade.py']:
                raise ValueError('공식 채점 어댑터가 snapshot과 다릅니다')
        original = GGUF(self.model)
        layers, experts, top_k = original.architecture()
        if layers != list(range(30)) or experts != 128 or top_k != 8:
            raise ValueError('예상 Gemma4 Expert 구조와 다릅니다')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        identity = {'config': self.config, 'scope': 'smoke' if smoke else 'benchmark',
                    'execution_mode': 'resident-v1', 'context_reset': 'kv-sampler-observer-v1',
                    'model': sha256(self.model), 'engine': sha256(self.engine),
                    'snapshot_manifest': sha256(manifest_path), 'snapshot_records': sha256(records_path)}
        if not smoke:
            from .benchmark_data import official_identity
            current_official = official_identity(root)
            if current_official.get('source_digest') != manifest.get('official_identity', {}).get('source_digest'):
                raise ValueError('공식 채점기 source가 snapshot과 다릅니다')
            identity.update({'official_identity': current_official, 'grader_identity': current_grader})
            identity['lcb_v5_id_proof'] = canonical_hash(proof)
        identity_hash = canonical_hash(identity)
        identity_path = self.run_dir / 'benchmark-identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
                raise ValueError('모델·데이터·설정이 달라 새 RUN_ID가 필요합니다')
        else:
            write_json(identity_path, identity)
        exp_config = {**self.config, 'engine': str(self.engine.resolve()), 'benchmark_identity': identity_hash}
        answers = []
        placement_index = 0
        generation_path = self.run_dir / 'generations.jsonl'
        with ResidentExperiment(self.run_dir, exp_config) as exp:
            for record in records:
                key = hashlib.sha256(record['id'].encode('utf-8')).hexdigest()
                answer_path = self.run_dir / 'answers' / f'{key}.json'
                if answer_path.exists():
                    answer = load_completed(answer_path, record['id'], identity_hash, self.run_dir)
                    answers.append(answer)
                    placement_index = PLACEMENTS.index(tuple(answer['placement']))
                    continue
                def invoke(placement):
                    # 2026-11-06: 실패 시도 폴더는 보존해 재시도가 이전 결과를 덮지 않도록 한다.
                    placement_key = f'{placement[0]}-{int(placement[1])}'
                    base = f'base-{key[:20]}-{placement_key}'
                    number = 0
                    while (self.run_dir / 'jobs' / f'{base}-{number}').exists():
                        existing = self.run_dir / 'jobs' / f'{base}-{number}'
                        if (existing / 'measurement.json').is_file() and (existing / 'events.jsonl').is_file():
                            break
                        number += 1
                    name = f'{base}-{number}'
                    folder = self.run_dir / 'jobs' / name
                    request = build_request(record, self.model, placement, self.config)
                    try:
                        result = exp.run(name, request)
                    except SessionCleanupError:
                        raise
                    except RuntimeError as error:
                        stderr = folder / 'stderr-failure.log'
                        if stderr.is_file():
                            text = stderr.read_text(encoding='utf-8', errors='replace').lower()
                            if re.search(r'out of memory|hiperroroutofmemory|cannot allocate memory|failed to allocate|allocation failed', text):
                                raise MemoryError(f'native GPU/host allocation failed: {folder}') from error
                        raise
                    return folder, result, placement, request
                try:
                    (folder, result, placement, request), placement_index = run_with_offload(invoke, placement_index)
                    if (result.get('cache_type_k'), result.get('cache_type_v')) != ('q8_0', 'q8_0'):
                        raise ValueError('native KV cache 형식 불일치')
                    if not result.get('instrumentation') or not result.get('flash_attention_requested'):
                        raise ValueError('계측 또는 Flash Attention 요청 누락')
                    if result.get('offload_kqv_requested') != placement[1]:
                        raise ValueError('KV 배치 요청 불일치')
                    if result.get('sampling', {}).get('strategy') != 'sampled':
                        raise ValueError('요청한 샘플링이 적용되지 않았습니다')
                    observed = verify_runtime(result, (folder / 'stderr.log').read_text(encoding='utf-8', errors='replace'), request)
                    observed['evidence_log'] = str((folder / 'stderr.log').relative_to(self.run_dir))
                    events_path = folder / 'events.jsonl'
                    count = validate_routing(read_jsonl(events_path), result)
                    if result.get('routing_records') != count:
                        raise ValueError('native 계측 총수와 검증 개수 불일치')
                    target = self.run_dir / 'routing' / f'{key}.jsonl.gz'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_suffix('.gz.tmp')
                    digest = hashlib.sha256()
                    with events_path.open('rb') as source, gzip.open(temporary, 'wb', compresslevel=3) as output:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            output.write(chunk)
                    verified = hashlib.sha256()
                    with gzip.open(temporary, 'rb') as stream:
                        while chunk := stream.read(1024 * 1024):
                            verified.update(chunk)
                    if digest.digest() != verified.digest():
                        raise ValueError('계측 압축 검증 실패')
                    exp.check()
                    temporary.replace(target)
                    answer = {'id': record['id'], 'domain': record['domain'], 'dataset': record['dataset'],
                              **result, 'observed_runtime': observed, 'placement': list(placement), 'identity_hash': identity_hash,
                              'routing_path': str(target.relative_to(self.run_dir)), 'routing_sha256': sha256(target)}
                    answer['answer_sha256'] = canonical_hash(answer)
                    write_json(answer_path, answer)
                    # 압축 내용을 검증하고 답안이 커밋된 다음에만 작업용 원본을 제거한다.
                    events_path.unlink()
                    answers.append(answer)
                    write_jsonl(generation_path, answers)
                    print(f"{len(answers)}/{len(records)} {record['id']} 완료 배치={placement} "
                          f"상주 세션 실행 중 메모리={result.get('measurement', {}).get('memory_last')}", flush=True)
                except BaseException:
                    write_jsonl(generation_path, answers)
                    write_json(self.run_dir / 'benchmark-status.json', {'complete': False, 'completed': len(answers), 'total': len(records)})
                    raise
        write_jsonl(generation_path, answers)
        samples = ((row['domain'], _read_compressed(self.run_dir / row['routing_path'])) for row in answers)
        profiles = aggregate_profiles(samples)
        profile_path = self.run_dir / 'routing-profile.json'
        write_json(profile_path, profiles)
        _write_heatmaps(self.run_dir / 'heatmaps', profiles)
        summary = {'complete': True, 'scope': 'smoke' if smoke else 'benchmark',
                   'completed': len(answers), 'total': len(records),
                   'generations_path': str(generation_path), 'profile_path': str(profile_path)}
        write_json(self.run_dir / 'benchmark-status.json', summary)
        return summary


def _read_compressed(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _write_heatmaps(destination, profiles):
    """2026-11-06: 별도 시각화 의존성 없이 레이어×Expert 비율을 기록한다."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for domain, phases in profiles.items():
        for phase, rows in phases.items():
            for metric in ('token_rate', 'problem_mean_rate'):
                title = f'{domain} / {phase} / {metric}'
                parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="560" viewBox="0 0 1100 560">',
                         '<rect width="1100" height="560" fill="white"/>',
                         f'<text x="70" y="25" font-size="16">{html.escape(title)}</text>',
                         '<text x="70" y="545" font-size="12">Expert ID 0–127; color: 0–100% selected per token</text>']
                for layer in range(30):
                    row = rows.get(str(layer))
                    if not row:
                        continue
                    parts.append(f'<text x="25" y="{47 + layer * 16}" font-size="10">{layer}</text>')
                    for expert, value in enumerate(row[metric]):
                        intensity = max(0, min(255, round(value * 255)))
                        color = f'#{255-intensity:02x}{255-intensity:02x}ff'
                        parts.append(f'<rect x="{70 + expert * 8}" y="{36 + layer * 16}" width="8" height="15" fill="{color}"><title>layer {layer}, expert {expert}: {value*100:.4f}%</title></rect>')
                parts.append('</svg>')
                (destination / f'{domain}-{phase}-{metric}.svg').write_text(''.join(parts), encoding='utf-8')
