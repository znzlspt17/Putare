"""2026-11-05: 결과 없는 연구 노트북을 재현 가능하게 작성한다."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []

def md(source):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(source).strip()))

def code(source):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(source).strip()))


md("""
# Gemma 4 수학·코딩 특화 Expert Pruning 연구

실제 실험은 사용자가 실행합니다. 모든 비용이 큰 단계는 기본 비활성화되어 있습니다.
원본 파일은 보존하며, 결과가 없으면 가설을 지지한다고 결론 내리지 않습니다.
기존 Router가 토큰별로 선택하는 원본 모델이 기본 비교 대상입니다.
""")
md("""
## 1. 가설

**연구 질문:** 원본 Router의 선택과 Expert 출력 기여도를 분석하여 수학·코딩에 필요한 계층별 집합을 오프라인에서 선정하고,
나머지 가중치를 물리적으로 제거하면 품질 손실을 제한하면서 더 적은 메모리에서 실행할 수 있는가?

|가설|지지 근거|반증 또는 보류 조건|
|---|---|---|
|H1 도메인 사용·기여 패턴|일반 텍스트와 분포가 다르고 독립 표본에서 선택 집합이 안정적|분포 차이·안정성이 없거나 표본 부족|
|H2 정적 집합의 품질 보존|고정 집합에서도 held-out 수학·코딩 능력 유지|품질 손상 또는 코딩 실행 평가 부재|
|H3 물리적 경량화|Masked 모델과 대응하고 실제 파일/자원 감소|절삭 동등성 실패 또는 자원 감소 미관측|
|H4 실행 효율|동일 조건 반복에서 TTFT/Decode 개선|개선 없음 또는 배치·계측 조건 불일치|

H1→H2→H3→H4는 자동 함의가 아닙니다. 수학·코딩 공동 모델 하나가 목표이며,
Router 유지·Top-K 유지·양자화 블록 보존이 기본 조건입니다. 무작위 선정은 필수 대조군이 아닙니다.
Router 제거, Top-K 축소, 재학습, 262K 컨텍스트와 KV Cache 최적화는 후속 연구입니다.
현재는 탐색 모드로, 임의의 품질 허용치로 성공을 선언하지 않습니다.
""")
code("""
from pathlib import Path
import json, sys, subprocess, platform, importlib.metadata, shutil
ROOT = Path.cwd()
if ROOT.name == 'ipynb':
    ROOT = ROOT.parent
assert (ROOT / 'moe_prune').is_dir(), '저장소 루트 또는 ipynb 폴더에서 실행하세요'
sys.path.insert(0, str(ROOT))

# 사용자가 단계별로 활성화합니다. Run All은 기본적으로 모델/네트워크/빌드를 실행하지 않습니다.
INSTALL_DEPENDENCIES = False
PREPARE_ENGINE = False
BUILD_ENGINE = False
DOWNLOAD_DATA = False
RUN_MODEL = False
WRITE_MODELS = False
RUN_EVALUATION = False
RUN_BENCHMARK = False

MODEL = ROOT / 'models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf'
ENGINE = ROOT / '.cache/llama-build/bin/llama-expert-research.exe'
RUN_ID = 'gemma4-exploration-001'
DATA = ROOT / 'runs/datasets/snapshot-001'
GPU_TARGET = 'gfx1201'  # 실제 HIP 장치와 일치하는지 실행 전 확인
CONFIG = {'seed': 42, 'context_size': 8192, 'n_gpu_layers': 99,
          'ram_budget_bytes': None,  # 고정 RSS 상한 없이 실제 시스템 여유를 감시
          'min_available_ram_bytes': 4 * 1024**3,
          'min_available_commit_bytes': 4 * 1024**3,
          'min_free_vram_bytes': 1 * 1024**3,
          'vram_workspace_bytes': 1 * 1024**3,
          'engine': str(ENGINE.resolve())}
CALIBRATION_N = 16       # 점검 16 → 탐색 64; 바꾸면 데이터 snapshot과 RUN_ID도 변경
SELECTION_N = 16
EVALUATION_N = 16        # 전체 평가는 None, 별도 RUN_ID로 실행
PROFILE_OUTPUT_TOKENS = 128
GENERATION_TOKENS = 1024
ABLATION_BUDGET = 8      # 일부 후보만 점검, 전체 Critical Expert 검증이 아님
APPROVED_KEEP_COUNTS = [120]  # 120개 대응·품질 검토 후 직접 96/64/32를 추가
PRIOR_SELECTION_REVIEW_HASH = None  # 더 작은 모델은 이전 단계 Selection 실측 요약의 hash를 확인 후 입력
assert len(APPROVED_KEEP_COUNTS) == 1, '연쇄 축소 전 품질 검토를 위해 한 번에 유지 개수 하나만 지정하세요'
print({'root': str(ROOT), 'platform': platform.platform(), 'model_present': MODEL.is_file()})
""")
md("""
### 도구 준비와 실행 계약

Python 데이터 도구, GGUF 바이트 절삭기와 **llama.cpp HIP 실험용 엔진**을 사용합니다. Strata/Unsloth 프레임워크를 실행하지 않습니다.
`requirements-research.txt`는 기존 ROCm lock과 별도입니다. 설치 셀은 사용자 실행 전용입니다.
엔진은 `tools/native/README.md`의 pinned revision을 사용합니다. CMake·C++ 빌드 도구·HIP SDK가 필요합니다.
시간 제한은 없습니다. 작업이 몇 시간 걸려도 메모리 여유가 유지되면 계속합니다.
고정 40GB RSS 상한 대신 시스템 가용 RAM·Windows commit 여유를 각각 4GiB,
GPU 전체 사용량 기준 VRAM 여유를 1GiB 남기도록 시작 전·실행 중 감시합니다.
시작 전에는 GGUF 파일 크기 + workspace 추정 1GiB + VRAM 여유 1GiB를 요구합니다.
workspace는 KV/계산 버퍼의 정확한 필요량이 아니며, 순간 할당과 드라이버 장애까지 보장하지 않습니다.
메모리 관측이 불가능하거나 여유가 부족하면 중단하고 원인을 기록합니다.
이전 실행과 CONFIG가 달라졌다면 새 RUN_ID를 사용하세요. 실패한 작업은 자동 재시도하지 않습니다.
엔진 GPU 할당 실패는 중단하며, CPU fallback으로 속도 결과를 조용히 대체하지 않습니다.
""")
code("""
if INSTALL_DEPENDENCIES:
    uv = shutil.which('uv')
    install_command = ([uv, 'pip', 'install', '--python', sys.executable] if uv else
                       [sys.executable, '-m', 'pip', 'install'])
    subprocess.run(install_command + ['-r', str(ROOT / 'requirements-research.txt')], check=True)
else:
    print('설치 생략. 필요하면 INSTALL_DEPENDENCIES=True로 변경하세요.')

print({name: importlib.metadata.version(name) if importlib.util.find_spec(name) else 'not installed'
       for name in ['numpy', 'psutil', 'nbformat']})
if PREPARE_ENGINE:
    command = [sys.executable, str(ROOT / 'tools/native/prepare.py'), '--gpu-target', GPU_TARGET]
    if BUILD_ENGINE:
        command.append('--execute')
    subprocess.run(command, cwd=ROOT, check=True)
else:
    print('엔진 준비/빌드 생략. tools/native/README.md를 먼저 읽으세요.')
""")
md("""
## 2. 원본 분석

이름의 IQ4_NL만으로 모든 텐서가 동일 형식이라고 가정하지 않습니다.
파일 해시·모델 구조·텐서 종류·형상·바이트·Router hidden scale과 Expert scale을 확인합니다.
원본 해시 계산은 파일 전체를 스트리밍해서 읽으며 추론은 하지 않습니다.
전체 메타데이터/Tokenizer는 JSON에 저장하고 화면에는 요약만 표시합니다.
파일이 복사 중이거나 읽기 권한이 없으면 복사가 끝난 뒤 다시 실행하세요.
""")
code("""
from moe_prune.gguf import GGUF, sha256, validate_allowlist
from moe_prune.research import (Experiment, read_jsonl, write_json, aggregate_routing, select_experts,
    routing_similarity, stability, compare_logits, ablation_sensitivity, summarize, benchmark,
    render_tradeoff, canonical_hash, assert_disjoint, paired_quality)
exp = None
inspection = None
if RUN_MODEL:
    exp = Experiment(ROOT / 'runs' / RUN_ID, CONFIG)
    original = GGUF(MODEL)
    layers, experts, top_k = original.architecture()
    assert experts == 128 and top_k == 8, '예상 구조와 다릅니다. 이 실험 사양을 재검토하세요.'
    inspection = original.inspect(hash_file=True)
    write_json(exp.root / 'original-inspection.json', inspection)
    exp.record('original_metadata', {'sha256': inspection['sha256'], 'experts': experts, 'top_k': top_k,
                                   'layers': layers, 'file_bytes': inspection['file_bytes']})
    print({k: inspection[k] for k in ['sha256', 'file_bytes', 'tensor_groups']})
    print([t for t in inspection['tensors'] if t['name'].startswith('blk.0.') and
           ('exps' in t['name'] or 'gate_inp' in t['name'])])
else:
    print('원본 파일 분석·해시·모델 실행은 아직 수행하지 않았습니다.')
""")
md("""
### 데이터 준비

Calibration/Selection/Evaluation은 ID와 정규화 본문 해시로 분리합니다.
GSM8K/MBPP 학습 분할은 통계·선정에만, 평가 분할/MATH-500/HumanEval+/MBPP+는 최종 관찰에 사용합니다.
일반 텍스트(WikiText)는 분포 대조와 언어 생성 점검에 사용합니다.
HF 데이터셋 revision은 다운로드 시 해소하여 snapshot에 고정합니다. EvalPlus 데이터는
패키지 버전과 선택 문제·본문 해시로 고정하며 테스트 코드를 실행하지 않습니다.
데이터 라이선스가 비어 있으면 배포/재사용 허가로 해석하지 말고 원본 카드에서 확인하세요.
""")
code("""
from moe_prune.datasets import prepare_datasets, load_snapshot
datasets = None
if DOWNLOAD_DATA:
    datasets, dataset_manifest = prepare_datasets(DATA, CALIBRATION_N, SELECTION_N, EVALUATION_N, CONFIG['seed'])
elif (DATA / 'manifest.json').is_file():
    datasets, dataset_manifest = load_snapshot(DATA)
if datasets is not None:
    print({name: len(rows) for name, rows in datasets.items()})
    if exp is not None:
        exp.record('dataset', {'manifest_sha256': sha256(DATA / 'manifest.json'), 'manifest': dataset_manifest})
else:
    print('데이터 없음: 다운로드 셀을 직접 활성화하거나 검증된 snapshot을 준비하세요.')
""")
code("""
# 원본이 먼저 같은 환경에서 추론 가능해야 합니다.
gate_records = None
if RUN_MODEL and datasets is not None:
    selection = datasets['selection']
    gate_records = [next(r for r in selection if r['domain'] == d) for d in ['math', 'code', 'general']]
    baseline_smoke = exp.collect('baseline-smoke', MODEL, gate_records, mode='generate', max_tokens=32)
    print([{k: r.get(k) for k in ['id', 'text', 'timings', 'gpu_memory_bytes', 'cpu_gpu_placement']} for r in baseline_smoke])
""")
md("""
## 3. 가설을 검증할 연결 고리 추적

**도메인 입력 → 원본 Routing → 기여도 → 집합 선정 → 제거 개입 → 출력/자원 변화**

빈도는 상관 근거입니다. Gate Weight와 실제 가중 출력 Norm은 별도 측정합니다.
기여도 callback이 모든 계층을 관측하지 못하면 엔진은 실패하며 빈도로 대체하지 않습니다.
계측 ON/OFF 동등성은 동일 prompt의 최종 logits로 검사하고, 작은 표본의 증거임을 기록합니다.
""")
code("""
profiles = None
if RUN_MODEL and gate_records is not None:
    print(exp.compare('instrumentation', MODEL, MODEL, gate_records,
                      instrumentation_a=False, instrumentation_b=True))
    exp.require('instrumentation')
    calibration = datasets['calibration']
    exp.collect('profile', MODEL, calibration, mode='profile', max_tokens=PROFILE_OUTPUT_TOKENS)
    profiles = aggregate_routing(exp.routing('profile', calibration), experts)
    write_json(exp.root / 'profiles.json', profiles)
    print([{k: r[k] for k in ['domain', 'layer', 'phase', 'tokens', 'contribution_observed']} for r in profiles[:8]])
""")
code("""
if profiles is not None:
    distribution_comparison = {}
    for a, b in [('math', 'code'), ('math', 'general'), ('code', 'general')]:
        distribution_comparison[a + '_vs_' + b] = routing_similarity(
            [r for r in profiles if r['domain'] == a], [r for r in profiles if r['domain'] == b])
    write_json(exp.root / 'domain-distributions.json', distribution_comparison)
    # 독립 표본 반분: 같은 수학/코딩 비중으로 집합 안정성을 검사합니다.
    folds = [calibration[::2], calibration[1::2]]
    fold_profiles = []
    for fold in folds:
        ids = {r['id'] for r in fold}
        fold_profiles.append(aggregate_routing((e for e in exp.routing('profile', calibration) if e['sample_id'] in ids), experts))
    try:
        masks = [select_experts(p, 120, 'frequency') for p in fold_profiles]
        write_json(exp.root / 'selection-stability.json', stability(masks))
        print('분포 차이와 집합 안정성을 저장했습니다. 통계적 전문성의 확정 증거는 아닙니다.')
    except ValueError as error:
        print('표본 분할에서 도메인이 부족합니다:', error)
""")
md("""
### 선행연구와 재현 범위

|연구|참고하는 연결 고리|이번 실험과의 차이/라이선스|
|---|---|---|
|[DEP](https://arxiv.org/html/2610.02951v1)|요청·역할을 이용한 Expert mask|요청별 Predictor 대신 오프라인 고정 집합. 공개 코드/라이선스 미확인|
|[EASY-EP](https://github.com/RUCAIBox/EASYEP)|gate×output 및 representation 기여도|정확한 재현 아님. Windows 지원·코드 재사용 라이선스 미확인|
|[REAP](https://github.com/CerebrasResearch/reap)|Router gate와 activation norm 중요도|개념 참고, 동일 알고리즘 재현 주장 안 함. Apache-2.0|
|[MoE-Infinity](https://github.com/EfficientMoE/MoE-Infinity)|활성 패턴·오프로딩 비용|캐시/Prefetch 구현은 이번 범위 제외. Apache-2.0|

신규성은 아직 주장하지 않습니다. 이번 산출물은 양자화 GGUF에서 관측–Mask–물리 절삭을 연결하는 재현 실험입니다.
""")
md("""
## 4. 가설 구현

모든 계층의 **유지 개수는 같게**, ID는 계층별로 선정합니다. Router 입력 hidden scale과 Shared FFN은 보존합니다.
Router 출력 행·Expert 가중치·Expert scale은 같은 old→new 매핑으로 절삭합니다.
원본 양자화 블록을 복사하며 미지원 tensor/축/정렬은 중단합니다.
""")
code("""
candidates = {}
if profiles is not None:
    for metric in ['frequency', 'contribution']:
        for keep in [128, 120, 96, 64, 32]:
            candidates[f'{metric}-{keep}'] = select_experts(profiles, keep, metric)
    write_json(exp.root / 'candidate-masks.json', candidates)
    full_mask = {layer: list(range(experts)) for layer in layers}
    print(exp.compare('full-mask', MODEL, MODEL, gate_records, mask_b=full_mask))
    exp.require('instrumentation', 'full-mask')
""")
code("""
sensitivity = None
if RUN_MODEL and candidates:
    # NLL 기반 일부 단일 제거 개입. 품질 하락 허용치나 완전한 Critical Expert 탐색을 뜻하지 않습니다.
    sensitivity_records = [r for d in ['math', 'code'] for r in [r for r in datasets['selection'] if r['domain'] == d][:2]]
    sensitivity = ablation_sensitivity(exp, MODEL, sensitivity_records,
        candidates['contribution-120'], experts, ABLATION_BUDGET)
    protected = {}
    for row in sensitivity:
        if row['mean_delta_nll'] > 1e-4:
            protected.setdefault(row['layer'], []).append(row['expert'])
    for keep in [120, 96, 64, 32]:
        candidates[f'impact-{keep}'] = select_experts(profiles, keep, 'contribution', protected)
    write_json(exp.root / 'candidate-masks.json', candidates)
    print(sensitivity)
""")
code("""
selection_loss = {}
if RUN_MODEL and candidates:
    balanced = [r for d in ['math', 'code'] for r in [r for r in datasets['selection'] if r['domain'] == d]]
    reference_original = exp.collect('selection-loss-original', MODEL, balanced, mode='reference_loss', max_tokens=0)
    for keep in APPROVED_KEEP_COUNTS:
        for method in ['frequency', 'impact']:
            key = f'{method}-{keep}'
            masked = exp.collect('selection-loss-' + key, MODEL, balanced, mode='reference_loss', mask=candidates[key], max_tokens=0)
            selection_loss[key] = [{'id': a['id'], 'domain': a['domain'], 'original_nll': a['reference_nll'],
                                    'masked_nll': b['reference_nll'], 'delta_nll': b['reference_nll'] - a['reference_nll']}
                                   for a, b in zip(reference_original, masked)]
    write_json(exp.root / f'selection-reference-loss-{APPROVED_KEEP_COUNTS[0]}.json', selection_loss)
    print(selection_loss)
""")
code("""
noop = ROOT / 'runs' / RUN_ID / 'models/noop.gguf'
if RUN_MODEL and WRITE_MODELS and gate_records is not None:
    exp.require('instrumentation', 'full-mask')
    if not noop.exists():
        write_json(exp.root / 'noop-export.json', original.rewrite(noop))
    print(exp.compare('noop', MODEL, noop, gate_records))
    exp.require('noop')
else:
    print('실제 GGUF 재작성 생략. WRITE_MODELS=True는 사용자 실행 단계입니다.')
""")
code("""
exports = {}
if RUN_MODEL and WRITE_MODELS and candidates:
    exp.require('instrumentation', 'full-mask', 'noop')
    if len(APPROVED_KEEP_COUNTS) != 1:
        raise ValueError('품질 검토 전 연쇄 축소를 막기 위해 한 번에 유지 개수 하나만 실행하세요')
    for keep in APPROVED_KEEP_COUNTS:
        if keep not in [120, 96, 64, 32]:
            raise ValueError('허용한 탐색 grid 밖입니다')
        if keep < 120:
            exp.require('physical-impact-120')
            previous = {96: 120, 64: 96, 32: 64}[keep]
            review_file = exp.root / f'selection-review-{previous}.json'
            if not review_file.is_file() or PRIOR_SELECTION_REVIEW_HASH != sha256(review_file):
                raise RuntimeError('이전 단계 Selection 실측 요약을 검토하고 출력된 hash를 PRIOR_SELECTION_REVIEW_HASH에 입력하세요')
            review = json.loads(review_file.read_text(encoding='utf-8'))
            if any(sha256(Path(path)) != digest for path, digest in review['evidence_sha256'].items()):
                raise RuntimeError('품질 검토 근거 파일이 변경되었습니다')
        # 순서대로 하나씩 생성합니다. 후보별 디스크 필요량을 실행 전 확인하세요.
        for method in ['frequency', 'impact']:
            key = f'{method}-{keep}'
            mask = candidates[key]
            path = exp.root / 'models' / f'{key}-{canonical_hash(mask)[:12]}.gguf'
            if not path.exists():
                write_json(path.with_suffix('.json'), original.rewrite(path, mask))
            proof = exp.compare('physical-' + key, MODEL, path, gate_records, mask_a=mask)
            if not proof['equivalent']:
                raise RuntimeError('Masked 원본과 절삭 모델의 출력 불일치: 확대 실험 중단')
            exports[key] = path
    write_json(exp.root / 'exports.json', {k: str(v) for k, v in exports.items()})
""")
md("""
## 5. 비교

|비교|확인할 효과|
|---|---|
|원본 ↔ 계측/전체 Mask/no-op|연구 도구 자체의 동작 보존|
|원본 ↔ 제한 Mask|고정 집합 제한의 품질 영향|
|제한 Mask ↔ 물리 Pruning|직렬화·재매핑 동등성|
|원본 ↔ 물리 Pruning|최종 크기·품질·자원·속도 변화|
|빈도 선정 ↔ 기여도·민감도 선정|선정 방법의 영향|

Unsloth 공개 성능은 모델·양자화·데이터·Thinking·생성 조건이 맞을 때만 직접 비교합니다.
공식 모델 점수를 UD-IQ4_NL 로컬 실측으로 오인하지 않습니다.
""")
code("""
public_references = [
    {'source': 'https://unsloth.ai/docs/models/gemma-4', 'model': 'Gemma 4 26B A4B',
     'metric': 'MMLU Pro', 'reported_score': 82.6, 'quantization': None, 'comparison': 'reference_only'},
    {'source': 'https://unsloth.ai/docs/models/gemma-4', 'model': 'Gemma 4 26B A4B',
     'metric': 'AIME 2026 no tools', 'reported_score': 88.3, 'quantization': None, 'comparison': 'reference_only'},
    {'source': 'https://unsloth.ai/docs/models/gemma-4', 'model': 'Gemma 4 26B A4B',
     'metric': 'LiveCodeBench v6', 'reported_score': 77.1, 'quantization': None, 'comparison': 'reference_only'},
    {'source': 'https://unsloth.ai/docs/models/gemma-4#unsloth-gguf-benchmarks',
     'metric': 'GGUF mean KLD plot', 'reported_score': None, 'quantization': 'verify exact row', 'comparison': 'not_transcribed'}]
print(json.dumps(public_references, ensure_ascii=False, indent=2))
if exp is not None:
    write_json(exp.root / 'public-references.json', public_references)
""")
code("""
selection_results = {}
if RUN_MODEL and exports:
    records = [r for r in datasets['selection'] if r['domain'] in ['math', 'code']]
    selection_results['original'] = exp.collect('selection-original', MODEL, records, mode='generate', max_tokens=GENERATION_TOKENS)
    for key, path in exports.items():
        selection_results[key] = exp.collect('selection-' + key, path, records, mode='generate', max_tokens=GENERATION_TOKENS)
    print({name: summarize(rows)['summary'] for name, rows in selection_results.items()})
    keep = APPROVED_KEEP_COUNTS[0]
    evidence_files = [exp.root / f'selection-{name}.json' for name in selection_results]
    evidence_files.append(exp.root / f'selection-reference-loss-{keep}.json')
    review_file = exp.root / f'selection-review-{keep}.json'
    write_json(review_file, {'keep': keep, 'exploratory': True,
                            'summary': {name: summarize(rows) for name, rows in selection_results.items()},
                            'evidence_sha256': {str(p.resolve()): sha256(p) for p in evidence_files}})
    print('다음 축소는 이 요약과 실제 생성 결과를 검토한 뒤 직접 결정하세요. review hash:', sha256(review_file))
    # Selection 결과를 읽고 품질 손상이 있으면 APPROVED_KEEP_COUNTS를 더 줄이지 마세요.
""")
md("""
## 6. 평가

최종 집합을 고정한 뒤 Evaluation을 실행합니다. 결과를 보고 집합을 바꾸면 그 데이터는 후속 Selection이 됩니다.
현재는 탐색 수학 숫자 정답 채점이며 MATH 기호식은 수동 검토가 필요합니다.
코드는 생성·AST 분석까지만 수행하고 **생성 코드 실행, HumanEval+/MBPP+ 통과율 산출은 하지 않습니다**.
한국어/영어 지시 이해·종료 점검은 아래 고정 입력을 별도로 사용합니다.
""")
code("""
evaluations = {}
if RUN_MODEL and RUN_EVALUATION and exports:
    evaluation_records = datasets['evaluation']
    exp.record('evaluation_frozen', {'mask_sha256': canonical_hash({k: candidates[k] for k in exports}),
                                   'records_sha256': canonical_hash(evaluation_records), 'exploratory': True})
    for name, path in {'original': MODEL, **exports}.items():
        evaluations[name] = exp.collect('evaluation-' + name, path, evaluation_records, mode='generate', max_tokens=GENERATION_TOKENS)
        write_json(exp.root / f'evaluation-{name}-analysis.json', summarize(evaluations[name]))
    sanity = [{'id': 'sanity-ko', 'domain': 'general', 'prompt': '두 문장으로 정적 Expert Pruning의 한계를 설명하고 답변을 끝내세요.', 'reference': ''},
              {'id': 'sanity-en', 'domain': 'general', 'prompt': 'Explain one limitation of static expert pruning in two sentences, then stop.', 'reference': ''}]
    for name, path in {'original': MODEL, **exports}.items():
        exp.collect('sanity-' + name, path, sanity, mode='generate', max_tokens=256)
    print({name: summarize(rows)['summary'] for name, rows in evaluations.items()})
    paired = {name: paired_quality(evaluations['original'], rows, CONFIG['seed'])
              for name, rows in evaluations.items() if name != 'original'}
    write_json(exp.root / 'paired-quality.json', paired)
    print(paired)
else:
    print('최종 평가 미실행. 코딩 정답률과 최종 품질 보존은 아직 미검증입니다.')
""")
code("""
speed_results = {}
if RUN_MODEL and RUN_BENCHMARK and exports:
    prompts = gate_records
    for name, path in {'original': MODEL, **exports}.items():
        speed_results[name] = benchmark(exp, 'speed-' + name, path, prompts, repetitions=5, max_tokens=256)
    print(speed_results)
    print('각 fresh process 안에서 워밍업 후 KV를 비우고 측정합니다. native GPU peak/배치/전송 null은 미관측입니다.')
    print('별도 Windows PDH 프로세스 dedicated GPU memory 표본은 adapter 합산·표본 최대이며 HIP allocation peak/상주 증거가 아닙니다.')
""")
code("""
from IPython.display import HTML, display
points = []
for name, rows in evaluations.items():
    stats = summarize(rows)['summary'].get('math', {})
    path = MODEL if name == 'original' else exports[name]
    points.append({'name': name, 'file_bytes': path.stat().st_size, 'accuracy': stats.get('numeric_math_accuracy')})
display(HTML(render_tradeoff(points)))
if exp is not None:
    write_json(exp.root / 'tradeoff-points.json', points)
""")
md("""
## 7. 결론

실측 전 결론을 채우지 않습니다. 현재 노트북은 탐색 실험이므로 고정 품질 허용치를 만족했다는 성공 판정은 하지 않습니다.
유용한 부분집합 발견과 특정 Expert의 독점적 도메인 전문성 발견은 다른 주장입니다.
파일 축소, VRAM 축소, Swapping 제거와 속도 개선도 각각 따로 근거가 필요합니다.
""")
code("""
conclusion = {
    'H1': {'status': '근거 부족', 'review': 'domain-distributions/selection-stability를 검토하고 표본 한계를 기록'},
    'H2': {'status': '근거 부족', 'review': '수학은 탐색 채점, 코딩 실행 평가 미실행이므로 전체 능력 보존 결론 보류'},
    'H3': {'status': '근거 부족', 'review': 'physical 동등성·파일 크기·실측 VRAM을 각각 확인'},
    'H4': {'status': '근거 부족', 'review': '동일 조건 반복 속도와 실제 배치 확인; 미관측 전송을 0으로 해석하지 않음'},
    'scope': '실행한 단계만 해석, Router 제거 미실험',
    'measurements_present': bool(evaluations),
    'evidence': ({name: str(exp.root / name) for name in [
        'manifest.json', 'domain-distributions.json', 'selection-stability.json',
        'exports.json', 'tradeoff-points.json']
        if (exp.root / name).is_file()} if exp is not None else {}),
    'observed_file_sizes': ({name: path.stat().st_size for name, path in {'original': MODEL, **exports}.items()}
                            if exports else {}),
    'observed_speed_metrics': speed_results,
}
print(json.dumps(conclusion, ensure_ascii=False, indent=2))
if exp is not None:
    write_json(exp.root / 'conclusion-review.json', conclusion)
print('결론을 수정할 때에는 지지/반증/근거 부족과 해당 실측 artifact 경로를 함께 남기세요.')
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {'display_name': 'wemakemodel (Python 3.12)', 'language': 'python', 'name': 'python3'}
notebook.metadata.language_info = {'name': 'python', 'version': '3.12'}
nbf.validate(notebook)
target = ROOT / 'ipynb/01_Gemma4_Expert_Pruning_연구.ipynb'
nbf.write(notebook, target)
print(target)
