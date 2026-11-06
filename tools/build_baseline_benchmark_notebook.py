"""2026-11-06: 기본 실행에 부작용이 없는 02번 벤치마크 노트북 생성."""
from pathlib import Path
import textwrap
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


md(r"""
# 02. Gemma 4 원본 수학·코딩 벤치마크와 Expert 계측

AIME 2026 30문항과 LiveCodeBench v6 신규 175문항입니다. Windows HIP에서 원본 GGUF를
실행하고, LiveCodeBench는 WSL Ubuntu의 격리된 공식 채점기를 사용합니다.
Vision·MTP 없이 추론하며 모든 토큰의 Expert 선택을 계측합니다.

**기본 Run All은 설치·다운로드·빌드·모델 추론·채점을 실행하지 않습니다.** 각 단계는 아래
플래그를 직접 켭니다. 샘플 실행과 전체 평가는 별도 RUN_ID를 사용합니다.
""")
code(r"""
from pathlib import Path
import json, sys, subprocess
ROOT = Path.cwd().resolve()
if ROOT.name == 'ipynb':
    ROOT = ROOT.parent
assert (ROOT / 'moe_prune').is_dir(), '저장소 루트 또는 ipynb에서 실행하세요'
sys.path.insert(0, str(ROOT))

PREPARE_WSL = False       # WSL 전용 채점 환경 및 공식 소스 준비
PREPARE_ENGINE = False    # native 빌드 명령 확인
BUILD_ENGINE = False      # PREPARE_ENGINE=True일 때 빌드 실행
DOWNLOAD_DATA = False     # 공개 데이터와 공식 prompt를 고정 snapshot으로 저장
VERIFY_DATA_IDS = False   # v5 원본 ID를 스트리밍 검증 (약 4.35 GB 네트워크 사용)
RUN_SMOKE = False         # 수학 1문항·코딩 1문항, 출력 상한 128토큰
RUN_BENCHMARK = False     # 전체 205문항 생성과 필수 계측
RUN_GRADING = False       # 전체 답안에 대한 AIME 및 WSL 공식 채점

MODEL = ROOT / 'models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf'
ENGINE = ROOT / '.cache/llama-build/bin/llama-expert-research.exe'
DATA = ROOT / 'runs/datasets/aime2026-lcbv6-new-001'
DATA_ID_PROOF = DATA.parent / (DATA.name + '-id-validation.json')
RUN_ID = 'gemma4-aime2026-lcbv6-001'
RUN_DIR = ROOT / 'runs' / RUN_ID
SMOKE_DIR = ROOT / 'runs' / (RUN_ID + '-smoke')
GPU_TARGET = 'gfx1201'
print({'model_present': MODEL.is_file(), 'engine_present': ENGINE.is_file(),
       'snapshot_present': (DATA / 'manifest.json').is_file(), 'run_id': RUN_ID})
""")
md(r"""
## 1. 준비

실행 위치: Windows 저장소. 채점 도구만 WSL `Ubuntu-24.04`의
`/home/vivy/.venvs/wemakemodel-lcb`와 고정 commit 소스를 사용합니다.
WSL 격리 환경은 네트워크와 사용자 파일 접근을 제한합니다.
준비 셀은 패키지·공식 채점기 소스를 내려받습니다. native 빌드는 로컬 CMake·HIP SDK를 사용합니다.
""")
code(r"""
from moe_prune.benchmark_data import prepare_wsl, prepare_snapshot, load_snapshot
if PREPARE_WSL:
    prepare_wsl(ROOT)
if PREPARE_ENGINE:
    command = [sys.executable, str(ROOT / 'tools/native/prepare.py'), '--gpu-target', GPU_TARGET]
    if BUILD_ENGINE:
        command.append('--execute')
    subprocess.run(command, cwd=ROOT, check=True)
if not PREPARE_WSL and not PREPARE_ENGINE:
    print('WSL 준비와 native 빌드는 실행하지 않았습니다.')
""")
md(r"""
## 2. 데이터 snapshot

AIME 2026 30문항과 동일 데이터 revision의 LiveCodeBench v6−v5 신규 ID 175개를 고정합니다.
문제와 정답/테스트는 로컬 snapshot에 보존하지만 모델 입력에는 정답과 테스트를 넣지 않습니다.
공식 채점기 및 WSL 패키지 버전도 함께 고정합니다. 기존 snapshot을 덮어쓰지 않습니다.
`VERIFY_DATA_IDS=True`이면 고정 revision의 v5 원본 다섯 파일을 다운로드 저장 없이
스트리밍하여 Explorer ID와 비교합니다. 약 4.35GB 네트워크 사용 후 별도 검증 증거를 남깁니다.
""")
code(r"""
if DOWNLOAD_DATA:
    manifest = prepare_snapshot(DATA, ROOT)
    print('데이터 준비:', manifest['counts'])
elif (DATA / 'manifest.json').is_file():
    records, manifest = load_snapshot(DATA)
    print('기존 snapshot 검증:', manifest['counts'])
else:
    print('데이터 없음: WSL 준비 후 DOWNLOAD_DATA를 활성화하세요.')
if VERIFY_DATA_IDS:
    subprocess.run([sys.executable, str(ROOT / 'tools/verify_benchmark_ids.py'),
                    '--snapshot', str(DATA), '--output', str(DATA_ID_PROOF)], check=True)
""")
md(r"""
## 3. 생성과 필수 계측

고정값: thinking 활성화, temperature 1.0 / top_p 0.95 / top_k 64, seed 42,
컨텍스트 65,535, 최대 출력 32,768, K/V Q8_0, Flash Attention, 슬롯 1개.
메모리 부족 시 CPU offload를 시도합니다. 계측을 끄거나 MTP를 사용하지 않습니다.
시작 전 GPU와 RAM 여유를 확인하고, 완료된 문제는 같은 조건에서 재사용합니다.
""")
code(r"""
from moe_prune.base_benchmark import BenchmarkRun, DEFAULT_CONFIG
CONFIG = dict(DEFAULT_CONFIG)
if RUN_SMOKE:
    smoke_config = {**CONFIG, 'max_tokens': 128}
    print(BenchmarkRun(smoke_config, MODEL, ENGINE, DATA, SMOKE_DIR).run(smoke=True))
if RUN_BENCHMARK:
    print(BenchmarkRun(CONFIG, MODEL, ENGINE, DATA, RUN_DIR).run())
if not RUN_SMOKE and not RUN_BENCHMARK:
    print('모델 추론·계측을 실행하지 않았습니다.')
""")
md(r"""
## 4. 채점

생성 완료 205문항만 전체 점수를 계산합니다. 수학 정답은 최종 답변의 boxed 정수를 비교합니다.
코딩 채점은 공식 추출·공개/비공개 테스트·pass@1을 사용합니다. 실행 환경 오류는 오답이 아닙니다.
""")
code(r"""
from moe_prune.benchmark_data import grade_snapshot
if RUN_GRADING:
    print(grade_snapshot(DATA, RUN_DIR / 'generations.jsonl', RUN_DIR, ROOT))
else:
    print('채점을 실행하지 않았습니다.')
""")
md(r"""
## 5. 결과 확인

`routing/`은 원시 FP32 Router 가중치와 FP64 누적 출력 norm을 문제별 압축 보존합니다.
`routing-profile.json`은 토큰 기준과 문제별 평균을 보유합니다. 히트맵의 셀에 마우스를 올리면
선택 비율을 확인할 수 있습니다. 활성화 빈도만으로 Expert 제거 가능성을 판단하지 않습니다.
""")
code(r"""
PROFILE = RUN_DIR / 'routing-profile.json'
SCORE = RUN_DIR / 'grades-summary.json'
if SCORE.is_file():
    print('완료 점수:', json.loads(SCORE.read_text(encoding='utf-8')))
if PROFILE.is_file():
    from IPython.display import display, SVG
    profiles = json.loads(PROFILE.read_text(encoding='utf-8'))
    def show_experts(domain, phase, layer):
        row = profiles[domain][phase][str(layer)]
        for expert in sorted(range(128), key=lambda i: -row['counts'][i]):
            count, rate = row['counts'][expert], row['token_rate'][expert] * 100
            weight, norm = row['mean_gate_per_selection'][expert], row['mean_norm_per_selection'][expert]
            print(f'{expert:3d} count={count} selected={rate:.4f}% '
                  f'weight={weight:.6f} norm={norm:.6g}' if count else
                  f'{expert:3d} count=0 selected={rate:.4f}% weight=미관측 norm=미관측')
    for graph in sorted((RUN_DIR / 'heatmaps').glob('*.svg')):
        display(SVG(filename=str(graph)))
    print("상세: show_experts('math', 'prefill', 0)")
if not PROFILE.is_file() and not SCORE.is_file():
    print('이 RUN_ID의 완료 결과가 없습니다.')
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'}})
nbf.validate(notebook)
nbf.write(notebook, ROOT / 'ipynb/02_base_model_benchmark.ipynb')
