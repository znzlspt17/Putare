# Gemma4 연구용 native runner

이 도구는 upstream llama.cpp `10d4846a572aebf21c0482d140b00e74c0f1ad20`의 공개 C API, Jinja chat renderer와 scheduler evaluation callback에 연결한 **별도 실행 파일**이다. 원본 소스에 패치를 적용하지 않고 CMake의 `add_subdirectory`로 연결한다. 공식 `llama-cli`의 기존 옵션으로 관측·절삭 기능을 가장하지 않는다.

## Colab CUDA 준비와 빌드

Colab의 NVIDIA A100, Python 3.12, CMake 3.24+, Ninja, C++ 컴파일러와 CUDA toolkit이 필요하다.
노트북에서 PREPARE_ENGINE=True로 명령을 확인하고, 추가로 BUILD_ENGINE=True를 설정하면
고정된 공개 llama.cpp revision을 준비해 CUDA backend로 빌드한다. 빌드는 모델을 실행하지 않는다.
기존 checkout의 revision이나 작업 트리가 다르면 중단하며, 빌드 결과는
.cache/llama-build/bin/llama-expert-research이다. 환경 검사는 다음 노트북 셀로도 실행할 수 있다.

~~~python
import subprocess, sys
subprocess.run([sys.executable, 'tools/native/prepare.py', '--backend', 'cuda',
                '--gpu-target', '80'], check=True)
~~~

## 계측과 GPU fusion

연구 엔진은 동일한 계측 조건을 위해 GPU fusion을 비활성화하고
gpu_fusion_disabled를 기록한다. 과거 환경과 다른 엔진·장비에서 얻은 결과는
기존 결과와 구분하여 새 RUN_ID로 기록한다.

## 원본 모델의 로컬 기준 측정

노트북에서 `RUN_MODEL=True`, `RUN_EVALUATION=True`, `RUN_BENCHMARK=True`,
`RUN_PRUNING_ANALYSIS=False`, `WRITE_MODELS=False`로 실행하면 원본 GGUF만으로
품질 관찰과 속도 측정을 수행한다. 설치·빌드·데이터 다운로드 플래그는 환경 준비 상태에 맞게 설정한다.
`WRITE_MODELS=False`여도 원본 평가와 벤치마크를 생략하지 않는다.
Pruning 분석을 켜면 기존 routing·mask·민감도 분석을 추가하며, 물리 모델 저장에는
`WRITE_MODELS=True`가 별도로 필요하다.

원본 측정 결과는 `runs/<RUN_ID>/evaluation-original.json`,
`evaluation-original-analysis.json`, `speed-original-benchmark.json`에 저장한다.
`baseline-summary.json`에는 GPU·드라이버·RAM의 조회 결과, 모델/데이터 해시,
실행 설정, 생성 조건, 품질·속도 요약과 원시 근거 경로를 함께 저장한다.
GPU 목록은 시스템 장치 조회이며 실제 레이어 배치의 관측 증거는 아니다.
원시 요청·응답·로그와 모델/엔진 해시는 같은 run의 `jobs/`에 보존한다.
속도는 fresh process 안에서 워밍업하고 KV를 초기화한 뒤 측정한 load 시간,
TTFT, prefill 및 decode tokens/s를 반복 표본의 median/min/max로 보고한다.
로드 시간과 TTFT의 측정 범위는 다르므로 서로 합쳐 해석하지 않는다.

현재 품질 관찰은 탐색 수학 숫자 정답·코드 문법·빈 출력·토큰 제한 종료를 포함한다.
코드 테스트 실행에 의한 pass@1이나 MATH 기호식 정답률은 산출하지 않는다.
`EVALUATION_N=16`은 각 데이터셋의 표본 제한이며 전체 평가 데이터 수 16건을 뜻하지 않는다.
공개 Unsloth 점수와 로컬 표본 결과는 모델·양자화·데이터·평가 방법·생성 조건이
같다는 근거 없이 직접 비교하지 않는다. 속도는 로컬 하드웨어와 offload 설정에 의존하며,
품질은 GPU 모델 차이만으로 설명하지 않는다.

## 메모리 사용량 경고

실제 Colab 시스템 RAM의 80%, 선택된 GPU 전체 VRAM의 97%를 초과하면 경고한다.
기준을 초과해도 작업은 계속되며, 100ms 간격 표본은 메모리 할당을 강제로 제한하지 않는다.
모델과 KV cache의 사전 필요량은 실제 모델과 설정을 바탕으로 추정한다.
순간적인 할당 실패나 런타임 종료는 방지할 수 없다. 측정 불가능한 상태는 오류로 구분한다.
전체 GPU 적재를 요청하고 실제 배치는 엔진 로그로 확인한다.

## 요청과 출력

```text
llama-expert-research --request request.json --output result.jsonl
```

요청 필드:

| 필드 | 의미 |
|---|---|
| `model`, `prompt` | GGUF 경로, 사용자 입력 |
| `mode` | `generate`, `profile`, `logits`, `reference_loss` |
| `max_tokens`, `context_size` | 기본 256, 8192. 초과 시 자르지 않고 실패 |
| `n_gpu_layers` | 기본 0. 실제 GPU offload 수는 실험 설정에서 명시 |
| `instrumentation` | 기본 `profile`에서만 true. false일 때 관측 전송 생략 |
| `warmup` | 기본 false. true이면 계측 없는 generate에서 동일 프로세스의 prompt·1 decode를 준비한 후 KV를 지우고 측정. 워밍업 시간은 load/prefill/decode 지표에서 제외 |
| `allowlist` | `{ "0": [0,1,...], ... }`; 모든 레이어를 포함하고 각각 Top-K 이상. 단일 계층 ablation을 위해 계층별 유지 수가 달라도 허용 |
| `reference` | `reference_loss`에서 필수인 참조 답변 |
| `apply_chat_template` | 기본 true. GGUF의 template를 Jinja로 적용 |
| `enable_thinking` | 기본 false. 모든 비교에서 동일하게 고정 |
| `seed` | 기본 42. 명시적 sampling에서 난수 seed로 사용하며 기본 greedy에서는 결과에 영향을 주지 않음 |
| `sampling` | 생략하면 기존 greedy. 명시하면 `temperature`, `top_p`, `top_k`를 적용 |
| `system_prompt` | 선택적 system 메시지. 사용자 prompt 및 thinking 설정과 함께 chat template에 전달 |
| `cache_type_k`, `cache_type_v` | `f16` 또는 `q8_0`. 생략하면 기존 `f16` |
| `flash_attention` | 명시 시 Flash Attention을 켜거나 끔. Q8 V cache에는 활성화가 필요 |
| `offload_kqv` | KV cache와 KQV 연산의 GPU 배치 여부. 생략하면 llama.cpp 기본값 |
| `all_token_logits` | true면 batch 내 모든 토큰의 최종 레이어를 계산해 전체 routing 계측. 생략하면 기존 마지막 토큰만 출력하는 경로 |
| `max_routing_records` | 기본1,000,000. 초과하면 부분 결과를 성공으로 반환하지 않음 |

`routing` JSONL event는 layer, phase, 절대 token_index, expert_ids, 정규화된 gate weights, 실제 weighted expert output의 L2 norm을 기록한다. `prefill`은 prompt, `decode`는 생성 토큰 또는 강제 참조 답변 문맥이다. weighted output에는 원본 graph의 expert scale과 gate 결합이 포함된다. `weights`는 expert scale 적용 이전의 정규화된 Router gate이므로 두 값을 혼동하지 않는다.

최종 `result`는 text/token_ids/stop_reason, 샘플링 요청, cache 형식, 실제 context 크기와 처리된 decode 토큰 수, 타이밍을 기록한다. Q8 cache와 Flash Attention의 실제 적용 여부는 native stderr의 KV 형식과 Flash Attention 로그로 검증한다. llama.cpp가 context 크기를 정렬할 수 있으므로 요청 크기와 실제 크기를 구분한다. `logits` 모드는 **prompt 직후 전체 vocabulary logits**를 출력한다. `reference_loss`는 prompt loss를 제외한 참조 토큰 평균 NLL과 토큰 수를 출력한다. 명시적인 sampling 설정 없이 실행하면 기존 greedy 동작을 유지한다.

GPU peak memory, 실제 CPU/GPU placement, Expert transfer는 이 runner가 관측하지 않으므로 null이다. `n_gpu_layers`를 근거로 전송0이나 전체 GPU 상주를 판정하지 않는다. stdout/stderr backend 로그는 별도 보존한다.

## 동등성 게이트

1. 동일 prompt의 `logits` 요청에서 instrumentation OFF/ON과 전체 allowlist를 수치 비교한다.
2. no-op GGUF rewrite와 원본을 비교한다.
3. 동일 원래 Expert ID 집합을 사용한 masked 원본과 재매핑된 physical GGUF를 비교한다. physical 파일 요청에는 원래 allowlist를 다시 적용하지 않는다.
4. 이 게이트를 통과한 경우에만 축소 비율을 늘린다. 고정된 gate clamp와 float 합산 순서의 수치 차이도 실제 측정으로 판단한다.

callback은 `ask=true`에서 필요한 노드만 요청하고 `ask=false`에서 계산된 Router logits의 제외 ID만 `-inf`로 바꾼다. Softmax/Top-K/정규화/Expert scale/Shared FFN은 upstream graph가 실행한다. 전체 허용 Mask는 값을 변경하지 않는다. callback 자체의 동기화·전송 비용은 속도 결과에 섞이지 않도록 관측 OFF 비교를 별도로 사용한다.

계층별 유지 수가 같아야 한다는 제약은 물리 GGUF export에만 적용한다. 런타임 Mask에서는 한 계층만128→127로 줄이는 제거 민감도 실험을 허용하며 나머지 계층은 그대로 유지한다. Mask를 사용하지 않는 요청은 `allowlist: null` 또는 해당 필드 생략으로 표현한다. 빈 객체는 누락 설정을 숨기지 않도록 거부한다.

## GPU 메모리 표본

NVML이 측정하는 GPU 전체 사용량과 프로세스별 사용량을 구분한다.
gpu_memory.ProcessGpuMemorySampler(pid)는 프로세스 메모리를 표본으로 기록하며,
미지원·실패는 None과 사유를 남긴다. 표본 최대는 순간적인 실제 최대치를 보장하지 않는다.
