# Gemma4 연구용 native runner

이 도구는 upstream llama.cpp `10d4846a572aebf21c0482d140b00e74c0f1ad20`의 공개 C API, Jinja chat renderer와 scheduler evaluation callback에 연결한 **별도 실행 파일**이다. 원본 소스에 패치를 적용하지 않고 CMake의 `add_subdirectory`로 연결한다. 공식 `llama-cli`의 기존 옵션으로 관측·절삭 기능을 가장하지 않는다.

## 준비와 빌드

Windows Native Tools 환경에서 CMake 3.24+, Ninja, AMD HIP SDK의 clang/clang++, hip/hipblas/rocblas가 필요하다. PyTorch ROCm wheel만으로 compiler 환경을 충족한다고 가정하지 않는다. 아래 명령은 모델을 실행하지 않으며, `--execute`를 지정하기 전에는 진단과 명령만 출력한다.

```powershell
.\.venv\Scripts\python.exe tools/native/prepare.py
# gfx 값은 hipInfo 등으로 실제 GPU에서 확인한다. 아래 placeholder를 그대로 실행하지 않는다.
.\.venv\Scripts\python.exe tools/native/prepare.py --gpu-target <확인한-gfx값>
.\.venv\Scripts\python.exe tools/native/prepare.py --gpu-target <확인한-gfx값> --execute
```

빌드 결과는 `.cache/llama-build/bin/llama-expert-research.exe`다. HIP DLL과 llama/ggml DLL의 검색 경로가 필요하다. native runner의 Windows HIP 빌드·실제 Gemma4 추론은 사용자 환경에서 검증해야 하며, 준비 시 성공한 것으로 기록하지 않는다. 기존 checkout의 revision이나 dirty 상태가 다르면 도구가 중단한다.

## 요청과 출력

```text
llama-expert-research.exe --request request.json --output result.jsonl
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
| `seed` | 재현 manifest용, 기본42. 현재 sampling은 온도0 greedy로 seed와 무관 |
| `max_routing_records` | 기본1,000,000. 초과하면 부분 결과를 성공으로 반환하지 않음 |

`routing` JSONL event는 layer, phase, 절대 token_index, expert_ids, 정규화된 gate weights, 실제 weighted expert output의 L2 norm을 기록한다. `prefill`은 prompt, `decode`는 생성 토큰 또는 강제 참조 답변 문맥이다. weighted output에는 원본 graph의 expert scale과 gate 결합이 포함된다. `weights`는 expert scale 적용 이전의 정규화된 Router gate이므로 두 값을 혼동하지 않는다.

최종 `result`는 text/token_ids/stop_reason, 타이밍을 기록한다. `logits` 모드는 **prompt 직후 전체 vocabulary logits**를 출력한다. `reference_loss`는 prompt loss를 제외한 참조 토큰 평균 NLL과 토큰 수를 출력한다. 첫 버전은 greedy 고정이며 Unsloth 공개 sampling 조건 재현을 주장하지 않는다.

GPU peak memory, 실제 CPU/GPU placement, Expert transfer는 이 runner가 관측하지 않으므로 null이다. `n_gpu_layers`를 근거로 전송0이나 전체 GPU 상주를 판정하지 않는다. stdout/stderr backend 로그는 별도 보존한다.

## 동등성 게이트

1. 동일 prompt의 `logits` 요청에서 instrumentation OFF/ON과 전체 allowlist를 수치 비교한다.
2. no-op GGUF rewrite와 원본을 비교한다.
3. 동일 원래 Expert ID 집합을 사용한 masked 원본과 재매핑된 physical GGUF를 비교한다. physical 파일 요청에는 원래 allowlist를 다시 적용하지 않는다.
4. 이 게이트를 통과한 경우에만 축소 비율을 늘린다. 고정된 gate clamp와 float 합산 순서의 수치 차이도 실제 측정으로 판단한다.

callback은 `ask=true`에서 필요한 노드만 요청하고 `ask=false`에서 계산된 Router logits의 제외 ID만 `-inf`로 바꾼다. Softmax/Top-K/정규화/Expert scale/Shared FFN은 upstream graph가 실행한다. 전체 허용 Mask는 값을 변경하지 않는다. callback 자체의 동기화·전송 비용은 속도 결과에 섞이지 않도록 관측 OFF 비교를 별도로 사용한다.

계층별 유지 수가 같아야 한다는 제약은 물리 GGUF export에만 적용한다. 런타임 Mask에서는 한 계층만128→127로 줄이는 제거 민감도 실험을 허용하며 나머지 계층은 그대로 유지한다. Mask를 사용하지 않는 요청은 `allowlist: null` 또는 해당 필드 생략으로 표현한다. 빈 객체는 누락 설정을 숨기지 않도록 거부한다.

## 외부 Windows GPU memory 표본

`gpu_memory.ProcessGpuMemorySampler(pid)`는 PDH의 language-neutral `GPU Process Memory / Dedicated Usage` counter를 localized path로 변환·확장하여 해당 PID의 값만 수집한다. `sample()`을 외부 monitor loop에서 호출하고 `snapshot()`을 결과에 저장한다. adapter 합산 OS 프로세스 dedicated memory의 **표본 최대값**이며 HIP allocation peak나 Expert-only memory·실제 GPU 상주 증거가 아니다. 실패·counter 없음은 `None`과 사유로 남긴다. Windows counter 자체의 알려진 오차와 표본 사이 peak 누락 가능성을 문서화하므로 native runner의 미관측 `gpu_memory_bytes`를 이 값으로 몰래 대체하지 않는다.
