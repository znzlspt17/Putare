# wemakemodel

**Gemma 4의 일부 Expert를 제거한 경량 모델을 만들고, 원본과 품질·실행 속도를 비교하는 노트북 기반 실험 도구입니다.** 수학·코딩 입력에서 어떤 Expert가 사용되고 출력에 얼마나 기여하는지 분석하여, 유지할 Expert 집합을 선택합니다.

MoE(Mixture of Experts) 모델은 입력 토큰마다 Router가 여러 Expert 중 일부를 골라 계산합니다. 이 프로젝트에서 사용하는 **정적 Expert Pruning**은 사용할 Expert를 미리 정하고 나머지 가중치를 모델 파일에서 제거하는 방식입니다. 남겨 둔 Expert 사이의 선택은 기존 Router가 계속 수행합니다.

대상 모델은 `gemma-4-26B-A4B-it-UD-IQ4_NL.gguf`입니다. Windows의 AMD GPU 환경에서 Python 3.12, Jupyter, llama.cpp 기반 실험 엔진을 사용합니다.

## 무엇을 할 수 있나요?

- **Expert 사용 패턴 분석**: 일반 텍스트·수학·코딩 입력의 Expert 선택과 출력 기여도를 수집합니다.
- **축소 모델 생성**: 유지할 Expert를 선택하고, 양자화된 가중치 블록을 보존하면서 별도 GGUF 파일을 만듭니다. 원본 파일은 덮어쓰지 않습니다.
- **원본과 비교**: Expert 사용을 제한한 원본과 실제 축소 파일의 출력이 대응하는지 확인하고, 참조 답변의 손실(NLL), 생성 결과, 파일 크기와 실행 속도를 비교합니다.
- **실험 기록 보존**: `runs/`에 데이터·모델·엔진 해시, 실행 설정과 생성 결과를 남겨 비교 조건을 추적합니다.

직접 모델을 실행하고 축소 전후의 차이를 확인하려는 사용자를 위한 도구입니다. 더 작은 파일이 만들어져도 품질 유지, VRAM 감소, 속도 향상이 보장되지는 않으며 각각 실측으로 확인해야 합니다.

## 시작하기

주 실행 파일은 [Gemma 4 Expert Pruning 연구 노트북](ipynb/01_Gemma4_Expert_Pruning_연구.ipynb)입니다. 모델 준비에는 [모델 준비 노트북](ipynb/00_모델준비.ipynb)을 참고하세요.

### 준비할 것

- Python 3.12와 Jupyter 환경
- `models/gemma-4-26B-A4B-it-UD-IQ4_NL.gguf` 모델 파일
- `requirements-research.txt`의 실험용 Python 의존성
- CMake·Ninja·C++ 빌드 도구·AMD HIP SDK를 갖춘 [native 엔진 빌드 환경](tools/native/README.md)
- 원본과 별도로 생성할 GGUF 파일을 저장할 디스크 여유 공간

PyTorch ROCm 패키지 설치와 HIP 컴파일러 준비는 별도입니다. 실제 GPU에 맞는 GPU target과 GPU offload 설정을 실행 전에 확인하세요. 노트북의 기본 설정은 컨텍스트 8K, 동시 요청 1개, 실험 프로세스 트리의 RAM 예산 40GB입니다. 이 예산은 시스템 전체 메모리 사용량을 보장하는 값이 아닙니다.

### 실행 흐름

1. 저장소 루트 또는 `ipynb/`를 작업 경로로 사용하여 연구 노트북을 엽니다.
2. 모델 경로와 GPU 설정을 확인하고, 필요한 설치·엔진 빌드·데이터 다운로드 단계를 활성화합니다.
3. `RUN_MODEL=True`로 원본 분석, 계측 전후 출력 비교, Expert 사용 통계 수집과 제거 민감도 분석을 실행합니다.
4. `WRITE_MODELS=True`로 Expert를 제거하지 않은 재저장 파일과 첫 축소 모델(레이어별 Expert 120개 유지)을 검증합니다.
5. 생성 결과와 참조 답변 손실을 검토한 뒤 다음 유지 개수를 결정합니다. 한 번에 하나의 유지 개수를 지정하며, 추가 축소에는 이전 단계 실측 요약의 검토 hash가 필요합니다.
6. 유지할 집합을 고정한 뒤 `RUN_EVALUATION`과 `RUN_BENCHMARK`를 활성화해 품질과 속도를 비교합니다.

**기본 설정에서 Run All은 모델 실행·네트워크 접근·빌드·축소 파일 생성을 수행하지 않습니다.** 각 단계는 사용자가 플래그를 켜서 실행합니다.

## 결과를 읽을 때

현재 제공하는 것은 실험 도구와 실행 절차입니다. 이 README는 실제 모델의 경량화 효과나 Windows HIP 빌드·추론 성공을 검증한 결과를 제시하지 않습니다.

| 결과 | 해석 범위 |
|---|---|
| 수학 자동 점수 | 마지막 숫자 추출을 이용한 탐색용 지표입니다. 기호식은 수동 검토가 필요하며 MATH-500 정확도로 해석할 수 없습니다. |
| 코딩 평가 | 생성과 구문 분석까지 수행합니다. 코드 실행 정답률(HumanEval+/MBPP+)은 측정하지 않습니다. |
| 모델 파일 크기 | 저장 공간의 변화입니다. VRAM 감소나 Expert 전송 제거를 의미하지 않습니다. |
| GPU 메모리 | native 엔진의 할당 peak·실제 CPU/GPU 배치·Expert 전송은 미관측입니다. Windows PDH 표본은 별도의 OS 프로세스 GPU 메모리 지표이며, 미지원·실패는 `null`과 사유로 기록합니다. |
| 실행 속도 | 계측을 끄고 프로세스 내부 워밍업 후 5회 반복합니다. 같은 엔진·입력·템플릿·Top-K·컨텍스트·offload 설정끼리 비교해야 합니다. |

노트북의 Unsloth 공개 모델 점수는 참고 자료이며, 대상 GGUF 파일을 로컬에서 측정한 값과 구분합니다.

## 구성과 검증

| 경로 | 내용 |
|---|---|
| `ipynb/` | 모델 준비와 Expert Pruning 실험 노트북 |
| `moe_prune/` | 데이터 처리, Expert 분석과 GGUF 축소 모듈 |
| `tools/native/` | llama.cpp 기반 실험 엔진과 빌드 안내 |
| `tests/` | 합성 GGUF·가짜 Routing 통계를 사용하는 검증 |
| `runs/` | 실행 시 생성되는 실험 기록과 결과 |

실제 모델을 로딩하지 않는 보조 도구 테스트는 다음과 같이 실행합니다.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tools/native -p "test_*.py" -v
```
