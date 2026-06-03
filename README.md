# SNN 기반 Light-Intensity → Frequency 변환 소자 학습

광응답 소자를 **뉴런/시냅스 하드웨어**로 보고, 소자의 물리적 "광세기 → 스파이크 주파수"
변환 특성을 뉴런 모델에 직접 반영하여 snnTorch로 시뮬레이션·학습하는 프로젝트.

- 프레임워크: **snnTorch** (PyTorch 기반, surrogate gradient 학습)
- 데이터: **합성 데이터**로 시작
- 개발 방식: **단계별(step-by-step)**

## 핵심 아이디어

실제 광응답 소자는 입사 광세기 Φ가 커질수록 광전류 I_ph가 증가하고,
이 전류가 막전위(membrane)를 충전시켜 스파이크 발화율(frequency)을 높인다.
즉 **소자 자체가 "광세기 → 주파수" 인코더 역할을 하는 LIF 뉴런**이다.

```
   빛(Φ) ──▶ [광응답 I_ph(Φ)] ──▶ [LIF 막전위 적분] ──▶ 스파이크열 (frequency ∝ Φ)
            (소자 물리 특성)        (뉴런 동역학)
```

## 로드맵

| 단계 | 파일 | 내용 | 상태 |
|------|------|------|------|
| Step 1 | `01_photo_neuron.py` | 광-뉴런(Photo-LIF) 소자 모델 + light→frequency 변환 곡선 (수식 기반, 참고용) | ✅ |
| Step 2 | `02_measured_encoder.py` | **실측 보정 곡선** 기반 grayscale→frequency 인코더 (Poisson rate coding) | ✅ |
| Step 3 | `03_train.py` | MNIST + 측정 인코더 → SNN 분류기 학습/평가 (surrogate gradient) | ✅ |

> 핵심 변경: 광세기→주파수 변환을 **수식**이 아니라 **소자 실측 데이터**로 수행 (`02_measured_encoder.py`).
> MNIST 흑백 명암 → 실측 곡선으로 주파수 변환 → Poisson 스파이크열 → SNN 학습.

## 실행

전역 conda의 torch 2.12.0이 `c10.dll` 초기화 실패로 깨져 있어, **전용 venv**를 만들어 사용한다.
venv 위치: `C:\Claude\snn_light2freq\.venv` (torch 2.5.1+cpu, snntorch 0.9.4, torchvision 0.20.1+cpu, pandas, matplotlib)

```powershell
# venv의 python으로 직접 실행
.\.venv\Scripts\python.exe 01_photo_neuron.py      # (참고) 수식 기반 소자 변환 특성 → fig1, fig2
.\.venv\Scripts\python.exe 02_measured_encoder.py  # 실측 곡선 기반 인코딩 시각화 → fig3~5
.\.venv\Scripts\python.exe 03_train.py             # MNIST SNN 학습 → snn_mnist.pt, fig6
```

## 검증 결과 (2026-06-03)

전 과정 실행 검증 완료. Step 3 (서브셋 train 6000 / test 1000, 3 epoch, T=50):

| epoch | loss | train acc | test acc |
|------:|-----:|----------:|---------:|
| 1 | 1.31 | 76.8% | 70.4% |
| 2 | 0.57 | 89.5% | 84.3% |
| 3 | 0.29 | 93.7% | **88.7%** |

정확도를 더 올리려면: `03_train.py`의 `CFG`에서 `n_train=None`(전체 6만장), `epochs↑`,
`num_steps↑`(rate 추정 정밀), `hidden↑` 조정.

## 측정 보정 파일

`C:\Claude\AI Cap\data\characteristic_curve.csv`
( 컬럼: `intensity_mW_cm2`, `frequency_Hz` / 0~9.43 mW/cm², 0~50 kHz, 포화형 )
- 운용 범위는 `02_measured_encoder.py`의 `CONFIG["intensity_white"]`(기본 2.0 mW/cm²)로 조정.

## 환경 메모

- 현재 전역 conda(torch 2.12.0)의 `c10.dll` 초기화 실패로 torch import 불가 → 복구 필요.
- 권장: 전용 venv + 검증된 CPU torch(예: 2.5.x) + snntorch 재설치.
