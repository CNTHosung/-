# SNN 기반 Light-Intensity → Frequency 변환 소자 학습 보고서

**작성일**: 2026-06-03
**프로젝트 경로**: `C:\Claude\snn_light2freq`
**프레임워크**: snnTorch 0.9.4 (PyTorch 2.5.1+cpu)

---

## 1. 개요 및 목적

본 프로젝트는 **광응답 소자(photoresponsive device)** 를 스파이킹 뉴런 하드웨어로
간주하고, 소자의 물리적 *"광세기(light intensity) → 스파이크 주파수(frequency)"*
변환 특성을 **실측 데이터**로부터 인코더에 직접 반영하여, MNIST 손글씨 숫자를
**스파이킹 신경망(Spiking Neural Network, SNN)** 으로 분류하는 것을 목표로 한다.

핵심 아이디어는 다음과 같다.

> 실제 광응답 소자는 입사 광세기 Φ가 커질수록 광전류 I_ph가 증가하고, 이 전류가
> 막전위(membrane potential)를 충전시켜 발화율(firing rate)을 높인다. 즉 **소자 자체가
> "광세기 → 주파수" 인코더 역할을 하는 뉴런**이다. MNIST 픽셀의 흑백 명암을 광세기로
> 보고, 소자의 실측 변환 곡선을 통해 주파수로 바꾼 뒤 스파이크열로 인코딩한다.

```
 MNIST 명암 ─[실측 소자 곡선]→ 주파수 ─[Poisson]→ 스파이크 ─[SNN + surrogate gradient]→ 분류
```

---

## 2. 이론적 배경

### 2.1 LIF (Leaky Integrate-and-Fire) 뉴런

이산 시간 LIF 뉴런의 막전위 업데이트는 다음과 같다.

$$ U[t+1] = \beta \, U[t] + I[t] - S[t]\,\theta $$

- β = exp(−dt/τ) : 막전위 누설 계수 (membrane decay)
- I[t] : 입력 전류 (광응답 소자에서는 광전류 I_ph)
- θ : 발화 임계전압(threshold), 도달 시 스파이크 S=1 발생 후 리셋

정전류 I를 계속 주입할 때, 발화가 일어나려면 정상상태 막전위 I/(1−β)가 임계를
넘어야 하므로 **rheobase 전류 = (1−β)·θ**. 이보다 큰 전류일수록 발화 주파수가 증가하는
단조 증가 f–I 곡선이 형성된다. → 이것이 "광세기 → 주파수" 변환의 물리적 근거.

### 2.2 Rate Coding (발화율 부호화)

아날로그 값(픽셀 명암, 주파수)을 스파이크열로 바꾸는 방식. 본 프로젝트는 **Poisson
rate coding** 을 사용한다. 각 타임스텝마다 발화 확률 p로 베르누이 시행을 수행하여
스파이크를 생성하며, p는 측정 주파수 f에 비례한다.

### 2.3 Surrogate Gradient

스파이크 함수(계단 함수)는 미분이 0 또는 ∞이라 역전파가 불가능하다. 순전파는 계단
함수를 그대로 쓰되, 역전파 시에만 매끄러운 함수(예: fast sigmoid)의 기울기로 근사하여
BPTT(Backpropagation Through Time)로 학습한다. snnTorch의 `surrogate.fast_sigmoid()` 사용.

---

## 3. 시스템 구조

| 단계 | 파일 | 역할 |
|------|------|------|
| Step 1 | `01_photo_neuron.py` | (참고) 수식 기반 광-뉴런 모델, 변환 곡선 검증 |
| Step 2 | `02_measured_encoder.py` | **실측 곡선 기반** grayscale→frequency→Poisson 인코더 |
| Step 3 | `03_train.py` | MNIST + 인코더 → 2층 SNN 학습/평가 |

---

## 4. 코드 상세

### 4.1 Step 1 — 광-뉴런 소자 모델 (`01_photo_neuron.py`)

소자를 하나의 LIF 뉴런으로 모델링한 **참고용** 구현. 광응답을 수식으로 기술한다.

**광응답(소자 물리 특성)** — 포화형(Michaelis-Menten) + 암전류:

$$ I_{ph}(\Phi) = I_{dark} + R \cdot \frac{\Phi^{\gamma}}{\Phi^{\gamma} + \Phi_{half}^{\gamma}} $$

| 파라미터 | 의미 |
|----------|------|
| R (responsivity) | 응답도(포화 시 최대 광전류) |
| Φ_half | 반포화 광세기 |
| γ (gamma) | 응답 비선형성 (γ<1 sub-linear, =1 포화, >1 super-linear) |
| I_dark | 암전류 |

`PhotoLIF` 클래스는 이 광전류를 snnTorch `snn.Leaky`에 입력하여 스파이크열을 생성한다.
`measure_transfer()` 는 여러 광세기를 batch로 흘려 각 광세기의 발화 주파수(Hz)를 측정한다
(주파수 = 총 스파이크 수 / 측정 시간). 출력: `fig1_transfer_curve.png`(변환 곡선),
`fig2_raster.png`(막전위 파형 + 래스터).

> Step 1은 소자 동작 원리를 이해하기 위한 수식 모델이며, 실제 학습에는 Step 2의
> **실측 데이터 기반 인코더**를 사용한다.

### 4.2 Step 2 — 측정 데이터 기반 인코더 (`02_measured_encoder.py`)

실제 소자에서 측정한 보정 곡선을 불러와, MNIST 명암을 실측 주파수로 변환하고 Poisson
스파이크열로 인코딩한다. **이 프로젝트의 핵심 모듈.**

**(1) 실측 곡선 로딩** — `load_calibration()`
- CSV/Excel 자동 판별, (광세기, 주파수) 배열 반환
- 광세기 기준 정렬 + 중복 광세기 평균 처리
- 파일이 없으면 합성 포화 곡선으로 fallback

**(2) 인코더** — `MeasuredEncoder`

매핑 과정:
```
 g ∈[0,1] ──(운용 범위)──▶ I_phys = g·intensity_white [mW/cm²]
          ──(실측 곡선 선형보간)──▶ f(g) [Hz]
          ──(정규화)──▶ p(g) = p_target · f(g)/f_ref   (auto_scale)
          ──(snntorch.spikegen.rate)──▶ Poisson 스파이크열 (T, ...)
```

설계상 중요한 두 가지:

- **운용 범위(`intensity_white`)**: 흰색(g=1)이 대응할 광세기. 측정 곡선이 강하게
  포화(50 kHz)하므로, 전 구간(0~9.43)을 쓰면 g>0.2 픽셀이 포화 영역에 몰려 명암 대비가
  뭉개진다. 따라서 응답이 살아있는 구간(기본 **2.0 mW/cm²**)만 사용해 대비를 확보한다.
- **auto-scale 정규화**: `p = p_target·f/f_ref` 형태라 dt와 절대 주파수가 상쇄되고
  *상대적* 명암 정보만 확률로 남는다. 측정 주파수의 단위·범위가 달라도 안전.

출력: `fig3`(소자 특성 + 인코딩 곡선), `fig4`(MNIST 원본/주파수맵/스파이크수),
`fig5`(픽셀 명암별 래스터).

### 4.3 Step 3 — SNN 학습 (`03_train.py`)

**네트워크 구조** (`SNN` 클래스):
```
 spk_in (T,B,784) ─▶ FC(784→256) ─▶ LIF ─▶ FC(256→10) ─▶ LIF ─▶ spk_out (T,B,10)
```
- 매 타임스텝 t마다 두 LIF 층을 통과시키고, 출력층 스파이크를 T스텝 누적
- 예측 = 출력 뉴런별 **총 스파이크 수가 최대인 클래스** (rate readout)

**학습 설정**:
- 손실: `SF.ce_count_loss` (출력 스파이크 수에 대한 cross-entropy)
- 최적화: Adam (lr=1e-3), surrogate gradient = `fast_sigmoid`
- β=0.9, num_steps=50, batch=128
- Step 2의 `MeasuredEncoder`를 `importlib`로 재사용 (파일명이 숫자로 시작하므로)

**데이터**: MNIST. CPU 학습 시간을 고려해 기본은 서브셋(train 6000 / test 1000);
전체(6만 장)는 `CFG["n_train"]=None`으로 전환.

출력: `snn_mnist.pt`(가중치), `fig6_training.png`(손실·정확도 곡선).

---

## 5. 측정 데이터 및 인코딩 설계

**측정 파일**: `C:\Claude\AI Cap\data\characteristic_curve.csv`
- 컬럼: `intensity_mW_cm2`, `frequency_Hz`
- 범위: 0~9.43 mW/cm², 0~50 kHz, **포화형 곡선**

`intensity_white = 2.0 mW/cm²` 설정에서 명암별 인코딩(발화 확률 분포):

| g (명암) | I [mW/cm²] | f [Hz] | p (per-step) |
|---------:|-----------:|-------:|-------------:|
| 0.00 | 0.0 | 0 | 0.00 |
| 0.25 | 0.5 | ~2,250 | ~0.05 |
| 0.50 | 1.0 | ~18,100 | ~0.40 |
| 0.75 | 1.5 | ~31,800 | ~0.70 |
| 1.00 | 2.0 | ~40,800 | 0.90 |

명암 전 구간에 발화 확률이 고르게 퍼져 학습에 유리한 동적 범위를 확보한다.

---

## 6. 실험 환경

- **OS**: Windows 11 Pro
- **Python**: 3.10 (전용 venv)
- **주요 패키지**: torch 2.5.1+cpu, snntorch 0.9.4, torchvision 0.20.1+cpu, pandas, matplotlib
- **환경 주의**: 전역 Miniconda의 torch 2.12.0이 `c10.dll` 초기화 실패(WinError 1114)로
  import 불가 → 전용 venv(`C:\Claude\snn_light2freq\.venv`)에 검증된 CPU torch 설치하여 해결.
- matplotlib 한글 표시: `Malgun Gothic` 폰트 적용.

---

## 7. 학습 결과 (100 epoch)

서브셋(train 6000 / test 1000), batch 128, lr 1e-3, num_steps 50, hidden 256 조건에서
100 에포크 학습한 결과는 다음과 같다.

| epoch | loss | train acc | test acc |
|------:|-----:|----------:|---------:|
| 1   | 1.312 | 76.8% | 70.4% |
| 5   | 0.122 | 97.7% | 92.4% |
| 10  | 0.034 | 98.9% | 93.1% |
| 20  | 0.004 | 99.9% | 94.6% |
| 30  | 0.001 | 100.0% | 94.8% |
| 50  | 0.000 | 100.0% | 94.7% |
| **60** | 0.004 | 99.9% | **95.1% (최고)** |
| 100 | 0.000 | 100.0% | 94.6% |

**관찰**
- 약 10 에포크 내에 빠르게 수렴(test 93% 도달), 이후 완만히 상승하여 **최고 test 95.1%(epoch 60)**.
- train 정확도는 ~30 에포크에서 100%, 손실은 0으로 포화 → **과적합**. 이후 test는
  94~95%에서 정체(개선 없음). 서브셋(6000장)만 사용했기 때문에 예상된 거동.
- epoch ~53 부근에서 손실이 일시적으로 튀고 정확도가 잠깐 하락 후 즉시 회복하는 transient가
  관찰됨 (Adam 최적화 중 일시적 불안정). 학습 안정성에는 영향 없음.

→ **결론**: 소자 실측 곡선 기반 인코딩 + SNN 학습 파이프라인이 정상 동작하며, 제한된
서브셋 조건에서 test 95% 수준의 분류 성능을 달성. 성능을 더 높이려면 전체 데이터 학습,
정규화(dropout/weight decay), 또는 early stopping(epoch 60 부근) 적용이 필요하다.

---

## 8. 고찰 및 향후 과제

- **인코딩 충실도**: 소자의 실측 변환 곡선을 인코더에 직접 반영하여, 실제 하드웨어
  동작에 근접한 스파이크 부호화를 구현했다.
- **운용 범위 선택**의 영향: 포화 곡선에서 `intensity_white`를 응답 구간으로 제한하는
  것이 명암 대비(=분류 성능)에 핵심적이다.
- **향후**:
  - 전체 MNIST(6만 장) 학습 및 GPU 가속
  - 측정 곡선의 시간 응답(상승/하강 시정수)까지 반영한 LIF 동역학(전류 역산 모드)
  - 다른 데이터셋, 소자별 변환 곡선 비교
