"""
Step 2 — 측정 데이터 기반 light→frequency 인코더 (Poisson rate coding)
================================================================================

목표
----
실제 소자에서 측정한 "광세기(light intensity) ↔ 주파수(frequency)" 보정 곡선을
불러와, MNIST 픽셀의 흑백 명암(grayscale, 0~1)을 → 측정 주파수로 변환하고 →
Poisson 스파이크열로 인코딩한다. 이 인코더는 Step 3 학습에서 그대로 재사용한다.

파이프라인
----------
    MNIST 픽셀 g∈[0,1]
        │  (실측 보정 곡선; 선형보간)
        ▼
    측정 주파수 f(g) [Hz]
        │  p = clip(gain · f · dt, 0, 1)   (per-step 발화 확률)
        ▼
    Poisson 스파이크열  (T, ...)   ← snntorch.spikegen.rate

설정(CONFIG)
-----------
실측 파일의 경로와 컬럼명을 아래 CONFIG에서 지정한다.
- 파일이 .csv 또는 .xlsx/.xls 모두 지원 (확장자로 자동 판별)
- 컬럼은 "광세기" 1개 + "주파수" 1개. 광세기 단위는 임의(자동 정규화).
- 파일을 못 찾으면 현실적 포화형 합성 곡선으로 대체(경고 출력) → 나중에 실측으로 교체.
"""

import os
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 폰트 (Windows)
plt.rcParams["axes.unicode_minus"] = False

import snntorch.spikegen as spikegen


# ===========================================================================
# CONFIG — 실측 파일에 맞춰 여기만 수정하세요
# ===========================================================================
CONFIG = {
    "path": r"C:\Claude\AI Cap\data\characteristic_curve.csv",  # 실측 파일 경로
    "col_intensity": "intensity_mW_cm2",  # 광세기 컬럼명 (mW/cm²)
    "col_frequency": "frequency_Hz",      # 주파수 컬럼명 (Hz)
    "sheet": 0,                            # Excel일 때 시트 (이름 또는 index)
    "invert_intensity": False,            # 밝을수록 주파수가 '낮아지는' 소자면 True
    # 흰색(g=1)이 대응할 광세기[mW/cm²] = 조명의 운용 범위.
    #   None  → 측정 최대(9.43)까지 사용(전 구간; 포화로 대비 뭉개짐)
    #   1.4~2.0 권장 → 응답이 살아있는 구간만 사용해 명암 대비 확보
    "intensity_white": 2.0,
}

# 시뮬레이션 파라미터
DT = 1e-3          # 1 ms / step (auto_scale=True면 인코딩에는 영향 없음)
NUM_STEPS = 100    # MNIST 한 장당 100 스텝 동안 스파이크 생성(클수록 rate 추정 정밀)
P_TARGET = 0.9     # 흰색 픽셀에서의 목표 per-step 발화 확률 (auto-scale 기준)


# ===========================================================================
# 1. 실측 보정 곡선 로딩
# ===========================================================================
def load_calibration(cfg):
    """실측 (광세기, 주파수) 배열을 반환. 파일이 없으면 합성 곡선 fallback."""
    path = cfg["path"]
    if not os.path.exists(path):
        print(f"[warn] 측정 파일을 찾지 못함: {path}")
        print("[warn] → 현실적 합성 곡선으로 대체합니다. 실측 파일을 넣고 CONFIG를 맞추세요.")
        intensity = np.linspace(0.0, 1.0, 50)
        # 포화형(Michaelis-Menten) 합성 곡선: 0~200 Hz
        frequency = 200.0 * intensity / (intensity + 0.35)
        return intensity, frequency

    import pandas as pd
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, sheet_name=cfg["sheet"])
    else:
        df = pd.read_csv(path)

    intensity = df[cfg["col_intensity"]].to_numpy(dtype=float)
    frequency = df[cfg["col_frequency"]].to_numpy(dtype=float)

    # 광세기 기준 정렬 + 중복 광세기는 평균
    order = np.argsort(intensity)
    intensity, frequency = intensity[order], frequency[order]
    uniq, inv = np.unique(intensity, return_inverse=True)
    if len(uniq) != len(intensity):
        freq_avg = np.zeros_like(uniq)
        for k in range(len(uniq)):
            freq_avg[k] = frequency[inv == k].mean()
        intensity, frequency = uniq, freq_avg

    print(f"[ok] 측정 곡선 로딩: {len(intensity)} pts, "
          f"I∈[{intensity.min():.3g},{intensity.max():.3g}], "
          f"f∈[{frequency.min():.3g},{frequency.max():.3g}] Hz")
    return intensity, frequency


# ===========================================================================
# 2. 측정-기반 인코더
# ===========================================================================
class MeasuredEncoder:
    """grayscale(0~1) → 측정 주파수 → Poisson 스파이크열.

    매핑:  g ──(운용 범위)──▶ I_phys = g·intensity_white [mW/cm²]
              ──(실측 곡선 보간)──▶ f(g) [Hz]
              ──(정규화)──▶ p(g) = p_target·f(g)/f_ref  (auto_scale)
    """

    def __init__(self, intensity, frequency, dt=DT, num_steps=NUM_STEPS,
                 p_target=P_TARGET, invert=False, intensity_white=None,
                 auto_scale=True):
        intensity = np.asarray(intensity, dtype=float)
        frequency = np.asarray(frequency, dtype=float)
        order = np.argsort(intensity)            # 물리 광세기 기준 오름차순
        self.intensity = intensity[order]        # [mW/cm²]
        self.frequency = frequency[order]        # [Hz]
        self.dt = dt
        self.num_steps = num_steps
        self.p_target = p_target
        self.invert = invert
        self.auto_scale = auto_scale

        # 흰색(g=1)이 대응할 광세기(운용 범위 상한)
        self.intensity_white = (float(self.intensity.max())
                                if intensity_white is None else float(intensity_white))

        # 정규화 기준 주파수 f_ref = g∈[0,1]에서 도달 가능한 최대 주파수
        g_grid = np.linspace(0.0, 1.0, 256)
        self.f_ref = float(self.grayscale_to_freq(g_grid).max()) + 1e-12

        if not auto_scale:
            raw_pmax = self.f_ref * dt
            if raw_pmax > 1.0:
                print(f"[warn] auto_scale=False인데 f_ref·dt={raw_pmax:.1f}>1 → "
                      f"확률이 포화됩니다. dt를 줄이거나 auto_scale=True 권장.")

    # --- grayscale → 광세기 → 주파수 (실측 곡선 선형보간) ---
    def grayscale_to_freq(self, g):
        g = np.clip(np.asarray(g, dtype=float), 0.0, 1.0)
        if self.invert:
            g = 1.0 - g                          # 밝을수록 주파수↓ 소자
        i_phys = g * self.intensity_white        # 운용 범위로 스케일
        return np.interp(i_phys, self.intensity, self.frequency)

    # --- grayscale → per-step 발화 확률 ---
    def grayscale_to_prob(self, g_tensor: torch.Tensor) -> torch.Tensor:
        g = g_tensor.detach().cpu().numpy()
        f = self.grayscale_to_freq(g)
        if self.auto_scale:                      # dt·절대주파수 상쇄, 상대 명암만 사용
            p = self.p_target * f / self.f_ref
        else:                                    # 물리적 p = f·dt
            p = f * self.dt
        p = np.clip(p, 0.0, 1.0)
        return torch.as_tensor(p, dtype=torch.float32, device=g_tensor.device)

    # --- grayscale 이미지/배치 → Poisson 스파이크열 (T, *shape) ---
    def encode(self, img: torch.Tensor) -> torch.Tensor:
        prob = self.grayscale_to_prob(img)
        return spikegen.rate(prob, num_steps=self.num_steps)


# ===========================================================================
# 3. 데모: MNIST 한 장을 인코딩하고 시각화
# ===========================================================================
def demo():
    intensity, frequency = load_calibration(CONFIG)
    enc = MeasuredEncoder(intensity, frequency,
                          invert=CONFIG["invert_intensity"],
                          intensity_white=CONFIG["intensity_white"])

    # --- (A) 좌: 소자 실측 특성 / 우: grayscale→주파수·확률 인코딩 ---
    g = np.linspace(0, 1, 200)
    f = enc.grayscale_to_freq(g)
    prob = enc.grayscale_to_prob(torch.tensor(g)).numpy()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5))

    # 좌: 측정 특성 곡선 (광세기 vs 주파수) + 운용 범위 표시
    i_fine = np.linspace(enc.intensity.min(), enc.intensity.max(), 300)
    axL.plot(enc.intensity, enc.frequency, "o", ms=6, color="tab:gray", label="측정점")
    axL.plot(i_fine, np.interp(i_fine, enc.intensity, enc.frequency), "-",
             color="tab:red", label="보간")
    axL.axvline(enc.intensity_white, color="tab:green", ls="--",
                label=f"운용 상한 g=1 → {enc.intensity_white:g} mW/cm²")
    axL.set_xlabel("광세기 [mW/cm²]")
    axL.set_ylabel("주파수 [Hz]")
    axL.set_title("소자 실측 특성")
    axL.legend(fontsize=8)
    axL.grid(alpha=0.3)

    # 우: 인코더가 보는 grayscale→f, grayscale→p
    axR.plot(g, f, "-", color="tab:red", label="f(g)")
    axR.set_xlabel("grayscale g (픽셀 명암)")
    axR.set_ylabel("주파수 f [Hz]", color="tab:red")
    axR.tick_params(axis="y", labelcolor="tab:red")
    axR.grid(alpha=0.3)
    axR2 = axR.twinx()
    axR2.plot(g, prob, "--", color="tab:blue", label="p(g)")
    axR2.set_ylabel("per-step 발화 확률 p", color="tab:blue")
    axR2.tick_params(axis="y", labelcolor="tab:blue")
    axR.set_title("측정 기반 grayscale→frequency 인코딩")
    fig.tight_layout()
    fig.savefig("fig3_measured_transfer.png", dpi=130)
    print("[saved] fig3_measured_transfer.png")

    # --- (B) MNIST 한 장 인코딩 ---
    try:
        from torchvision import datasets, transforms
        ds = datasets.MNIST(root="./data", train=True, download=True,
                            transform=transforms.ToTensor())
        img, label = ds[0]            # img: (1,28,28) in [0,1]
        img = img.squeeze(0)          # (28,28)
    except Exception as e:
        print(f"[warn] MNIST 로딩 실패({e}); 합성 패턴으로 대체")
        yy, xx = np.mgrid[0:28, 0:28]
        img = torch.tensor(np.exp(-((xx-14)**2 + (yy-14)**2) / 60.0), dtype=torch.float32)
        label = -1

    spikes = enc.encode(img)          # (T, 28, 28)
    spike_count = spikes.sum(0)       # (28,28) — 총 스파이크 수 ≈ 주파수 비례

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img.numpy(), cmap="gray")
    axes[0].set_title(f"원본 명암 (label={label})")
    im1 = axes[1].imshow(enc.grayscale_to_freq(img.numpy()), cmap="inferno")
    axes[1].set_title("측정 주파수 맵 [Hz]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    im2 = axes[2].imshow(spike_count.numpy(), cmap="viridis")
    axes[2].set_title(f"스파이크 수 (T={enc.num_steps})")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    fig.savefig("fig4_mnist_encoding.png", dpi=130)
    print("[saved] fig4_mnist_encoding.png")

    # --- (C) 밝기 다른 대표 픽셀들의 래스터 ---
    g_samples = torch.tensor([0.1, 0.4, 0.8])
    sp = enc.encode(g_samples)        # (T, 3)
    t_axis = np.arange(enc.num_steps) * enc.dt * 1e3
    fig, ax = plt.subplots(figsize=(8, 3))
    for k, gv in enumerate(g_samples):
        st = t_axis[sp[:, k].numpy() > 0]
        ax.scatter(st, np.full_like(st, k), marker="|", s=200)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"g={v:.1f}\n{enc.grayscale_to_freq(float(v)):.0f}Hz"
                        for v in g_samples])
    ax.set_xlabel("시간 [ms]")
    ax.set_title("픽셀 명암별 Poisson 스파이크 래스터")
    fig.tight_layout()
    fig.savefig("fig5_pixel_raster.png", dpi=130)
    print("[saved] fig5_pixel_raster.png")


if __name__ == "__main__":
    demo()
