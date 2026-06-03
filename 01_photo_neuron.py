"""
Step 1 — 광-뉴런(Photo-LIF) 소자 모델 & "light intensity → frequency" 변환 곡선 검증
================================================================================

목표
----
광응답 소자를 하나의 LIF(Leaky Integrate-and-Fire) 뉴런 하드웨어로 모델링한다.
입사 광세기 Φ 가 소자의 광전류 I_ph(Φ) 를 만들고, 이 전류가 막전위를 충전시켜
스파이크를 발생시킨다. 광세기가 클수록 발화 주파수(frequency)가 높아지는,
즉 "광세기 → 주파수" 변환 특성을 눈으로 확인하는 것이 이 단계의 목표.

물리 모델
---------
1) 광응답 (소자 특성):  포화형(Michaelis-Menten) + 암전류(dark current)
        I_ph(Φ) = I_dark + R · Φ^γ / (Φ^γ + Φ_half^γ)
   - R       : 응답도(responsivity), 포화 시 최대 광전류
   - Φ_half  : 반포화 광세기 (이 값에서 응답이 절반)
   - γ       : 응답 비선형성 (γ<1 → sub-linear, =1 → 포화곡선, >1 → super-linear)
   - I_dark  : 암전류 (빛이 없을 때의 누설 전류)

2) 뉴런 동역학 (snnTorch Leaky):  이산 LIF
        U[t+1] = β·U[t] + I_ph        (임계 도달 시 발화 후 리셋)
   - β = exp(-dt/τ_m) : 막전위 누설 (membrane decay)
   - 임계전압 threshold 도달 시 스파이크 1 발생 → 막전위 리셋

발화율(rheobase) 직관
---------------------
정전류 I 를 계속 넣을 때 누설만 있고 발화가 없다면 정상상태 막전위는 I/(1-β).
따라서 발화가 일어나려면  I > (1-β)·threshold  (= rheobase 전류).
광응답을 이 rheobase 근처~수배 범위로 스케일하면, 어두울 때는 거의 발화하지 않고
밝아질수록 주파수가 증가하는 깔끔한 변환 곡선이 나온다.
"""

import numpy as np
import torch
import torch.nn as nn
import snntorch as snn
import matplotlib

matplotlib.use("Agg")  # 창 없이 파일로 저장 (Windows/headless 안전)
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 폰트 (Windows)
plt.rcParams["axes.unicode_minus"] = False      # 마이너스 기호 깨짐 방지


# ---------------------------------------------------------------------------
# 1. 광-뉴런 소자 모델
# ---------------------------------------------------------------------------
class PhotoLIF(nn.Module):
    """광응답 소자 = LIF 뉴런 하드웨어.

    forward(phi, num_steps):
        phi       : (batch,) 텐서, 각 원소가 '시간에 대해 일정한' 광세기 Φ
        num_steps : 시뮬레이션 스텝 수
        반환       : spk_rec (T, batch), mem_rec (T, batch), i_ph (batch,)
    """

    def __init__(
        self,
        beta: float = 0.9,          # 막전위 누설 β = exp(-dt/τ)
        threshold: float = 1.0,     # 발화 임계전압
        responsivity: float = 1.5,  # 광응답 포화 전류 R
        phi_half: float = 0.35,     # 반포화 광세기 Φ_half
        gamma: float = 1.0,         # 광응답 비선형성 γ
        i_dark: float = 0.02,       # 암전류 I_dark
    ):
        super().__init__()
        # snnTorch의 표준 Leaky 뉴런이 막전위 적분/발화/리셋을 담당
        self.lif = snn.Leaky(beta=beta, threshold=threshold)
        self.responsivity = responsivity
        self.phi_half = phi_half
        self.gamma = gamma
        self.i_dark = i_dark

    def photocurrent(self, phi: torch.Tensor) -> torch.Tensor:
        """광세기 Φ → 광전류 I_ph (소자 물리 특성)."""
        phi = torch.clamp(phi, min=0.0)
        num = phi ** self.gamma
        resp = self.responsivity * num / (num + self.phi_half ** self.gamma)
        return self.i_dark + resp

    def forward(self, phi: torch.Tensor, num_steps: int):
        i_ph = self.photocurrent(phi)            # (batch,) 일정한 입력 전류
        mem = torch.zeros_like(i_ph)             # 막전위 초기화
        spk_rec, mem_rec = [], []
        for _ in range(num_steps):
            spk, mem = self.lif(i_ph, mem)       # 한 스텝 적분 + 발화
            spk_rec.append(spk)
            mem_rec.append(mem)
        return torch.stack(spk_rec), torch.stack(mem_rec), i_ph


# ---------------------------------------------------------------------------
# 2. 변환 특성 측정 (light → frequency)
# ---------------------------------------------------------------------------
def measure_transfer(device: PhotoLIF, phis, num_steps: int, dt: float):
    """여러 광세기를 한 번에(batch) 흘려 각 광세기의 발화 주파수를 측정."""
    phis = torch.as_tensor(np.asarray(phis), dtype=torch.float32)
    with torch.no_grad():
        spk, mem, i_ph = device(phis, num_steps)
    counts = spk.sum(dim=0)                       # (batch,) 총 스파이크 수
    freq = counts / (num_steps * dt)              # Hz = 스파이크수 / 측정시간(s)
    return freq.numpy(), i_ph.numpy()


# ---------------------------------------------------------------------------
# 3. 메인: 변환 곡선 + 샘플 파형 그리기
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(0)

    dt = 1e-3            # 1 ms / step
    num_steps = 1000     # → 1.0 s 측정 윈도우
    device = PhotoLIF(beta=0.9, threshold=1.0,
                      responsivity=1.5, phi_half=0.35, gamma=1.0, i_dark=0.02)

    # (A) 광세기 스윕 → 변환 곡선 ------------------------------------------------
    phis = np.linspace(0.0, 1.0, 41)             # 정규화 광세기 0~1
    freq, i_ph = measure_transfer(device, phis, num_steps, dt)
    rheobase = (1 - 0.9) * 1.0                    # (1-β)·threshold

    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(phis, freq, "o-", color="tab:red", label="발화 주파수 f(Φ)")
    ax1.set_xlabel("광세기 Φ (정규화)")
    ax1.set_ylabel("스파이크 주파수 [Hz]", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(phis, i_ph, "--", color="tab:blue", label="광전류 I_ph(Φ)")
    ax2.axhline(rheobase, color="gray", ls=":", lw=1)
    ax2.text(0.02, rheobase, " rheobase", color="gray", va="bottom", fontsize=8)
    ax2.set_ylabel("광전류 I_ph [a.u.]", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("Step 1 — 소자 변환 특성: light intensity → frequency")
    fig.tight_layout()
    fig.savefig("fig1_transfer_curve.png", dpi=130)
    print("[saved] fig1_transfer_curve.png")

    # (B) 대표 광세기 3개의 막전위 + 래스터 --------------------------------------
    sample_phis = torch.tensor([0.1, 0.4, 0.9])
    with torch.no_grad():
        spk, mem, i_ph = device(sample_phis, num_steps)
    t_axis = np.arange(num_steps) * dt * 1e3      # ms

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    colors = ["tab:green", "tab:orange", "tab:red"]
    for k, phi in enumerate(sample_phis):
        axes[0].plot(t_axis, mem[:, k].numpy(), color=colors[k],
                     label=f"Φ={phi:.1f}  (I_ph={i_ph[k]:.2f})")
    axes[0].axhline(1.0, color="gray", ls=":", lw=1, label="threshold")
    axes[0].set_ylabel("막전위 U(t)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("대표 광세기의 막전위 파형 & 스파이크 래스터")

    for k, phi in enumerate(sample_phis):
        spike_times = t_axis[spk[:, k].numpy() > 0]
        axes[1].scatter(spike_times, np.full_like(spike_times, k),
                        marker="|", s=200, color=colors[k])
    axes[1].set_yticks(range(len(sample_phis)))
    axes[1].set_yticklabels([f"Φ={p:.1f}" for p in sample_phis])
    axes[1].set_xlabel("시간 [ms]")
    axes[1].set_ylabel("채널")
    fig.tight_layout()
    fig.savefig("fig2_raster.png", dpi=130)
    print("[saved] fig2_raster.png")

    # (C) 콘솔 요약 --------------------------------------------------------------
    print("\n광세기 → 주파수 변환 요약")
    print(f"{'Φ':>6} {'I_ph':>8} {'f[Hz]':>8}")
    for p in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        f, i = measure_transfer(device, [p], num_steps, dt)
        print(f"{p:6.2f} {i[0]:8.3f} {f[0]:8.1f}")


if __name__ == "__main__":
    main()
