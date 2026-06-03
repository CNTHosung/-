"""
Step 3 — MNIST + 측정 기반 인코더 → SNN 분류기 학습/평가
================================================================================

목표
----
Step 2의 MeasuredEncoder(소자 실측 곡선 기반 grayscale→frequency→Poisson)를 써서
MNIST 흑백 명암을 스파이크열로 인코딩하고, 2층 SNN을 surrogate gradient(BPTT)로 학습한다.

네트워크
--------
    spk_in (T,B,784) ──▶ FC(784→H) ──▶ LIF ──▶ FC(H→10) ──▶ LIF ──▶ spk_out (T,B,10)
    예측 = 출력 뉴런별 총 스파이크 수가 가장 큰 클래스 (rate readout)

손실 = SF.ce_count_loss (출력 스파이크 수에 대한 cross-entropy)
"""

import os
import importlib.util
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import snntorch as snn
from snntorch import surrogate
import snntorch.functional as SF

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Malgun Gothic"   # 한글 폰트 (Windows)
plt.rcParams["axes.unicode_minus"] = False


# ===========================================================================
# 0. Step 2의 인코더 모듈 재사용 (파일명이 숫자로 시작 → importlib로 로드)
# ===========================================================================
HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "enc02", os.path.join(HERE, "02_measured_encoder.py"))
enc02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(enc02)


# ===========================================================================
# 1. 학습 설정
# ===========================================================================
CFG = {
    "hidden": 256,
    "num_steps": 50,       # 인코딩/시뮬 스텝 (작을수록 빠름, 클수록 rate 추정 정밀)
    "batch": 128,
    "epochs": 100,
    "lr": 1e-3,
    "beta": 0.9,           # LIF 막전위 누설
    # CPU에서 빠르게 보려면 서브셋 사용. 전체로 돌리려면 None.
    "n_train": 6000,
    "n_test": 1000,
    "seed": 0,
}


# ===========================================================================
# 2. SNN 모델
# ===========================================================================
class SNN(nn.Module):
    def __init__(self, n_in=784, n_hidden=256, n_out=10, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()      # surrogate gradient
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, spk_in):                     # spk_in: (T, B, 784)
        T, B, _ = spk_in.shape
        dev = spk_in.device
        mem1 = torch.zeros(B, self.fc1.out_features, device=dev)
        mem2 = torch.zeros(B, self.fc2.out_features, device=dev)
        spk2_rec = []
        for t in range(T):
            cur1 = self.fc1(spk_in[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
        return torch.stack(spk2_rec)               # (T, B, 10)


# ===========================================================================
# 3. 학습 / 평가
# ===========================================================================
def run():
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # --- 인코더 (Step 2 실측 곡선) ---
    intensity, frequency = enc02.load_calibration(enc02.CONFIG)
    encoder = enc02.MeasuredEncoder(
        intensity, frequency,
        num_steps=CFG["num_steps"],
        invert=enc02.CONFIG["invert_intensity"],
        intensity_white=enc02.CONFIG["intensity_white"],
    )

    # --- 데이터 ---
    from torchvision import datasets, transforms
    tfm = transforms.ToTensor()                    # → (1,28,28) in [0,1]
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    if CFG["n_train"]:
        train_ds = Subset(train_ds, range(CFG["n_train"]))
    if CFG["n_test"]:
        test_ds = Subset(test_ds, range(CFG["n_test"]))
    train_dl = DataLoader(train_ds, batch_size=CFG["batch"], shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=CFG["batch"], shuffle=False)
    print(f"[data] train={len(train_ds)} test={len(test_ds)}")

    # --- 모델 ---
    net = SNN(784, CFG["hidden"], 10, beta=CFG["beta"]).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=CFG["lr"])
    loss_fn = SF.ce_count_loss()                   # 출력 스파이크 수 기반 CE

    def encode_batch(imgs):
        """(B,1,28,28) → (T,B,784) Poisson 스파이크."""
        g = imgs.view(imgs.size(0), -1)            # (B,784) in [0,1]
        spk = encoder.encode(g)                    # (T,B,784)
        return spk.to(device)

    @torch.no_grad()
    def evaluate(dl):
        net.eval()
        correct = total = 0
        for imgs, labels in dl:
            spk_in = encode_batch(imgs)
            spk_out = net(spk_in)                   # (T,B,10)
            pred = spk_out.sum(0).argmax(1).cpu()   # rate readout
            correct += (pred == labels).sum().item()
            total += labels.size(0)
        return correct / total

    # --- 학습 루프 ---
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(CFG["epochs"]):
        net.train()
        running = 0.0
        for i, (imgs, labels) in enumerate(train_dl):
            labels = labels.to(device)
            spk_in = encode_batch(imgs)
            spk_out = net(spk_in)
            loss = loss_fn(spk_out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()
            if (i + 1) % 10 == 0:
                print(f"  epoch {epoch+1} batch {i+1}/{len(train_dl)} "
                      f"loss {loss.item():.3f}")
        avg_loss = running / len(train_dl)
        tr_acc = evaluate(train_dl)
        te_acc = evaluate(test_dl)
        hist["loss"].append(avg_loss)
        hist["train_acc"].append(tr_acc)
        hist["test_acc"].append(te_acc)
        print(f"[epoch {epoch+1}] loss {avg_loss:.3f} | "
              f"train {tr_acc*100:.1f}% | test {te_acc*100:.1f}%")

    # --- 결과 저장 ---
    torch.save(net.state_dict(), os.path.join(HERE, "snn_mnist.pt"))
    print("[saved] snn_mnist.pt")

    ep = range(1, CFG["epochs"] + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    a1.plot(ep, hist["loss"], "o-")
    a1.set_xlabel("epoch"); a1.set_ylabel("loss"); a1.set_title("학습 손실")
    a1.grid(alpha=0.3)
    a2.plot(ep, [a*100 for a in hist["train_acc"]], "o-", label="train")
    a2.plot(ep, [a*100 for a in hist["test_acc"]], "s-", label="test")
    a2.set_xlabel("epoch"); a2.set_ylabel("accuracy [%]")
    a2.set_title("정확도"); a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig6_training.png"), dpi=130)
    print("[saved] fig6_training.png")


if __name__ == "__main__":
    run()
