"""相位作为 token 位置信息:区域 × 相位的 16 个测量 token,内容 = 量的值,位置 = 区域嵌入 + 相位嵌入,
单头注意力汇聚 -> 线性头。对照:去掉相位嵌入(四个相位的 token 只靠区域嵌入区分);只用 4 个区域级 token。
67 人,同一批 10 份划分 × 5 折,每折训练一个小模型;报患者级平衡准确率 / AUC 与区域 × 相位的平均注意力。"""
import sys, pickle, math
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, "analysis")
from gait_phase import load_patients, DOCTOR_Q, N_PHASE, PHASES
from attribute_repeated_splits import DOCTOR5, load as load_attr
torch.set_num_threads(4)

pat, _ = load_patients(Path("logs/gait_phase_3d"))
bank = pickle.load(open("logs/measure_bank_3d/bank.pkl", "rb"))
miss = {Path(v["json"]).stem.split("-")[0] for v in bank["videos"].values() if v["frac_ok"] == 0}
pids, X5, y = load_attr(Path("logs/geom_attributes_3d_doctor"), DOCTOR5)
keep = np.array([p not in miss for p in pids]); pids = [p for p, k in zip(pids, keep) if k]; y = y[keep]
cols = [f"ph{k}_{q}" for q in DOCTOR_Q for k in range(N_PHASE)]
M = np.array([[pat[p]["feats"].get(c, np.nan) for c in cols] for p in pids]); M = np.where(np.isnan(M), np.nanmedian(M, 0), M)
region_idx = torch.tensor([i for i in range(len(DOCTOR_Q)) for _ in range(N_PHASE)])
phase_idx = torch.tensor([k for _ in DOCTOR_Q for k in range(N_PHASE)])


class TokenAttn(nn.Module):
    def __init__(self, n_tok, d=8, use_phase=True, use_region=True):
        super().__init__()
        self.val = nn.Linear(1, d)
        self.e_region = nn.Embedding(len(DOCTOR_Q), d)
        self.e_phase = nn.Embedding(N_PHASE, d)
        self.q = nn.Parameter(torch.randn(d) / math.sqrt(d))
        self.head = nn.Linear(d, 1)
        self.use_phase, self.use_region, self.d = use_phase, use_region, d

    def forward(self, x, ridx, pidx):
        t = self.val(x.unsqueeze(-1))
        if self.use_region:
            t = t + self.e_region(ridx)
        if self.use_phase:
            t = t + self.e_phase(pidx)
        a = torch.softmax((t @ self.q) / math.sqrt(self.d), dim=1)
        pooled = (a.unsqueeze(-1) * t).sum(1)
        return self.head(pooled).squeeze(-1), a


def fit_predict(Xtr, ytr, Xte, ridx, pidx, use_phase, use_region, seed, epochs=400, lr=1e-2, wd=1e-2):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr_t, Xte_t = torch.tensor((Xtr - mu) / sd, dtype=torch.float32), torch.tensor((Xte - mu) / sd, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    pos_w = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), dtype=torch.float32)
    m = TokenAttn(Xtr.shape[1], use_phase=use_phase, use_region=use_region)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for _ in range(epochs):
        opt.zero_grad(); lo, _ = m(Xtr_t, ridx, pidx); loss = lossf(lo, ytr_t); loss.backward(); opt.step()
    with torch.no_grad():
        lo, a = m(Xte_t, ridx, pidx)
    return torch.sigmoid(lo).numpy(), a.numpy()


def macro(pred, y):
    return float(np.mean([((pred == c) & (y == c)).sum() / max((y == c).sum(), 1) for c in np.unique(y)]))


def run(X, ridx, pidx, use_phase, use_region, n=10):
    ms, aucs, attn = [], [], []
    for s in range(n):
        prob = np.zeros(len(y)); A = np.zeros((len(y), X.shape[1]))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=1000 + s).split(X, y):
            p, a = fit_predict(X[tr], y[tr], X[te], ridx, pidx, use_phase, use_region, seed=s)
            prob[te] = p; A[te] = a
        ms.append(macro((prob > 0.5).astype(int), y)); aucs.append(roc_auc_score(y, prob)); attn.append(A.mean(0))
    return np.array(ms), np.array(aucs), np.mean(attn, 0)


print(f"患者 {len(y)}, ASD {y.sum()}")
print(f"{'模型':44s} {'macro':>16s} {'AUC':>7s}")
for name, X, ridx, pidx, up, ur in (
    ("16 token: 区域嵌入 + 相位嵌入 + 注意力汇聚", M, region_idx, phase_idx, True, True),
    ("16 token: 只有区域嵌入(相位 token 不可区分)", M, region_idx, phase_idx, False, True),
    ("16 token: 只有相位嵌入(区域不可区分)", M, region_idx, phase_idx, True, False),
    ("16 token: 无位置嵌入(纯袋装)", M, region_idx, phase_idx, False, False),
):
    ms, aucs, attn = run(X, ridx, pidx, up, ur)
    print(f"{name:44s} {ms.mean():.3f} ± {ms.std():.3f}   {aucs.mean():.3f}")
    if up and ur:
        print("  平均注意力(行 = 量, 列 = " + " / ".join(PHASES) + "):")
        for i, q in enumerate(DOCTOR_Q):
            print(f"    {q:6s} " + "  ".join(f"{attn[i * N_PHASE + k]:.3f}" for k in range(N_PHASE)))
# 4 个区域级 token(周期均值),只有区域嵌入
C = np.stack([M[:, i * N_PHASE:(i + 1) * N_PHASE].mean(1) for i in range(len(DOCTOR_Q))], 1)
ms, aucs, _ = run(C, torch.arange(len(DOCTOR_Q)), torch.zeros(len(DOCTOR_Q), dtype=torch.long), False, True)
print(f"{'4 token: 区域级, 不分相位':44s} {ms.mean():.3f} ± {ms.std():.3f}   {aucs.mean():.3f}")
