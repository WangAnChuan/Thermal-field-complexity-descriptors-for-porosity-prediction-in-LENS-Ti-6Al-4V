"""Control ablations: is the gain information or simply three more features?

五种条件，均为 18 维基线 + 3 维新增：
  base        基线
  permuted    复杂度描述符整体按帧打乱（边缘分布相同、与标签无关）——真零假设
  moments     同数量的通用高阶统计量（偏度、峰度、RMS）——等量但无空间结构主张
  randproj    标准化温度场上的 3 个随机线性投影——携带信息但非特定结构
  complexity  本文的熵 + 两个梯度统计量
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from scipy.stats import skew, kurtosis
from catboost import CatBoostClassifier
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build()
Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
EG = Xs[:, [idx['hist_entropy'], idx['grad_mean'], idx['grad_p99']]]
img = np.load('data/cache/thermal.npy')

# 通用高阶统计量
flat = img.reshape(len(img), -1)
MOM = np.column_stack([skew(flat, axis=1), kurtosis(flat, axis=1),
                       np.sqrt((flat ** 2).mean(1))]).astype(np.float32)

# 随机线性投影（先逐帧标准化，避免被整体亮度主导）
Z = (flat - flat.mean(1, keepdims=True)) / (flat.std(1, keepdims=True) + 1e-9)
def projections(seed):
    rng = np.random.RandomState(seed)
    R = rng.randn(Z.shape[1], 3).astype(np.float32)
    P = Z @ R
    return (P / (P.std(0) + 1e-9)).astype(np.float32)

def permuted(seed):
    rng = np.random.RandomState(seed)
    P = EG.copy()
    for j in range(P.shape[1]):
        P[:, j] = rng.permutation(P[:, j])
    return P

gkf = list(GroupKFold(n_splits=5).split(Xb, y, g))
fwd = [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in (20, 30, 40, 50)]

def cat():
    return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05,
                              auto_class_weights='Balanced', random_seed=0,
                              verbose=0, thread_count=4)

def ev(X, sp):
    a = []
    for tr, te in sp:
        p = cat().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        if (y[te] == 1).any():
            a.append(average_precision_score(y[te], p))
    return float(np.mean(a))

def add(extra):
    return np.column_stack([Xb, extra]).astype(np.float32)

rows = []
def record(cond, f_vals, g_vals):
    rows.append(dict(condition=cond,
                     forward=float(np.mean(f_vals)), forward_sd=float(np.std(f_vals)),
                     grouped=float(np.mean(g_vals)), grouped_sd=float(np.std(g_vals)),
                     n_repeats=len(f_vals)))
    r = rows[-1]
    print(f"{cond:<12} forward {r['forward']:.4f}±{r['forward_sd']:.4f}   "
          f"grouped {r['grouped']:.4f}±{r['grouped_sd']:.4f}   (n={r['n_repeats']})", flush=True)

print("condition    forward                    grouped")
record('base', [ev(Xb, fwd)], [ev(Xb, gkf)])
record('complexity', [ev(add(EG), fwd)], [ev(add(EG), gkf)])
record('moments', [ev(add(MOM), fwd)], [ev(add(MOM), gkf)])

perm_f, perm_g = [], []
for s in range(10):
    perm_f.append(ev(add(permuted(s)), fwd)); perm_g.append(ev(add(permuted(s)), gkf))
record('permuted', perm_f, perm_g)

rp_f, rp_g = [], []
for s in range(5):
    rp_f.append(ev(add(projections(s)), fwd)); rp_g.append(ev(add(projections(s)), gkf))
record('randproj', rp_f, rp_g)

df = pd.DataFrame(rows)
df.to_csv('results/results_control_ablation.csv', index=False)
print("\nsaved results_control_ablation.csv")

base_f = df[df.condition == 'base'].forward.iloc[0]
print("\nForward gain over the baseline:")
for _, r in df.iterrows():
    print(f"  {r.condition:<12} {r.forward-base_f:+.4f}")
