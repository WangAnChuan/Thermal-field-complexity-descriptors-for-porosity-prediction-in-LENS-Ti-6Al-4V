"""Methodological follow-ups: protocol comparison, positional ablation, held-out layers, block bootstrap.

E1 位置特征消融：去掉 elapsed time / Y / Z 后，描述符增益是否仍在
E2 随机划分同口径对比：同一特征集在随机划分下的差异（PR-AUC 与 F1）
E3 嵌套选择：51-60 层完全留出，从不参与任何设计决策，做一次性最终评价
E4 分块 bootstrap：按前推边界聚类重采样，修正重叠窗口导致的 CI 偏窄
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
EG = Xs[:, [idx['hist_entropy'], idx['grad_mean'], idx['grad_p99']]]
POS = [nb.index(c) for c in ('Time', 'Y', 'Z')]
Xb_no = np.delete(Xb, POS, axis=1); nb_no = [c for i, c in enumerate(nb) if i not in POS]
X_no_eg = np.column_stack([Xb_no, EG]).astype(np.float32)
print(f"基线 {Xb.shape[1]} 维 → 去位置后 {Xb_no.shape[1]} 维（去掉 {['Time','Y','Z']}）")

def mk(name, seed=0):
    if name == 'XGB':
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=20, eval_metric='logloss',
                             random_state=seed, n_jobs=4)
    if name == 'RF':
        return RandomForestClassifier(n_estimators=600, class_weight='balanced_subsample',
                                      min_samples_leaf=2, random_state=seed, n_jobs=4)
    return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05,
                              auto_class_weights='Balanced', random_seed=seed, verbose=0, thread_count=4)

BOUNDS = [15, 20, 25, 30, 35, 40, 45]
def fwd_splits(bounds=BOUNDS):
    out = []
    for k in bounds:
        tr = np.where(g <= k)[0]; te = np.where((g > k) & (g <= k + 10))[0]
        if (y[te] == 1).any(): out.append((k, tr, te))
    return out

rows = []

# ---------------- E1 位置特征消融 ----------------
print("\n=== E1 位置特征消融（前推 7 边界 × 3 种子）===")
for model in ('Cat', 'XGB', 'RF'):
    for cond, X in (('with-position, base', Xb), ('with-position, +EG', np.column_stack([Xb, EG])),
                    ('no-position, base', Xb_no), ('no-position, +EG', X_no_eg)):
        aps = []
        for seed in range(3):
            for k, tr, te in fwd_splits():
                p = mk(model, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                aps.append(average_precision_score(y[te], p))
        rows.append(dict(experiment='E1', model=model, condition=cond,
                         mean=float(np.mean(aps)), sd=float(np.std(aps)), n=len(aps)))
        print(f"{model:<4}{cond:<22}{np.mean(aps):.4f} ± {np.std(aps):.4f}")
    wb = [r for r in rows if r['experiment'] == 'E1' and r['model'] == model and r['condition'] == 'with-position, base'][0]['mean']
    we = [r for r in rows if r['experiment'] == 'E1' and r['model'] == model and r['condition'] == 'with-position, +EG'][0]['mean']
    nb_ = [r for r in rows if r['experiment'] == 'E1' and r['model'] == model and r['condition'] == 'no-position, base'][0]['mean']
    ne = [r for r in rows if r['experiment'] == 'E1' and r['model'] == model and r['condition'] == 'no-position, +EG'][0]['mean']
    print(f"  → 有位置增益 {we-wb:+.4f} | 无位置增益 {ne-nb_:+.4f}\n")

# ---------------- E2 随机划分同口径 ----------------
print("=== E2 随机划分同口径（70/30 + SMOTE，10 次）===")
for model in ('Cat', 'XGB', 'RF'):
    for cond, X in (('base', Xb), ('+EG', np.column_stack([Xb, EG]))):
        prs, f1s = [], []
        for seed in range(10):
            from sklearn.model_selection import train_test_split
            tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
            Xtr, ytr = SMOTE(random_state=seed, k_neighbors=5).fit_resample(X[tr], y[tr])
            p = mk(model, seed).fit(Xtr, ytr).predict_proba(X[te])[:, 1]
            prs.append(average_precision_score(y[te], p)); f1s.append(f1_score(y[te], (p >= 0.5).astype(int)))
        rows.append(dict(experiment='E2', model=model, condition=cond,
                         mean=float(np.mean(prs)), sd=float(np.std(prs)),
                         f1=float(np.mean(f1s)), n=10))
        print(f"{model:<4}{cond:<8}PR-AUC {np.mean(prs):.4f} ± {np.std(prs):.4f}   F1 {np.mean(f1s):.4f}")
    a = [r for r in rows if r['experiment'] == 'E2' and r['model'] == model and r['condition'] == 'base'][0]
    b = [r for r in rows if r['experiment'] == 'E2' and r['model'] == model and r['condition'] == '+EG'][0]
    print(f"  → ΔPR {b['mean']-a['mean']:+.4f}   ΔF1 {b['f1']-a['f1']:+.4f}\n")

# ---------------- E3 留出末段 ----------------
print("=== E3 留出第 51-60 层（从不参与设计）===")
tr = np.where(g <= 50)[0]; te = np.where(g > 50)[0]
print(f"训练 {len(tr)} 帧 ({y[tr].sum()} 正例) | 测试 {len(te)} 帧 ({y[te].sum()} 正例)")
for model in ('Cat', 'XGB', 'RF'):
    for cond, X in (('base', Xb), ('+EG', np.column_stack([Xb, EG]))):
        prs = []
        for seed in range(5):
            p = mk(model, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            prs.append(average_precision_score(y[te], p))
        rows.append(dict(experiment='E3', model=model, condition=cond,
                         mean=float(np.mean(prs)), sd=float(np.std(prs)), n=5))
        print(f"{model:<4}{cond:<8}{np.mean(prs):.4f} ± {np.std(prs):.4f}")

# ---------------- E4 分块 bootstrap ----------------
print("\n=== E4 分块 bootstrap（按边界聚类）===")
for model in ('Cat', 'XGB'):
    X0 = Xb; X1 = np.column_stack([Xb, EG])
    per_bound = {}
    for seed in range(3):
        for k, tr, te in fwd_splits():
            a = average_precision_score(y[te], mk(model, seed).fit(X0[tr], y[tr]).predict_proba(X0[te])[:, 1])
            b = average_precision_score(y[te], mk(model, seed).fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1])
            per_bound.setdefault(k, []).append(b - a)
    ks = sorted(per_bound); means = np.array([np.mean(per_bound[k]) for k in ks])
    rng = np.random.RandomState(0)
    bs = np.array([rng.choice(means, len(means)).mean() for _ in range(5000)])
    lo, hi = np.percentile(bs, [2.5, 97.5]); p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    print(f"{model}: 边界均值 ΔPR={means.mean():+.4f}  分块 95%CI [{lo:+.4f}, {hi:+.4f}]  p={max(p,1e-4):.4f}  (k={ks})")
    rows.append(dict(experiment='E4', model=model, condition='complexity-moments-free delta',
                     mean=float(means.mean()), sd=float(means.std()), n=len(ks)))

pd.DataFrame(rows).to_csv('results/results_e1_e4.csv', index=False)
print("\nsaved results_e1_e4.csv")
