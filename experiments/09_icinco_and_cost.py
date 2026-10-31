"""Reproduce the published random-split protocol and derive cost-sensitive operating points."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.model_selection import StratifiedShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_score, recall_score,
                             f1_score, accuracy_score)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
cols18 = list(range(len(nb))) + [15 + list(ns[15:]).index(c) for c in ('hist_entropy','grad_mean','grad_p99')]
X18 = Xs[:, cols18]
img = np.load('data/cache/thermal.npy')
rng = np.random.RandomState(0)

# ---------- (1) 复现 ICINCO 的 11 维统计Feature ----------
print("=== (1) ICINCO protocol reproduction (random 70/30 + SMOTE, 10 repeats) ===")
flat = img.reshape(len(img), -1)
Xic = np.column_stack([
    flat.min(1), flat.max(1), flat.std(1), np.median(flat, 1),
    np.percentile(flat, 25, 1), np.percentile(flat, 75, 1),
    np.percentile(flat, 75, 1) - np.percentile(flat, 25, 1),
    skew(flat, axis=1), kurtosis(flat, axis=1),
    (img > 1500).sum((1, 2)),
]).astype(np.float32)

def mk(name):
    if name == 'XGB': return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, scale_pos_weight=20, eval_metric='logloss',
                                           random_state=0, n_jobs=4)
    return RandomForestClassifier(n_estimators=500, class_weight='balanced_subsample', min_samples_leaf=3,
                                  random_state=0, n_jobs=4)

rows = []
for fname, X in (('icinco_stats', Xic), ('ours_21d', X18)):
    for m in ('RF', 'XGB'):
        met = {k: [] for k in ('P', 'R', 'F1', 'Acc', 'PR')}
        for seed in range(10):
            tr, te = next(StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=seed).split(X, y))
            Xtr, ytr = SMOTE(random_state=seed, k_neighbors=5).fit_resample(X[tr], y[tr])
            clf = mk(m).fit(Xtr, ytr)
            p = clf.predict_proba(X[te])[:, 1]
            pr = (p >= 0.5).astype(int)
            met['P'].append(precision_score(y[te], pr, zero_division=0))
            met['R'].append(recall_score(y[te], pr, zero_division=0))
            met['F1'].append(f1_score(y[te], pr, zero_division=0))
            met['Acc'].append(accuracy_score(y[te], pr))
            met['PR'].append(average_precision_score(y[te], p))
        rows.append(dict(feat=fname, model=m, **{k: np.mean(v) for k, v in met.items()},
                         **{f'{k}_sd': np.std(v) for k, v in met.items()}))
        r = rows[-1]
        print(f"{fname:<12}{m:<4} P={r['P']:.3f} R={r['R']:.3f} F1={r['F1']:.3f} Acc={r['Acc']:.3f} "
              f"PR-AUC={r['PR']:.3f}", flush=True)
pd.DataFrame(rows).to_csv('results/results_icinco_repro.csv', index=False)
print("  Reported in the source: CNN P0.90/R0.86/F1 0.88/Acc 0.99；RF 0.86/0.86/0.86/0.98；XGB 0.89/0.76/0.82/0.98")

# ---------- (2) cost-sensitive operating points ----------
print("\n=== (2) cost-sensitive operating points(forward chaining, CatBoost + entropy + gradient)===")
fwd = [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in (20, 30, 40, 50)]
def cat(): return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05, auto_class_weights='Balanced',
                                     random_seed=0, verbose=0, thread_count=4)
cost_rows = []
for R in (1, 2, 5, 10, 20, 50):
    recs, precs, costs = [], [], []
    for tr, te in fwd:
        # 内层 OOF 选阈值
        inner = GroupKFold(n_splits=3); oof = np.full(len(tr), np.nan)
        for itr, iva in inner.split(X18[tr], y[tr], g[tr]):
            oof[iva] = cat().fit(X18[tr][itr], y[tr][itr]).predict_proba(X18[tr][iva])[:, 1]
        ok = ~np.isnan(oof)
        ths = np.linspace(0.01, 0.99, 99)
        c = [R * ((y[tr][ok] == 1) & (oof[ok] < t)).sum() + ((y[tr][ok] == 0) & (oof[ok] >= t)).sum() for t in ths]
        th = ths[int(np.argmin(c))]
        p = cat().fit(X18[tr], y[tr]).predict_proba(X18[te])[:, 1]
        pred = (p >= th).astype(int)
        recs.append(recall_score(y[te], pred, zero_division=0))
        precs.append(precision_score(y[te], pred, zero_division=0))
        costs.append((R * ((y[te] == 1) & (pred == 0)).sum() + ((y[te] == 0) & (pred == 1)).sum()) / len(te) / R)
    cost_rows.append(dict(R=R, recall=np.mean(recs), precision=np.mean(precs), norm_cost=np.mean(costs)))
    r = cost_rows[-1]
    print(f"  Cost ratio {R:>2}:1  recall={r['recall']:.3f}  precision={r['precision']:.3f}  norm. cost={r['norm_cost']:.4f}", flush=True)
pd.DataFrame(cost_rows).to_csv('results/results_cost.csv', index=False)

print("\n(3) Abstention curves are produced by 10_abstention.py")
