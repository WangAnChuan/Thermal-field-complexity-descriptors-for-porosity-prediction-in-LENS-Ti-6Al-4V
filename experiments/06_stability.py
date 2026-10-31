"""Ten seeds x seven forward-chaining boundaries, with paired bootstrap intervals."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
Xc = np.column_stack([Xb] + [Xs[:, idx[c]] for c in ('hist_entropy','grad_mean','grad_p99')])

def mk(name, seed):
    if name == 'XGB': return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, scale_pos_weight=20, eval_metric='logloss',
                                           random_state=seed, n_jobs=4)
    if name == 'Cat': return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05, auto_class_weights='Balanced',
                                                random_seed=seed, verbose=0, thread_count=4)
    return RandomForestClassifier(n_estimators=600, class_weight='balanced_subsample', min_samples_leaf=2,
                                  random_state=seed, n_jobs=4)

BOUNDS = [15, 20, 25, 30, 35, 40, 45]
rows = []
print(f"{'Model':<5}{'Protocol':<9}{'Feature':<12}{'PR-AUC mean±sd':>20}{'95%CI':>20}{'n':>4}")
for name in ('XGB', 'Cat', 'RF'):
    for proto in ('group5', 'forward'):
        for fname, X in (('base', Xb), ('entropy+gradient', Xc)):
            vals = []
            for seed in range(10):
                if proto == 'group5':
                    sp = list(GroupKFold(n_splits=5).split(X, y, g))
                else:
                    sp = [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in BOUNDS]
                for tr, te in sp:
                    p = mk(name, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
                    if (y[te] == 1).any(): vals.append(average_precision_score(y[te], p))
            vals = np.array(vals)
            lo, hi = np.percentile(vals, [2.5, 97.5])
            rows.append(dict(model=name, protocol=proto, feat=fname, mean=vals.mean(),
                             sd=vals.std(), ci_lo=lo, ci_hi=hi, n=len(vals)))
            r = rows[-1]
            print(f"{name:<5}{proto:<9}{fname:<12}{vals.mean():>14.3f}±{vals.std():.3f}"
                  f"[{lo:>7.3f},{hi:>6.3f}]{len(vals):>4}", flush=True)
pd.DataFrame(rows).to_csv('results/results_multiseed.csv', index=False)

# 增益的配对检验（同一 seed×split 配对）
print("\n=== Paired gain (identical splits, 1000 bootstrap)===")
for name in ('XGB', 'Cat', 'RF'):
    for proto in ('group5', 'forward'):
        d = []
        for seed in range(10):
            if proto == 'group5':
                sp = list(GroupKFold(n_splits=5).split(Xb, y, g))
            else:
                sp = [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in BOUNDS]
            for tr, te in sp:
                if not (y[te] == 1).any(): continue
                a = average_precision_score(y[te], mk(name, seed).fit(Xb[tr], y[tr]).predict_proba(Xb[te])[:, 1])
                b = average_precision_score(y[te], mk(name, seed).fit(Xc[tr], y[tr]).predict_proba(Xc[te])[:, 1])
                d.append(b - a)
        d = np.array(d); rng = np.random.RandomState(0)
        bs = np.array([rng.choice(d, len(d)).mean() for _ in range(1000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        print(f"{name:<5}{proto:<9} ΔPR={d.mean():+.3f}  95%CI [{lo:+.3f}, {hi:+.3f}]  p={max(p,1e-4):.4f}  (n={len(d)})", flush=True)
print("\nsaved results_multiseed.csv")

import os, sys
