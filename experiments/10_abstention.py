"""Selective prediction: miss rate as a function of automatic coverage."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
X18 = np.column_stack([Xb] + [Xs[:, idx[c]] for c in ('hist_entropy','grad_mean','grad_p99')])
fwd = [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in (20, 30, 40, 50)]
def cat(): return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05, auto_class_weights='Balanced',
                                     random_seed=0, verbose=0, thread_count=4)

print("=== Selective prediction (forward chaining) ===")
print(f"{'Feature':<10}{'Coverage':>8}{'Miss rate':>9}{'False-alarm rate':>9}")
rows = []
for fname, X in (('base', Xb), ('entropy+gradient', X18)):
    P = np.full(len(y), np.nan)
    for tr, te in fwd:
        P[te] = cat().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    m = ~np.isnan(P); conf = np.abs(P[m] - 0.5); yy = y[m]; pp = P[m]
    order = np.argsort(-conf)
    for cov in (0.2, 0.4, 0.6, 0.8, 0.9, 1.0):
        sub = order[:int(len(yy) * cov)]
        miss = ((yy[sub] == 1) & (pp[sub] < 0.5)).sum() / max((yy == 1).sum(), 1)
        fa = ((yy[sub] == 0) & (pp[sub] >= 0.5)).sum() / max((yy == 0).sum(), 1)
        rows.append(dict(feat=fname, coverage=cov, miss_rate=miss, false_alarm=fa))
        print(f"{fname:<10}{cov:>8.1f}{miss:>9.3f}{fa:>9.3f}", flush=True)
pd.DataFrame(rows).to_csv('results/results_selective.csv', index=False)
print("\nsaved results_selective.csv")

import os, sys
