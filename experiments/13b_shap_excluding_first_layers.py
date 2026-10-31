"""SHAP analysis after removing layers 1-2, testing the mechanism robustness."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, shap
from xgboost import XGBClassifier
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
cols = list(range(len(nb))) + [15 + list(ns[15:]).index(c) for c in ('hist_entropy','grad_mean','grad_p99')]
X = Xs[:, cols]; names = [ns[i] for i in cols]
m0 = g >= 3; X, y, g = X[m0], y[m0], g[m0]
print(f"Excluding layers 1-2: {X.shape[0]} frames, {y.sum()} positives")

m = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  scale_pos_weight=20, eval_metric='logloss', random_state=0, n_jobs=4).fit(X, y)
sv = shap.TreeExplainer(m).shap_values(X)
imp = np.abs(sv).mean(0); order = np.argsort(imp)[::-1]
print("\nGlobal importance (layers 1-2 excluded):")
for i in order[:8]: print(f"  {names[i]:<14}{imp[i]:.4f}")
print("\nSHAP response by quintile (low to high):")
for fname in ('hist_entropy','grad_mean','std'):
    j = names.index(fname); q = pd.qcut(X[:, j], 5, labels=False, duplicates='drop')
    s = [sv[q==b, j].mean() for b in range(q.max()+1)]
    pr = [y[q==b].mean() for b in range(q.max()+1)]
    print(f"  {fname:<14}SHAP: " + " → ".join(f"{v:+.2f}" for v in s))
    print(f"  {'':<14}pore rate: " + " → ".join(f"{v:.3f}" for v in pr))

import os, sys
