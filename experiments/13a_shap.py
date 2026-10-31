"""SHAP dependence analysis of the thermal-field complexity descriptors."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, shap
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
cols = list(range(len(nb))) + [15 + list(ns[15:]).index(c) for c in ('hist_entropy','grad_mean','grad_p99')]
X = Xs[:, cols]; names = [ns[i] for i in cols]

m = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                  scale_pos_weight=20, eval_metric='logloss', random_state=0, n_jobs=4).fit(X, y)
expl = shap.TreeExplainer(m)
sv = expl.shap_values(X)
print(f"SHAP matrix {sv.shape}")
imp = np.abs(sv).mean(0); order = np.argsort(imp)[::-1]
print("\nGlobal importance (mean |SHAP|):")
for i in order[:10]:
    print(f"  {names[i]:<14}{imp[i]:.4f}")

# 机理：关键Feature的分位曲线
print("\nSHAP response by feature quintile:")
rows = []
for fname in ('hist_entropy', 'grad_mean', 'std'):
    j = names.index(fname)
    q = pd.qcut(X[:, j], 5, labels=False, duplicates='drop')
    for b in range(q.max() + 1):
        mask = q == b
        rows.append(dict(feature=fname, bin=b, x_mean=float(X[mask, j].mean()),
                         shap_mean=float(sv[mask, j].mean()),
                         pos_rate=float(y[mask].mean())))
    sub = [r for r in rows if r['feature'] == fname]
    print(f"  {fname}: " + " → ".join(f"{r['shap_mean']:+.2f}" for r in sub) + "  (SHAP, low to high)")
    print(f"  {'':<{len(fname)}}  pore rate: " + " → ".join(f"{r['pos_rate']:.2f}" for r in sub))
pd.DataFrame(rows).to_csv('results/shap_bins.csv', index=False)

np.save('results/shap_values.npy', sv)
pd.Series(imp, index=names).to_csv('results/shap_importance.csv')
print("\nsaved results/shap_bins.csv and shap_importance.csv")

import os, sys
