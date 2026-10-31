"""Search for the smallest feature combination that retains the full gain."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
def cols_of(names): return [idx[c] for c in names if c in idx]

COMBOS = {
    'base': [],
    'base+entropy': ['hist_entropy'],
    'base+gradient': ['grad_mean', 'grad_p99'],
    'base+entropy+gradient': ['hist_entropy', 'grad_mean', 'grad_p99'],
    'base+entropy+gradient+area': ['hist_entropy', 'grad_mean', 'grad_p99', 't1000_area','t1200_area','t1400_area','t1600_area'],
    'full_spatial': list(set(range(len(nb), len(ns)))),
}
fwd = [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in (20, 30, 40, 50)]
grp5 = list(GroupKFold(n_splits=5).split(Xb, y, g))

def mk(name):
    if name == 'Cat': return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05, auto_class_weights='Balanced',
                                                random_seed=0, verbose=0, thread_count=4)
    return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         scale_pos_weight=20, eval_metric='logloss', random_state=0, n_jobs=4)
def ev(X, sp, m):
    aps = []
    for tr, te in sp:
        p = mk(m).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        if (y[te] == 1).any(): aps.append(average_precision_score(y[te], p))
    return np.mean(aps), np.std(aps)

print(f"{'Feature set':<20}{'Dim':>4}{'Cat fwd':>16}{'Cat grp':>9}{'XGB fwd':>9}")
rows = []
for name, cs in COMBOS.items():
    ci = cols_of(cs) if cs and isinstance(cs[0], str) else cs
    X = np.column_stack([Xb] + [Xs[:, c] for c in ci]) if ci else Xb
    cf, csd = ev(X, fwd, 'Cat'); cg, _ = ev(X, grp5, 'Cat'); xf, _ = ev(X, fwd, 'XGB')
    rows.append(dict(combo=name, dim=X.shape[1], cat_forward=cf, cat_forward_sd=csd, cat_group5=cg, xgb_forward=xf))
    print(f"{name:<20}{X.shape[1]:>4}{cf:>10.3f}±{csd:.3f}{cg:>9.3f}{xf:>9.3f}", flush=True)
pd.DataFrame(rows).to_csv('results/results_combo.csv', index=False)
print("\nsaved results_combo.csv")

import os, sys
