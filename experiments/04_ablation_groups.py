"""Group-wise ablation: which families of spatial descriptors actually help."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build()
Xs, ns, _, _ = build_spatial()
sp_idx = {n: i for i, n in enumerate(ns) if i >= len(nb)}

def group(name):
    keys = list(sp_idx)
    if name == 'entropy':        return [k for k in keys if 'hist_entropy' in k]
    if name == 'gradient':      return [k for k in keys if k.startswith('grad')]
    if name == 'topology':   return [k for k in keys if 'nreg' in k or 'bigfrac' in k or 'circ' in k]
    if name == 'moran':    return [k for k in keys if k in ('moran', 'peak_centroid_dist', 'cc')]
    if name == 'area':     return [k for k in keys if '_area' in k and not k.startswith('grad')]
    return []

fwd = [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in (20, 30, 40, 50)]
grp5 = list(__import__('sklearn.model_selection', fromlist=['GroupKFold']).GroupKFold(n_splits=5).split(Xb, y, g))

def evaluate(cols, splits):
    X = np.column_stack([Xb] + [Xs[:, sp_idx[c]][:, None] for c in cols]) if cols else Xb
    aps = []
    for tr, te in splits:
        m = CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05, auto_class_weights='Balanced',
                               random_seed=0, verbose=0, thread_count=4).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        if (y[te] == 1).any(): aps.append(average_precision_score(y[te], p))
    return np.mean(aps)

print(f"{'Group':<16}{'Dim':>5}{'Forward PR-AUC':>13}{'Grouped PR-AUC':>13}")
base_f, base_g = evaluate([], fwd), evaluate([], grp5)
print(f"{'(base, 18-d)':<16}{0:>5}{base_f:>13.3f}{base_g:>13.3f}")
rows = []
for gname in ['entropy', 'gradient', 'topology', 'moran', 'area']:
    cols = group(gname)
    if not cols: continue
    af, ag = evaluate(cols, fwd), evaluate(cols, grp5)
    rows.append(dict(group=gname, n=len(cols), forward=af, group5=ag))
    print(f"{gname:<16}{len(cols):>5}{af:>13.3f}{ag:>13.3f}   dFwd={af-base_f:+.3f}  dGrp={ag-base_g:+.3f}", flush=True)
pd.DataFrame(rows).to_csv('results/results_ablation.csv', index=False)
print("\nsaved results_ablation.csv")

import os, sys
