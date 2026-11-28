"""Selection-free replication: choose three features using only the random split,
then evaluate them under forward chaining.

The published descriptors were chosen after the forward-chained splits were
inspected. Here the selection sees only i.i.d. data, so the forward estimate
is not contaminated by that choice.
"""
import os, sys, warnings
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from src.ptr import Xb, y, g, BOUNDS, mk, base_scores
from src.features_spatial import build_spatial

Xs, ns, _, _ = build_spatial()
S = Xs[:, 18:]
names = [n for n in ns[18:]]
print('spatial features:', len(names))

# --- select on the random split only (XGBoost, 10 seeds, SMOTE)
rows = []
for j, nm in enumerate(names):
    ds = []
    for s in range(10):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=s)
        a0, yy = SMOTE(random_state=s, k_neighbors=5).fit_resample(Xb[tr], y[tr])
        a1, yy1 = SMOTE(random_state=s, k_neighbors=5).fit_resample(
            np.column_stack([Xb, S[:, [j]]])[tr], y[tr])
        p0 = mk('XGB', s).fit(a0, yy).predict_proba(Xb[te])[:, 1]
        p1 = mk('XGB', s).fit(a1, yy1).predict_proba(
            np.column_stack([Xb, S[:, [j]]])[te])[:, 1]
        ds.append(average_precision_score(y[te], p1) - average_precision_score(y[te], p0))
    rows.append(dict(feature=nm, d_rand=float(np.mean(ds)), sd=float(np.std(ds, ddof=1))))
sel = pd.DataFrame(rows).sort_values('d_rand', ascending=False)
print(sel.head(8).round(4).to_string(index=False))
sel.to_csv('results/results_selection_random.csv', index=False)

top3 = sel.head(3).feature.tolist()
idx = [names.index(t) for t in top3]
blk = S[:, idx].astype(np.float32)
print('selected by random split only:', top3)

# --- evaluate under forward chaining on the aligned population
mask = g >= 12
out = []
for model in ('Cat', 'XGB'):
    _, bf = base_scores(model, mask)
    per_seed = []
    for s in range(3):
        per = []
        for k in BOUNDS:
            tr = np.where(g[mask] <= k)[0]; te = np.where((g[mask] > k) & (g[mask] <= k + 10))[0]
            if (y[mask][te] == 1).sum() < 1: continue
            X1 = np.column_stack([Xb[mask], blk[mask]]).astype(np.float32)
            p = mk(model, s).fit(X1[tr], y[mask][tr]).predict_proba(X1[te])[:, 1]
            per.append(average_precision_score(y[mask][te], p))
        per_seed.append(np.mean(per))
    ef = np.array(per_seed)
    out.append(dict(model=model, base_fwd=round(float(bf.mean()), 3),
                    ext_fwd=round(float(ef.mean()), 3), delta=round(float((ef - bf).mean()), 3)))
    print(out[-1], flush=True)
pd.DataFrame(out).to_csv('results/results_selection_forward.csv', index=False)
print('selected:', top3)
