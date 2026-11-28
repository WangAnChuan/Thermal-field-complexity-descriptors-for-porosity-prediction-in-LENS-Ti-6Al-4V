"""Leave-one-positive-out: how much of the forward-chained increment rests on
a single pore frame in the test segment?

For every (seed, boundary) we fit once, then recompute AP with each positive
frame removed in turn. The spread of the increment answers whether 0.151 is
carried by a handful of events.
"""
import os, sys, warnings
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score
from src.ptr import Xb, y, g, BOUNDS, mk, EG

mask = g >= 12
rows = []
for model in ('Cat', 'XGB'):
    for s in range(3):
        for k in BOUNDS:
            tr = np.where(g[mask] <= k)[0]
            te = np.where((g[mask] > k) & (g[mask] <= k + 10))[0]
            pos = te[y[mask][te] == 1]
            if len(pos) < 2: continue
            X0 = Xb[mask]; X1 = np.column_stack([X0, EG[mask]]).astype(np.float32)
            ym = y[mask]
            p0 = mk(model, s).fit(X0[tr], ym[tr]).predict_proba(X0[te])[:, 1]
            p1 = mk(model, s).fit(X1[tr], ym[tr]).predict_proba(X1[te])[:, 1]
            full = average_precision_score(ym[te], p1) - average_precision_score(ym[te], p0)
            keep_all = []
            for j in range(len(te)):
                if ym[te][j] == 0: continue
                keep = np.ones(len(te), bool); keep[j] = False
                if ym[te][keep].sum() < 1: continue
                d = (average_precision_score(ym[te][keep], p1[keep]) -
                     average_precision_score(ym[te][keep], p0[keep]))
                keep_all.append(d)
            if keep_all:
                rows.append(dict(model=model, seed=s, k=int(k), n_pos=len(pos), full=full,
                                 loo_min=min(keep_all), loo_max=max(keep_all),
                                 loo_mean=float(np.mean(keep_all)), spread=max(keep_all) - min(keep_all)))
df = pd.DataFrame(rows)
df.to_csv('results/results_leave_one_out.csv', index=False)
print(df.groupby('model')[['full', 'loo_min', 'loo_max', 'loo_mean', 'spread']].mean().round(3).to_string())
print()
print('runs where the increment changes sign when one positive is dropped:',
      int(((df.loo_min < 0)).sum()), 'of', len(df))
print(df.groupby('model').apply(lambda d: pd.Series({
    'full_mean': d.full.mean().round(3), 'loo_min_of_min': d.loo_min.min().round(3),
    'loo_max_of_max': d.loo_max.max().round(3), 'mean_spread': d.spread.mean().round(3)})).to_string())
