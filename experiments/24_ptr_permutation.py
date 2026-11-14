"""Permutation null distribution for the PTR experiment."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from src.ptr import base_scores, ext_scores, summarize_parts, EG, y, g, permuted

ALL = np.ones(len(y), bool)
BULK = g >= 12

rows = []
for mask_name, mask in (('全部层', ALL), ('仅≥12层', BULK)):
    br, bf = base_scores('Cat', mask)
    er, ef = ext_scores('Cat', mask, EG)
    obs = summarize_parts(br, bf, er, ef)
    d_fwd, d_rand = [], []
    for rep in range(10):
        pr, pf = ext_scores('Cat', mask, permuted(rep))
        s = summarize_parts(br, bf, pr, pf)
        d_fwd.append(s['d_fwd']); d_rand.append(s['d_rand'])
    d_fwd, d_rand = np.array(d_fwd), np.array(d_rand)
    p_fwd = (d_fwd >= obs['d_fwd']).mean()
    ptrs = np.array([a / b for a, b in zip(d_fwd, d_rand) if b > 1e-3])
    print(f"{mask_name:<10}观测 Δ前推 {obs['d_fwd']:+.4f}  打乱 Δ前推 {d_fwd.mean():+.4f}"
          f"±{d_fwd.std():.4f}  p = {p_fwd:.3f}  (n_null={len(d_fwd)})", flush=True)
    if obs['defined']:
        ptrs_all = np.array([a / b for a, b in zip(d_fwd, d_rand) if b > 1e-3])
        print(f"{'':<10}观测 PTR {obs['ptr']:.2f}  打乱 PTR {ptrs_all.mean():.2f}±{ptrs_all.std():.2f}  "
              f"p = {(ptrs_all >= obs['ptr']).mean():.3f}", flush=True)
    rows.append(dict(mask=mask_name, obs_ptr=obs['ptr'], obs_d_fwd=obs['d_fwd'],
                     obs_d_rand=obs['d_rand'],
                     null_d_fwd_mean=d_fwd.mean(), null_d_fwd_sd=d_fwd.std(),
                     null_d_rand_mean=d_rand.mean(), null_d_rand_sd=d_rand.std(),
                     p_d_fwd=p_fwd, n_null=len(d_fwd)))
pd.DataFrame(rows).to_csv('results_ptr_permtest.csv', index=False)
print('saved results_ptr_permtest.csv')
