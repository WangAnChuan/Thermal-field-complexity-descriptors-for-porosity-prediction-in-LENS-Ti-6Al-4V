"""Protocol transfer ratio (PTR): how much of a feature's gain survives extrapolation.

PTR = Δ_forward / Δ_random，仅当分母通过 t≥2 的可分辨性检验时报告。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from src.ptr import (BLOCKS, base_scores, ext_scores, summarize, y, g)

ALL = np.ones(len(y), bool)
BULK = g >= 12

rows = []
print(f"{'设置':<10}{'模型':<5}{'特征块':<11}{'Δ随机':>9}{'t':>6}{'Δ前推':>9}"
      f"{'PTR':>7}{'95%CI':>16}{'PTR_rel':>9}")
for mask_name, mask in (('全部层', ALL), ('仅≥12层', BULK)):
    for model in ('Cat', 'XGB'):
        br, bf = base_scores(model, mask)
        for bname, fn in BLOCKS.items():
            er, ef = ext_scores(model, mask, fn(0))
            s = summarize(br, bf, er, ef)
            ci = f"[{s['ptr_ci_lo']:5.2f},{s['ptr_ci_hi']:5.2f}]" if s['defined'] else "  未定义  "
            ptr = f"{s['ptr']:7.2f}" if s['defined'] else "    n/a"
            prl = f"{s['ptr_rel']:9.2f}" if s['defined'] else "      n/a"
            print(f"{mask_name:<10}{model:<5}{bname:<11}{s['d_rand']:>+9.4f}{s['t_rand']:>6.1f}"
                  f"{s['d_fwd']:>+9.4f}{ptr}{ci:>16}{prl}", flush=True)
            rows.append(dict(mask=mask_name, model=model, block=bname, **s))

pd.DataFrame(rows).to_csv('results_ptr.csv', index=False)
print('\nsaved results_ptr.csv')
