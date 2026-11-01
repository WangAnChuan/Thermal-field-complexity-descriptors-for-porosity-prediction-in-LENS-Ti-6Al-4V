"""Pore-size analysis: regression on the positive class and size-stratified detection.

问题：描述符预测的是"有无孔隙"，还是也与孔隙严重程度相关？
     对小孔（<0.3 mm）是否同样有效？
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBClassifier, XGBRegressor
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
df = pd.read_parquet('data/cache/labels.parquet')
size = df['Porosity Size (mm)'].values
idx = {n: i for i, n in enumerate(ns)}
EG = Xs[:, [idx['hist_entropy'], idx['grad_mean'], idx['grad_p99']]]
X0, X1 = Xb, np.column_stack([Xb, EG])

BOUNDS = [15, 20, 25, 30, 35, 40, 45]
def splits():
    return [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in BOUNDS]

rows = []

# ---------------- ① 孔径回归 ----------------
print("=== ① 孔径回归（仅正例，前推 7 边界）===")
print("   目标：Porosity Size (mm)，只在有孔帧上拟合")
for model in ('Cat', 'XGB'):
    for cond, X in (('base', X0), ('+EG', X1)):
        maes, r2s = [], []
        for tr, te in splits():
            ptr = tr[y[tr] == 1]; pte = te[y[te] == 1]
            if len(ptr) < 5 or len(pte) < 2: continue
            mk = (lambda: CatBoostRegressor(iterations=400, depth=3, learning_rate=0.05,
                                            random_seed=0, verbose=0, thread_count=4)) if model == 'Cat' \
                 else (lambda: XGBRegressor(n_estimators=400, max_depth=3, learning_rate=0.05,
                                            subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=4))
            m = mk().fit(X[ptr], size[ptr])
            p = m.predict(X[pte])
            maes.append(mean_absolute_error(size[pte], p)); r2s.append(r2_score(size[pte], p) if len(pte) > 2 else np.nan)
        rows.append(dict(analysis='R1', model=model, condition=cond,
                         mae=float(np.nanmean(maes)), r2=float(np.nanmean(r2s))))
        print(f"  {model:<4}{cond:<6}MAE={np.mean(maes):.3f} mm   R2={np.nanmean(r2s):+.3f}")

print("\n   参考：孔径标准差 =", round(size[y == 1].std(), 3), "mm；中位数=", round(np.median(size[y == 1]), 3))
# 描述符与孔径的直接相关（仅正例）
pos = y == 1
for name, v in (('entropy', Xs[pos][:, idx['hist_entropy']]), ('grad_mean', Xs[pos][:, idx['grad_mean']])):
    r, p = spearmanr(v, size[pos])
    print(f"   Spearman({name}, pore size) = {r:+.3f}  p={p:.3f}")

# ---------------- ② 按孔径分层 ----------------
print("\n=== ② 按孔径分层的检测性能（前推 7 边界 × 3 种子）===")
groups = {'all': (size > 0) & (y == 1),
          'small <0.3mm': (y == 1) & (size < 0.3),
          'medium 0.3-0.6': (y == 1) & (size >= 0.3) & (size <= 0.6),
          'large >0.6mm': (y == 1) & (size > 0.6)}
for gname, gmask in groups.items():
    for cond, X in (('base', X0), ('+EG', X1)):
        aps = []
        for seed in range(3):
            for tr, te in splits():
                keep = ~((y == 1) & ~gmask)          # 测试集里去掉不属于该组的正例
                te_k = te[keep[te]]
                if (y[te_k] == 1).sum() < 1: continue
                m = CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05,
                                       auto_class_weights='Balanced', random_seed=seed,
                                       verbose=0, thread_count=4).fit(X[tr], y[tr])
                p = m.predict_proba(X[te_k])[:, 1]
                aps.append(average_precision_score(y[te_k], p))
        rows.append(dict(analysis='R2', model='Cat', condition=f'{gname}|{cond}',
                         mae=float(np.mean(aps)), r2=float(np.std(aps))))
        print(f"  {gname:<16}{cond:<6}AP={np.mean(aps):.3f} ± {np.std(aps):.3f}  (n_pos={int(gmask.sum())})")

pd.DataFrame(rows).to_csv('results/results_pore_size.csv', index=False)
print("\nsaved results_pore_size.csv")
