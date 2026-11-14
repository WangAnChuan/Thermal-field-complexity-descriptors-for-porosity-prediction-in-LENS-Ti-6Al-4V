"""Pool geometry from solidification isotherms: does the published physics feature family transfer?

物理启发的几何量（液相线/固相线等温线）与三个热场描述符的增量对比，以及冗余度诊断。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from src.features import build
from src.features_spatial import build_spatial
from src.features_geometry import geometry_block

Xb, nb, y, g = build()
Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
EG = Xs[:, [idx['hist_entropy'], idx['grad_mean'], idx['grad_p99']]]
GEO, gnames = geometry_block()      # 无液相区的帧以 -1 哨兵填充

CORE = [gnames.index(c) for c in ('geo_ar', 'geo_circ', 'geo_mushy', 'geo_G_mushy')]

SETS = {
    'base': Xb,
    'geo': np.column_stack([Xb, GEO]).astype(np.float32),
    'ours': np.column_stack([Xb, EG]).astype(np.float32),
    'geo+ours': np.column_stack([Xb, GEO, EG]).astype(np.float32),
    'geo_core': np.column_stack([Xb, GEO[:, CORE]]).astype(np.float32),
}

def mk(name, seed):
    if name == 'XGB':
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=20, eval_metric='logloss',
                             random_state=seed, n_jobs=4)
    return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05,
                              auto_class_weights='Balanced', random_seed=seed,
                              verbose=0, thread_count=4)

BOUNDS = [15, 20, 25, 30, 35, 40, 45]
N_RAND, N_GRP, N_FWD = 10, 3, 3

def cv_random(model, X, seed):
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed)
    a, yy = SMOTE(random_state=seed, k_neighbors=5).fit_resample(X[tr], y[tr])
    p = mk(model, seed).fit(a, yy).predict_proba(X[te])[:, 1]
    return average_precision_score(y[te], p), roc_auc_score(y[te], p)

def cv_grouped(model, X, seed):
    a = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        p = mk(model, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        a.append(average_precision_score(y[te], p))
    return float(np.mean(a))

def cv_forward(model, X, seed):
    a = []
    for k in BOUNDS:
        tr = np.where(g <= k)[0]; te = np.where((g > k) & (g <= k + 10))[0]
        if (y[te] == 1).sum() < 1: continue
        p = mk(model, seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        a.append(average_precision_score(y[te], p))
    return float(np.mean(a))

rows = []
print(f"{'模型':<5}{'特征集':<10}{'随机':>16}{'分组':>16}{'前推':>16}")
for model in ('Cat', 'XGB'):
    stats = {}
    for sname, X in SETS.items():
        r = [cv_random(model, X, s)[0] for s in range(N_RAND)]
        gp = [cv_grouped(model, X, s) for s in range(N_GRP)]
        fw = [cv_forward(model, X, s) for s in range(N_FWD)]
        stats[sname] = (np.mean(r), np.mean(gp), np.mean(fw),
                        np.std(r, ddof=1), np.std(fw, ddof=1))
        m = stats[sname]
        print(f"{model:<5}{sname:<10}{m[0]:>8.4f}±{m[3]:.3f}{m[1]:>16.4f}{m[2]:>8.4f}±{m[4]:.3f}",
              flush=True)
        rows.append(dict(model=model, feats=sname, rand=m[0], rand_sd=m[3],
                         grouped=m[1], forward=m[2], forward_sd=m[4]))
    b = stats['base']
    print(f"  Δ相对 base：geo {stats['geo'][2]-b[2]:+.4f}  ours {stats['ours'][2]-b[2]:+.4f}  "
          f"geo+ours {stats['geo+ours'][2]-b[2]:+.4f}  (前推)", flush=True)

out = pd.DataFrame(rows)
out.to_csv('results_geometry_eval.csv', index=False)

# ---- 几何特征与已有量的相关性 ----
print("\n=== 几何特征与基线/描述符的相关（|Spearman|，Top 3）===")
from scipy.stats import spearmanr
allf = pd.DataFrame(np.column_stack([Xb, GEO, EG]),
                    columns=nb + gnames + ['hist_entropy', 'grad_mean', 'grad_p99'])
base_names = nb + ['hist_entropy', 'grad_mean', 'grad_p99']
cor = allf[gnames + base_names].corr(method='spearman').loc[gnames, base_names]
for c in gnames:
    top = cor.loc[c].abs().sort_values(ascending=False).head(3)
    print(f"  {c:<14}" + "  ".join(f"{k}={cor.loc[c, k]:+.2f}" for k in top.index))
cor.to_csv('results_geometry_corr.csv')

# ---- 判别力：单变量 AUC ----
print("\n=== 单变量 AUC（正例 71 / 1564）===")
from sklearn.metrics import roc_auc_score as auc
uni = []
for name, col in [(c, GEO[:, i]) for i, c in enumerate(gnames) if 'geo_valid' not in c] + \
                 [('hist_entropy', EG[:, 0]), ('grad_mean', EG[:, 1]), ('grad_p99', EG[:, 2]),
                  ('MP_Area', Xb[:, nb.index('MP_Area')]), ('MP_Ecc', Xb[:, nb.index('MP_Eccentricity')]),
                  ('Tmax', Xb[:, nb.index('Tmax')])]:
    m = np.ones(len(y), bool) if 'geo_' in name else ~np.isnan(col)
    a = auc(y[m], col[m])
    uni.append(dict(feature=name, auc=a, n=int(m.sum())))
    print(f"  {name:<14}{a:.3f}")
pd.DataFrame(uni).to_csv('results_geometry_univariate.csv', index=False)

print("\nsaved results_geometry_eval.csv / _corr.csv / _univariate.csv")
