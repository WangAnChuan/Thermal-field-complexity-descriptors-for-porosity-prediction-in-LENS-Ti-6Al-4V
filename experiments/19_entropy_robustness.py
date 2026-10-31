"""Descriptor robustness: is entropy redundant with conventional statistics, and does it depend on the histogram settings?

L2 熵是否只是标准差或热区面积的替身？
L3 直方图参数改变后结论是否稳定？
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, skew, kurtosis
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
img = np.load('data/cache/thermal.npy')

def entropy(a, bins=32, lo=0, hi=2000):
    h, _ = np.histogram(a, bins=bins, range=(lo, hi)); p = h / h.sum()
    return -(p[p > 0] * np.log(p[p > 0])).sum()

Z = (img - img.mean((1, 2), keepdims=True)) / (img.std((1, 2), keepdims=True) + 1e-9)
grad_mean = (np.abs(np.diff(Z, axis=1)).mean((1, 2)) + np.abs(np.diff(Z, axis=2)).mean((1, 2)))
grad_p99 = np.percentile(np.concatenate([np.abs(np.diff(Z, axis=1)).reshape(len(Z), -1),
                                         np.abs(np.diff(Z, axis=2)).reshape(len(Z), -1)], 1), 99, axis=1)

std = img.std((1, 2)); a15 = (img > 1500).mean((1, 2)); a12 = (img > 1200).mean((1, 2))
H = np.array([entropy(img[i]) for i in range(len(img))])

def mk(model, seed=0):
    if model == 'XGB':
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=20, eval_metric='logloss',
                             random_state=seed, n_jobs=4)
    return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05,
                              auto_class_weights='Balanced', random_seed=seed, verbose=0, thread_count=4)

BOUNDS = [15, 20, 25, 30, 35, 40, 45]
SPLITS = [(k, np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in BOUNDS]

def gain(model, extra, base=Xb):
    X0 = base; X1 = np.column_stack([base, extra])
    out = []
    for seed in range(3):
        for k, tr, te in SPLITS:
            a = average_precision_score(y[te], mk(model, seed).fit(X0[tr], y[tr]).predict_proba(X0[te])[:, 1])
            b = average_precision_score(y[te], mk(model, seed).fit(X1[tr], y[tr]).predict_proba(X1[te])[:, 1])
            out.append(b - a)
    return float(np.mean(out)), float(np.std(out))

print("=== L2a 熵与既有统计量的相关性（Spearman）===")
for name, v in (('std', std), ('area>1500', a15), ('area>1200', a12), ('grad_mean', grad_mean), ('Tmax', img.max((1, 2)))):
    r, p = spearmanr(H, v)
    print(f"  entropy vs {name:<10} rho={r:+.3f}  p={p:.1e}")

print("\n=== L2b 熵在已有 std+面积 之上是否还有增量 ===")
rows = []
for model in ('Cat', 'XGB'):
    base_plus = np.column_stack([Xb, std, a12, a15])
    g1, s1 = gain(model, H[:, None], base=Xb)
    g2, s2 = gain(model, np.column_stack([std, a12, a15]), base=Xb)
    g3, s3 = gain(model, H[:, None], base=base_plus)
    print(f"{model}: +entropy {g1:+.4f}±{s1:.4f} | +std/area {g2:+.4f}±{s2:.4f} | +entropy given std/area {g3:+.4f}±{s3:.4f}")
    rows.append(dict(analysis='L2', model=model, extra='entropy', gain=g1, sd=s1))
    rows.append(dict(analysis='L2', model=model, extra='std+area', gain=g2, sd=s2))
    rows.append(dict(analysis='L2', model=model, extra='entropy|std+area', gain=g3, sd=s3))

print("\n=== L3 直方图参数敏感性（CatBoost，三类合一共 3 维）===")
for bins, lo, hi in ((16, 0, 2000), (32, 0, 2000), (64, 0, 2000), (32, 0, 2500), (32, 500, 2000)):
    Hs = np.array([entropy(img[i], bins, lo, hi) for i in range(len(img))])
    gH, sH = gain('Cat', Hs[:, None])
    gAll, sAll = gain('Cat', np.column_stack([Hs, grad_mean, grad_p99]))
    note = '  ← 论文设置' if (bins, lo, hi) == (32, 0, 2000) else ''
    print(f"  bins={bins:>3} range=[{lo},{hi}]  +entropy {gH:+.4f}  +all three {gAll:+.4f}±{sAll:.4f}{note}")
    rows.append(dict(analysis='L3', model='Cat', extra=f'bins{bins}_range{lo}-{hi}', gain=gAll, sd=sAll))

pd.DataFrame(rows).to_csv('results/results_entropy_robustness.csv', index=False)
print("\nsaved results_p1_entropy.csv")
