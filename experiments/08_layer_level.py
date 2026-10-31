"""Layer-level evaluation: aggregate frame probabilities to the layer decision unit."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
Xc = np.column_stack([Xb] + [Xs[:, idx[c]] for c in ('hist_entropy','grad_mean','grad_p99')])
layers = np.unique(g)
y_layer = np.array([int(y[g == L].max()) for L in layers])   # 层内有任一孔即为正
print(f"Layers {len(layers)}，positive layers {y_layer.sum()} ({y_layer.mean():.1%})")

def mk(name):
    if name == 'XGB': return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, scale_pos_weight=1.3, eval_metric='logloss', random_state=0, n_jobs=4)
    return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05, auto_class_weights='Balanced',
                              random_seed=0, verbose=0, thread_count=4)

def agg(probs_by_layer, how='max'):
    return np.array([p.max() if how == 'max' else p.mean() for p in probs_by_layer])

print(f"\n{'Model':<5}{'Agg':<5}{'Feature':<12}{'Protocol':<8}{'Layer PR-AUC':>12}{'Layer ROC':>9}")
rows = []
for name in ('XGB', 'Cat'):
    for fname, X in (('base', Xb), ('entropy+gradient', Xc)):
        for proto in ('group5', 'forward'):
            if proto == 'group5':
                sp = [(np.where(np.isin(g, [layers[i] for i in tr]))[0],
                       np.where(np.isin(g, [layers[i] for i in te]))[0])
                      for tr, te in GroupKFold(n_splits=5).split(layers, y_layer, layers)]
            else:
                sp = [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in (25, 35, 45)]
            for how in ('max', 'mean'):
                aps, aucs = [], []
                for tr, te in sp:
                    if (y[tr] == 1).sum() < 2 or (y[te] == 1).sum() < 1: continue
                    m = mk(name).fit(X[tr], y[tr])
                    p = m.predict_proba(X[te])[:, 1]
                    tl = np.unique(g[te])
                    pl = agg([p[g[te] == L] for L in tl], how)
                    yl = np.array([int(y[g == L].max()) for L in tl])
                    if yl.sum() < 1: continue
                    aps.append(average_precision_score(yl, pl)); aucs.append(roc_auc_score(yl, pl))
                if aps:
                    rows.append(dict(model=name, agg=how, feat=fname, protocol=proto,
                                     pr=np.mean(aps), roc=np.mean(aucs)))
                    r = rows[-1]
                    print(f"{name:<5}{how:<5}{fname:<12}{proto:<8}{np.mean(aps):>12.3f}{np.mean(aucs):>9.3f}", flush=True)
pd.DataFrame(rows).to_csv('results/results_layerlevel.csv', index=False)
print("\nsaved results_layerlevel.csv")

import os, sys
