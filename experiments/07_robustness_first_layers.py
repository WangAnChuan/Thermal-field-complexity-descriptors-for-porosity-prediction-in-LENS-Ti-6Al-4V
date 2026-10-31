"""Robustness check after removing layers 1-2 (substrate-interface lack of fusion)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y_all, g_all = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
extra = ['hist_entropy', 'grad_mean', 'grad_p99']
Xc = np.column_stack([Xb] + [Xs[:, idx[c]] for c in extra])

def mk(name):
    if name == 'XGB': return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, scale_pos_weight=20, eval_metric='logloss', random_state=0, n_jobs=4)
    if name == 'Cat': return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05, auto_class_weights='Balanced',
                                                random_seed=0, verbose=0, thread_count=4)
    return RandomForestClassifier(n_estimators=600, class_weight='balanced_subsample', min_samples_leaf=2, random_state=0, n_jobs=4)

def ev(X, y, g, protocol):
    if protocol.startswith('group'):
        sp = list(GroupKFold(n_splits=5).split(X, y, g))
    else:
        ks = [25, 35, 45]
        sp = [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in ks]
    aps, aucs = [], []
    for tr, te in sp:
        if (y[tr] == 1).sum() < 2 or (y[te] == 1).sum() < 1: continue
        p = mk(MODEL).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        aps.append(average_precision_score(y[te], p)); aucs.append(roc_auc_score(y[te], p))
    return np.mean(aps), np.std(aps), np.mean(aucs)

print("Excluding layers 1-2 (58 layers / 1513 frames / 29 positives)")
print(f"{'Model':<5}{'Feature':<10}{'Protocol':<8}{'PR-AUC':>16}{'ROC':>8}")
rows = []
for MODEL in ('XGB', 'Cat', 'RF'):
    for fname, X in (('base', Xb), ('base+entropy+gradient', Xc)):
        for proto in ('group5', 'forward'):
            m0 = g_all >= 3
            pr, sd, auc = ev(X[m0], y_all[m0], g_all[m0], proto)
            rows.append(dict(model=MODEL, feat=fname, protocol=proto, pr=pr, sd=sd, roc=auc))
            print(f"{MODEL:<5}{fname:<10}{proto:<8}{pr:>10.3f}±{sd:.3f}{auc:>8.3f}")
pd.DataFrame(rows).to_csv('results/results_exclude_firstlayers.csv', index=False)
print("\nsaved")
