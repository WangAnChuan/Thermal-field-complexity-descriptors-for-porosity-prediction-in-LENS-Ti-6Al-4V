"""Effect of evaluation protocol (random, grouped, forward) over ten seeds."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys, warnings
os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', '1')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from src.features import build

X, names, y, g = build()
def mk(name):
    if name == 'XGB': return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                                           colsample_bytree=0.8, scale_pos_weight=20, eval_metric='logloss', random_state=0, n_jobs=4)
    return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05, auto_class_weights='Balanced',
                              random_seed=0, verbose=0, thread_count=4)

def splits_for(pname, seed):
    if pname == 'random':  return list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y))
    if pname == 'group':   return list(GroupKFold(n_splits=5).split(X, y, g))
    return [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in (20, 30, 40, 50)]

rows = []
for pname, plabel in (('random', 'random_5fold'), ('group', 'grouped_5fold'), ('forward', 'forward_chaining')):
    seeds = range(10) if pname == 'random' else [0]
    for m in ('XGB', 'Cat'):
        aps, accs, aucs = [], [], []
        for sd in seeds:
            for tr, te in splits_for(pname, sd):
                mm = clone(mk(m)).fit(X[tr], y[tr]); p = mm.predict_proba(X[te])[:, 1]
                if (y[te] == 1).any():
                    aps.append(average_precision_score(y[te], p)); aucs.append(roc_auc_score(y[te], p))
                    accs.append(((p >= 0.5).astype(int) == y[te]).mean())
        rows.append(dict(protocol=plabel, model=m, pr=np.mean(aps), pr_sd=np.std(aps),
                         roc=np.mean(aucs), acc=np.mean(accs), n_runs=len(aps)))
        r = rows[-1]
        print(f"{plabel:<14}{m:<5} PR-AUC={r['pr']:.3f}±{r['pr_sd']:.3f}  ROC={r['roc']:.3f}  Acc={r['acc']:.3f}  (n={r['n_runs']})")
pd.DataFrame(rows).to_csv('results/results_leakage.csv', index=False)
print("\nsaved results_leakage.csv")

import os, sys
