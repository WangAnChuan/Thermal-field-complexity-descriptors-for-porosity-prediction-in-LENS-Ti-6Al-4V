"""Six classifiers under two evaluation protocols: layer-grouped CV and forward chaining."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, time, warnings
os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', '1')
os.environ.setdefault('TABPFN_MODEL_CACHE_DIR', os.path.expanduser('~/.cache/tabpfn'))
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from src.features import build
from src.features_temporal import build_all

Xb, nb, y, g = build()
Xt, nt, _, _ = build_all()

def make(name):
    if name == 'LogReg':  return make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=3000, C=0.5))
    if name == 'RF':      return RandomForestClassifier(n_estimators=600, class_weight='balanced_subsample', min_samples_leaf=2, random_state=0, n_jobs=4)
    if name == 'XGB':     return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                               scale_pos_weight=20, eval_metric='logloss', random_state=0, n_jobs=4)
    if name == 'LGBM':    return LGBMClassifier(n_estimators=500, num_leaves=15, learning_rate=0.05, subsample=0.8,
                                                colsample_bytree=0.8, class_weight='balanced', random_state=0, n_jobs=4, verbose=-1)
    if name == 'CatBoost':return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05, auto_class_weights='Balanced',
                                                    random_seed=0, verbose=0, thread_count=4)
    if name == 'TabPFN':  return TabPFNClassifier(device='cpu', n_estimators=4, random_state=0)
    raise ValueError(name)

MODELS = ['LogReg', 'RF', 'XGB', 'LGBM', 'CatBoost', 'TabPFN']

def rec_at(yy, s, t):
    return (s[yy == 1] >= t).mean() if (yy == 1).any() else np.nan
def thr_at_fpr(yy, s, fpr=0.05):
    return np.quantile(s[yy == 0], 1 - fpr) if (yy == 0).any() else np.nan

def evaluate(protocol):
    rows = []
    if protocol == 'group5':
        splits = list(GroupKFold(n_splits=5).split(Xb, y, g))
    else:  # 前推
        splits = [(np.where((g >= 1) & (g <= k))[0], np.where((g > k) & (g <= k + 10))[0]) for k in (20, 30, 40, 50)]
    for feat_name, X in (('base', Xb), ('temporal', Xt)):
        for m in MODELS:
            t0 = time.time(); aps, aucs, brs = [], [], []
            for tr, te in splits:
                mm = clone(make(m)).fit(X[tr], y[tr])
                p = mm.predict_proba(X[te])[:, 1]
                if (y[te] == 1).any():
                    aps.append(average_precision_score(y[te], p)); aucs.append(roc_auc_score(y[te], p))
                    brs.append(brier_score_loss(y[te], p))
            rows.append(dict(protocol=protocol, feat=feat_name, model=m,
                             pr=float(np.mean(aps)), pr_std=float(np.std(aps)), roc=float(np.mean(aucs)),
                             brier=float(np.mean(brs)), secs=round(time.time() - t0, 1)))
            print(f"[{protocol}/{feat_name}] {m:<9} PR={rows[-1]['pr']:.3f}±{rows[-1]['pr_std']:.3f} "
                  f"ROC={rows[-1]['roc']:.3f} ({rows[-1]['secs']}s)", flush=True)
    return rows

allrows = evaluate('group5') + evaluate('forward')
pd.DataFrame(allrows).to_csv('results/results_bench_suite.csv', index=False)
print("\nsaved results_bench_suite.csv")
