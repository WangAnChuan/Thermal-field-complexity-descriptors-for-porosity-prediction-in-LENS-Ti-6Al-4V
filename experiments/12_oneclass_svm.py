"""One-class SVM trained on pore-free frames only, versus supervised XGBoost."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys

import numpy as np, pandas as pd, json, time, warnings, itertools
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from xgboost import XGBClassifier
from src.features import build

X, feat_names, y, g = build()
rng = np.random.RandomState(0)

def pr_auc(yy, s):  return average_precision_score(yy, s)
def rec_at_fpr(yy, s, fpr=0.05):
    """在正常样本假阳率 <= fpr 的阈值下，正样本的召回率"""
    sk = s[yy == 0]; thr = np.quantile(sk, 1 - fpr)
    return (s[yy == 1] >= thr).mean()
def prec_at_fpr(yy, s, fpr=0.05):
    sk = s[yy == 0]; thr = np.quantile(sk, 1 - fpr)
    pred = s >= thr
    return (yy[pred] == 1).mean() if pred.any() else 0.0

# ---------------- XGBoost ----------------
XGB_GRID = [dict(max_depth=d, learning_rate=lr, n_estimators=n)
            for d in (3, 4) for lr in (0.05, 0.1) for n in (300, 600)]

def fit_xgb(Xtr, ytr, gtr, params):
    p = dict(subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
             scale_pos_weight=(ytr == 0).sum() / max((ytr == 1).sum(), 1),
             random_state=0, n_jobs=4, **params)
    return XGBClassifier(**p).fit(Xtr, ytr)

# ---------------- One-Class SVM ----------------
OCSVM_GRID = [dict(nu=nu, gamma=gm) for nu in (0.01, 0.03, 0.05, 0.1) for gm in ('scale', 0.02, 0.05, 0.1)]

def fit_ocsvm(Xtr, ytr, gtr, params):
    """只用正常帧训练；Scaler 也只在正常帧上拟合。返回打分函数（越高越异常）。"""
    Xn = Xtr[ytr == 0]
    sc = StandardScaler().fit(Xn)
    m = OneClassSVM(kernel='rbf', **params).fit(sc.transform(Xn))
    return lambda Xq: -m.decision_function(sc.transform(Xq))

def inner_search_ocsvm(Xtr, ytr, gtr, grid):
    inner = GroupKFold(n_splits=3); best, best_s = None, -np.inf
    for params in grid:
        ss = []
        for itr, iva in inner.split(Xtr, ytr, gtr):
            if (ytr[itr] == 0).sum() < 10: continue
            f = fit_ocsvm(Xtr[itr], ytr[itr], gtr[itr], params)
            ss.append(pr_auc(ytr[iva], f(Xtr[iva])))
        if ss and np.mean(ss) > best_s: best, best_s = params, np.mean(ss)
    return best

gkf = GroupKFold(n_splits=5)
res = {'xgb': [], 'ocsvm': []}
oof = {'xgb': np.zeros(len(y)), 'ocsvm': np.zeros(len(y))}
t0 = time.time()

for k, (tr, te) in enumerate(gkf.split(X, y, g)):
    # XGBoost
    # 内层按层分组网格搜索
    inner = GroupKFold(n_splits=3); bs, best_s = None, -np.inf
    for params in XGB_GRID:
        ss = []
        for itr, iva in inner.split(X[tr], y[tr], g[tr]):
            m = fit_xgb(X[tr][itr], y[tr][itr], g[tr][itr], params)
            ss.append(pr_auc(y[tr][iva], m.predict_proba(X[tr][iva])[:, 1]))
        if np.mean(ss) > best_s: bs, best_s = params, np.mean(ss)
    mx = fit_xgb(X[tr], y[tr], g[tr], bs)
    p = mx.predict_proba(X[te])[:, 1]; oof['xgb'][te] = p

    # OCSVM
    bo = inner_search_ocsvm(X[tr], y[tr], g[tr], OCSVM_GRID)
    fo = fit_ocsvm(X[tr], y[tr], g[tr], bo)
    s = fo(X[te]); oof['ocsvm'][te] = s

    res['xgb'].append(dict(fold=k+1, params=bs, pr_auc=pr_auc(y[te], p), roc_auc=roc_auc_score(y[te], p),
                           rec_fpr5=rec_at_fpr(y[te], p, .05), prec_fpr5=prec_at_fpr(y[te], p, .05)))
    res['ocsvm'].append(dict(fold=k+1, params=bo, pr_auc=pr_auc(y[te], s), roc_auc=roc_auc_score(y[te], s),
                             rec_fpr5=rec_at_fpr(y[te], s, .05), prec_fpr5=prec_at_fpr(y[te], s, .05)))
    print(f"fold {k+1}: XGB PR={res['xgb'][-1]['pr_auc']:.3f} | OCSVM PR={res['ocsvm'][-1]['pr_auc']:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

print("\n" + "="*78)
for name, key in (('XGBoost（有监督）', 'xgb'), ('One-Class SVM（半监督）', 'ocsvm')):
    r = res[key]
    print(f"{name}")
    for m in ('pr_auc', 'roc_auc', 'rec_fpr5', 'prec_fpr5'):
        v = np.array([x[m] for x in r])
        print(f"   {m:<10} mean {v.mean():.3f} ± {v.std():.3f}   per fold {np.round(v,3).tolist()}")
    print()

np.save('data/cache/oof_xgb.npy', oof['xgb']); np.save('data/cache/oof_ocsvm.npy', oof['ocsvm'])
json.dump({'xgb': res['xgb'], 'ocsvm': res['ocsvm']}, open('results/results_xgb_ocsvm.json', 'w'), ensure_ascii=False, indent=1)
print("Selected hyperparameters per fold:")
for key in ('xgb', 'ocsvm'):
    print(f"  {key}: {[x['params'] for x in res[key]]}")
