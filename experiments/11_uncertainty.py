"""Calibration, split conformal prediction, and paired bootstrap comparisons."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json, warnings
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier
from tabpfn import TabPFNClassifier
from src.features import build

X, names, y, g = build()
rng = np.random.RandomState(0)

def make(name):
    if name == 'LogReg': return make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=3000, C=0.5))
    if name == 'XGB':    return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, scale_pos_weight=20, eval_metric='logloss', random_state=0, n_jobs=4)
    if name == 'RF':     return RandomForestClassifier(n_estimators=600, class_weight='balanced_subsample', min_samples_leaf=2, random_state=0, n_jobs=4)
    if name == 'TabPFN': return TabPFNClassifier(device='cpu', n_estimators=4, random_state=0)

def ece(yy, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() > 0: e += m.mean() * abs(yy[m].mean() - p[m].mean())
    return e

layers = np.unique(g)
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(X, y, g))

# ---------- 1) 前推Protocol下的 OOF 预测（用于校准与显著性） ----------
fwd = [(np.where(g <= k)[0], np.where((g > k) & (g <= k + 10))[0]) for k in (20, 30, 40, 50)]
MODELS = ['LogReg', 'RF', 'XGB', 'TabPFN']
oof = {m: np.full(len(y), np.nan) for m in MODELS}
for m in MODELS:
    for tr, te in fwd:
        oof[m][te] = clone(make(m)).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    print(f"{m} forward predictions done", flush=True)

mask = ~np.isnan(oof['XGB'])
print("\n=== Forward chaining: calibration quality ===")
print(f"{'Model':<9}{'PR-AUC':>9}{'ROC-AUC':>9}{'Brier':>9}{'ECE':>8}")
cal = {}
for m in MODELS:
    p = oof[m][mask]
    cal[m] = dict(pr=float(average_precision_score(y[mask], p)), roc=float(roc_auc_score(y[mask], p)),
                  brier=float(brier_score_loss(y[mask], p)), ece=float(ece(y[mask], p)))
    c = cal[m]
    print(f"{m:<9}{c['pr']:>9.3f}{c['roc']:>9.3f}{c['brier']:>9.3f}{c['ece']:>8.3f}")

# ---------- 2) Split Conformal（按层分组校准） ----------
print("\n=== Split Conformal Coverage（target 1-alpha）===")
print(f"{'Model':<9}{'alpha':>7}{'Coverage':>9}{'Avg set size':>11}")
conf_rows = []
for m in ['XGB', 'TabPFN']:
    for alpha in (0.10, 0.05):
        covs, sizes = [], []
        for tr, te in fwd:
            tr_l = np.unique(g[tr]); rng2 = np.random.RandomState(0); rng2.shuffle(tr_l)
            n_cal = max(4, int(0.3 * len(tr_l)))
            cal_l = tr_l[:n_cal]; fit_l = tr_l[n_cal:]
            fit_idx = tr[np.isin(g[tr], fit_l)]; cal_idx = tr[np.isin(g[tr], cal_l)]
            if (y[fit_idx] == 1).sum() < 2 or (y[cal_idx] == 1).sum() < 1: continue
            mm = clone(make(m)).fit(X[fit_idx], y[fit_idx])
            pc = mm.predict_proba(X[cal_idx])
            # 非一致性分数 s_i = 1 - p(y_i|x_i)
            s_cal = 1 - pc[np.arange(len(cal_idx)), y[cal_idx]]
            k_q = min(1.0, np.ceil((len(cal_idx) + 1) * (1 - alpha)) / len(cal_idx))
            q = np.quantile(s_cal, k_q, method='higher')
            pt = mm.predict_proba(X[te])
            # 标准 split conformal：预测集 = {k : 1 - p_k <= q}
            sets = np.stack([(1 - pt[:, k]) <= q for k in (0, 1)], axis=1)
            covs.append(sets[np.arange(len(te)), y[te]].mean()); sizes.append(sets.sum(1).mean())
        if covs:
            conf_rows.append(dict(model=m, alpha=alpha, coverage=float(np.mean(covs)), set_size=float(np.mean(sizes))))
            print(f"{m:<9}{alpha:>7.2f}{np.mean(covs):>9.3f}{np.mean(sizes):>11.2f}")

# ---------- 3) 配对 bootstrap 显著性 ----------
print("\n=== Paired bootstrap (PR-AUC difference, 1000 reps)===")
def paired_boot(a, b, yt, n=1000):
    idx = np.where(yt == 1)[0]; idx0 = np.where(yt == 0)[0]
    d = np.empty(n)
    for i in range(n):
        s = np.concatenate([rng.choice(idx, len(idx)), rng.choice(idx0, len(idx0))])
        d[i] = average_precision_score(yt[s], a[s]) - average_precision_score(yt[s], b[s])
    return d

yy = y[mask]
pairs = [('TabPFN', 'XGB'), ('TabPFN', 'RF'), ('XGB', 'RF'), ('XGB', 'LogReg')]
sig_rows = []
for A, B in pairs:
    d = paired_boot(oof[A][mask], oof[B][mask], yy)
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    sig_rows.append(dict(A=A, B=B, delta=float(d.mean()), ci_lo=float(np.percentile(d, 2.5)),
                         ci_hi=float(np.percentile(d, 97.5)), p=float(p)))
    print(f"{A:>7} - {B:<7} ΔPR={d.mean():+.3f}  95%CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}]  p={p:.3f}")

json.dump({'calibration': cal, 'conformal': conf_rows, 'bootstrap': sig_rows},
          open('results/results_uq.json', 'w'), ensure_ascii=False, indent=1)
print("\nsaved results_uq.json")
