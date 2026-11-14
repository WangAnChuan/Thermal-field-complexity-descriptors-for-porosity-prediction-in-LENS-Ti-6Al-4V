"""PTR 指标的公共实现：数据、模型工厂、两种协议的基线/增量打分、自助法汇总。"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build()
Xs, ns, _, _ = build_spatial()
_idx = {n: i for i, n in enumerate(ns)}
EG = Xs[:, [_idx['hist_entropy'], _idx['grad_mean'], _idx['grad_p99']]]
_img = np.load('data/cache/thermal.npy')
_flat = _img.reshape(len(_img), -1)
MOM = np.column_stack([skew(_flat, axis=1), kurtosis(_flat, axis=1),
                       np.sqrt((_flat ** 2).mean(1))]).astype(np.float32)
_Z = (_flat - _flat.mean(1, keepdims=True)) / (_flat.std(1, keepdims=True) + 1e-9)


def randproj(seed=0):
    rng = np.random.RandomState(seed)
    P = _Z @ rng.randn(_Z.shape[1], 3).astype(np.float32)
    return (P / (P.std(0) + 1e-9)).astype(np.float32)


def permuted(seed):
    rng = np.random.RandomState(seed)
    P = EG.copy()
    for j in range(3):
        P[:, j] = rng.permutation(P[:, j])
    return P


BLOCKS = {
    'complexity': lambda s: EG,
    'moments': lambda s: MOM,
    'randproj': lambda s: randproj(0),
    'permuted': lambda s: permuted(0),
    'entropy': lambda s: EG[:, [0]],
    'grad': lambda s: EG[:, [1, 2]],
}


def mk(name, seed):
    if name == 'XGB':
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=20, eval_metric='logloss',
                             random_state=seed, n_jobs=4)
    if name == 'RF':
        return RandomForestClassifier(n_estimators=600, class_weight='balanced_subsample',
                                      min_samples_leaf=2, random_state=seed, n_jobs=4)
    return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05,
                              auto_class_weights='Balanced', random_seed=seed,
                              verbose=0, thread_count=4)


BOUNDS = [15, 20, 25, 30, 35, 40, 45]
N_RAND, N_FWD = 10, 3


def base_scores(model, mask):
    """基线特征在两种协议下的 AP（随机：逐分片一个；前推：逐种子一个）。"""
    X0m, ym, gm = Xb[mask], y[mask], g[mask]
    br = []
    for s in range(N_RAND):
        tr, te = train_test_split(np.arange(len(ym)), test_size=0.3, stratify=ym, random_state=s)
        a0, yy = SMOTE(random_state=s, k_neighbors=5).fit_resample(X0m[tr], ym[tr])
        p0 = mk(model, s).fit(a0, yy).predict_proba(X0m[te])[:, 1]
        br.append(average_precision_score(ym[te], p0))
    bf = []
    for s in range(N_FWD):
        per = []
        for k in BOUNDS:
            tr = np.where(gm <= k)[0]; te = np.where((gm > k) & (gm <= k + 10))[0]
            if (ym[te] == 1).sum() < 1: continue
            p0 = mk(model, s).fit(X0m[tr], ym[tr]).predict_proba(X0m[te])[:, 1]
            per.append(average_precision_score(ym[te], p0))
        bf.append(np.mean(per))
    return np.array(br), np.array(bf)


def ext_scores(model, mask, block):
    """增量特征块在两种协议下的 AP，与 base_scores 逐项配对。"""
    X0m, ym, gm = Xb[mask], y[mask], g[mask]
    X1m = np.column_stack([X0m, block[mask]]).astype(np.float32)
    er = []
    for s in range(N_RAND):
        tr, te = train_test_split(np.arange(len(ym)), test_size=0.3, stratify=ym, random_state=s)
        a1, yy = SMOTE(random_state=s, k_neighbors=5).fit_resample(X1m[tr], ym[tr])
        p1 = mk(model, s).fit(a1, yy).predict_proba(X1m[te])[:, 1]
        er.append(average_precision_score(ym[te], p1))
    ef = []
    for s in range(N_FWD):
        per = []
        for k in BOUNDS:
            tr = np.where(gm <= k)[0]; te = np.where((gm > k) & (gm <= k + 10))[0]
            if (ym[te] == 1).sum() < 1: continue
            p1 = mk(model, s).fit(X1m[tr], ym[tr]).predict_proba(X1m[te])[:, 1]
            per.append(average_precision_score(ym[te], p1))
        ef.append(np.mean(per))
    return np.array(er), np.array(ef)


def summarize_parts(br, bf, er, ef):
    dr, df = er - br, ef - bf
    t = dr.mean() / (dr.std(ddof=1) / np.sqrt(len(dr)))
    defined = bool(t >= 2.0)
    return dict(d_rand=float(dr.mean()), d_fwd=float(df.mean()),
                d_rand_sd=float(dr.std(ddof=1)), t_rand=float(t), defined=defined,
                ptr=float(df.mean() / dr.mean()) if defined else np.nan,
                ptr_rel=float((df.mean() / bf.mean()) / (dr.mean() / br.mean())) if defined else np.nan,
                base_rand=float(br.mean()), base_fwd=float(bf.mean()))


def summarize(br, bf, er, ef, n_boot=4000, seed=0):
    s = summarize_parts(br, bf, er, ef)
    dr, df = er - br, ef - bf
    rng = np.random.RandomState(seed)
    ratio, ratio_rel, valid = [], [], 0
    for _ in range(n_boot):
        a = rng.choice(dr, len(dr), replace=True).mean()
        b = rng.choice(df, len(df), replace=True).mean()
        if a > 1e-3:
            valid += 1
            ratio.append(b / a)
            ratio_rel.append((b / bf.mean()) / (a / br.mean()))
    lo, hi = (np.percentile(ratio, [2.5, 97.5]) if len(ratio) > 100 else (np.nan, np.nan))
    s.update(ptr_ci_lo=float(lo), ptr_ci_hi=float(hi),
             ptr_rel_ci_lo=float(np.percentile(ratio_rel, 2.5)) if ratio_rel else np.nan,
             ptr_rel_ci_hi=float(np.percentile(ratio_rel, 97.5)) if ratio_rel else np.nan,
             frac_valid=valid / n_boot, n_rand=len(dr), n_fwd=len(df))
    return s
