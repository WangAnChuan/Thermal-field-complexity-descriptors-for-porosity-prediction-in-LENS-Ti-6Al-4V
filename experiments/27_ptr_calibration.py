"""Calibration of the PTR estimator on synthetic data with known ground truth."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

RNG = np.random.RandomState(0)


def make_data(n, omega, beta0=0.30, seed=0):
    """omega = 前推测试窗（31–40 层）内该特征的真实效应相对其余层的倍数。

    ω = 1 表示效应在整段建造上恒定（无协议依赖）；
    ω > 1 表示效应沿建造线性增强，越靠后的层越强——前推测试窗集中在这一段。
    """
    rng = np.random.RandomState(seed)
    layer = rng.randint(1, 61, n)
    z = rng.randn(n)
    coef = beta0 * (1 + (layer - 1) / 59.0 * (omega - 1))
    p = 1 / (1 + np.exp(-(-1.4 + coef * z)))
    y = (rng.rand(n) < p).astype(int)
    Xb = rng.randn(n, 4)                       # 纯噪声基线
    return Xb, z.reshape(-1, 1), y, layer


def fit_ap(X, y, tr, te, seed):
    m = XGBClassifier(n_estimators=150, max_depth=3, learning_rate=0.1,
                      subsample=0.9, colsample_bytree=1.0, eval_metric='logloss',
                      tree_method='hist', random_state=seed, n_jobs=4)
    m.fit(X[tr], y[tr])
    return average_precision_score(y[te], m.predict_proba(X[te])[:, 1])


BOUNDS = (20, 30, 40, 50)


def protocol_deltas(Xb, G, y, layer, block=True, n_split=10, n_fwd=3):
    """与主论文完全相同的两种协议：随机 70/30（10 个种子）与前推
    （k∈{20,30,40,50}，训练 ≤k 层、测试 k+1..k+10 层，3 个种子）。"""
    X0 = Xb
    X1 = np.column_stack([Xb, G]) if block else Xb
    d_rand = []
    for s in range(n_split):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                                  stratify=y, random_state=s)
        d_rand.append(fit_ap(X1, y, tr, te, s) - fit_ap(X0, y, tr, te, s))
    d_fwd = []
    for s in range(n_fwd):
        per = []
        for k in BOUNDS:
            tr = np.where(layer <= k)[0]
            te = np.where((layer > k) & (layer <= k + 10))[0]
            if len(te) < 20:
                continue
            per.append(fit_ap(X1, y, tr, te, s + 100) - fit_ap(X0, y, tr, te, s + 100))
        if per:
            d_fwd.append(float(np.mean(per)))
    return np.array(d_rand), np.array(d_fwd)


def calib(n_rep=15, omegas=(1, 2, 3, 5), n=6000):
    rows = []
    for om in omegas:
        for r in range(n_rep):
            Xb, G, y, layer = make_data(n, om, seed=r * 17 + int(om))
            dr, df = protocol_deltas(Xb, G, y, layer)
            t = dr.mean() / (dr.std(ddof=1) / np.sqrt(len(dr)))
            rows.append(dict(omega=om, rep=r, d_rand=dr.mean(), d_fwd=df.mean(),
                             t_rand=t,
                             ptr=df.mean() / dr.mean() if abs(dr.mean()) > 1e-6 else np.nan,
                             defined=t >= 2))
            print(f'ω={om} rep {r:>2}  Δrand {dr.mean():+.4f}  Δfwd {df.mean():+.4f}  '
                  f't {t:5.1f}  PTR {rows[-1]["ptr"]:.2f}', flush=True)
    return pd.DataFrame(rows)


def gate_size(n_rep=100, n=6000):
    """无信息特征块：门槛应当很少通过。"""
    hits, ok = 0, 0
    for r in range(n_rep):
        Xb, G, y, layer = make_data(n, 1.0, seed=9000 + r)
        G = np.random.RandomState(r).randn(len(y), 1)    # 与标签无关
        dr, _ = protocol_deltas(Xb, G, y, layer)
        t = dr.mean() / (dr.std(ddof=1) / np.sqrt(len(dr)))
        hits += int(t >= 2)
        ok += 1
        if (r + 1) % 20 == 0:
            print(f'  门槛检验 {r+1}/{n_rep}  通过 {hits}', flush=True)
    return hits / ok


if __name__ == '__main__':
    print('=== Q1 校准 ===')
    c = calib()
    c.to_csv('results_ptr_calibration.csv', index=False)
    g = c.groupby('omega').agg(ptr_mean=('ptr', 'mean'), ptr_sd=('ptr', 'std'),
                               ptr_med=('ptr', 'median'),
                               defined=('defined', 'mean')).round(3)
    print('\n', g.to_string())

    print('\n=== Q2 门槛误报率 ===')
    rate = gate_size()
    print(f'\n无信息特征块的门槛通过率 = {rate:.3f}（名义水平 0.025）')
    pd.DataFrame([dict(rep=100, gate_rate=rate)]).to_csv(
        'results_ptr_gate_size.csv', index=False)
    print('saved results_ptr_calibration.csv / results_ptr_gate_size.csv')


def gate_permutation(n_rep=20, n_perm=10, n_split=5, n=6000):
    """置换式门槛：Δ_rand 是否超过标签置换零分布的 95 分位。

    这是对 t 检验的替代。t 检验把 10 个共享同一份数据的划分当作独立样本，
    低估了抽样误差；置换零分布则不依赖这个假设。
    """
    hits, vals = 0, []
    for r in range(n_rep):
        Xb, G, y, layer = make_data(n, 1.0, seed=7000 + r)
        G = np.random.RandomState(r).randn(len(y), 1)      # 无信息
        X1 = np.column_stack([Xb, G])
        obs = _mean_delta_rand(Xb, X1, y, n_split, base_seed=0)
        # 关键：置换与原观测必须用同一组划分，只换标签
        null = [_mean_delta_rand(Xb, X1, np.random.RandomState(100 + r * 7 + p).permutation(y),
                                 n_split, base_seed=0) for p in range(n_perm)]
        thr = np.percentile(null, 95)
        hits += int(obs > thr)
        vals.append((obs, thr))
        print(f'  rep {r:>2}  Δ_rand {obs:+.4f}  零分布 95 分位 {thr:+.4f}  '
              f'{"通过" if obs > thr else "拒绝"}', flush=True)
    return hits / n_rep, vals


def _mean_delta_rand(X0, X1, y, n_split, base_seed=0):
    out = []
    for s in range(n_split):
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                                  stratify=y, random_state=base_seed * 100 + s)
        out.append(fit_ap(X1, y, tr, te, s) - fit_ap(X0, y, tr, te, s))
    return float(np.mean(out))


def _delta_rand_on(X0, X1, y, tr, te, seed):
    return fit_ap(X1, y, tr, te, seed) - fit_ap(X0, y, tr, te, seed)


def gate_bootstrap(n_rep=20, B=30, n_split=3, n=6000, alpha=0.05):
    """数据集级自助法的门槛：把抽样单位从"划分"改成"样本"。

    10 个共享同一份数据的划分不是独立样本，按它们算 t 会低估误差；
    这里对样本行做分层自助，每次重算 Δ_rand，再用自助分布的标准误做 t 检验。
    """
    hits, stats = 0, []
    for r in range(n_rep):
        Xb, G, y, layer = make_data(n, 1.0, seed=7000 + r)
        G = np.random.RandomState(r).randn(len(y), 1)
        X1 = np.column_stack([Xb, G])
        rng = np.random.RandomState(500 + r)
        vals = []
        for b in range(B):
            idx = np.concatenate([
                rng.choice(np.where(y == c)[0], int((y == c).sum()), replace=True)
                for c in (0, 1)])
            d = []
            for s in range(n_split):
                tr, te = train_test_split(idx, test_size=0.3, stratify=y[idx],
                                          random_state=s)
                d.append(_delta_rand_on(Xb, X1, y, tr, te, s))
            vals.append(np.mean(d))
        vals = np.array(vals)
        t = vals.mean() / (vals.std(ddof=1) / np.sqrt(B))
        hits += int(t >= 2)
        stats.append((vals.mean(), vals.std(ddof=1), t))
        print(f'  rep {r:>2}  Δ {vals.mean():+.4f}  SE {vals.std(ddof=1):.4f}  t {t:5.2f}', flush=True)
    return hits / n_rep, stats
