"""Permutation gate for the protocol transfer ratio denominator."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from src.ptr import Xb, y, g, mk, BLOCKS, EG

BULK_GATE = 12


def delta_rand(model, mask, block, tr_list, seed=0):
    """在固定的划分列表上计算 Δ_random（基线 → 加块）。"""
    X0m, ym = Xb[mask], y[mask]
    X1m = np.column_stack([X0m, block[mask]]).astype(np.float32)
    d = []
    for s, (tr, te) in enumerate(tr_list):
        a0, yy = SMOTE(random_state=s, k_neighbors=5).fit_resample(X0m[tr], ym[tr])
        a1, _ = SMOTE(random_state=s, k_neighbors=5).fit_resample(X1m[tr], ym[tr])
        p0 = mk(model, s + seed).fit(a0, yy).predict_proba(X0m[te])[:, 1]
        p1 = mk(model, s + seed).fit(a1, yy).predict_proba(X1m[te])[:, 1]
        d.append(average_precision_score(ym[te], p1) - average_precision_score(ym[te], p0))
    return float(np.mean(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mask', default='bulk', choices=['bulk', 'all'])
    ap.add_argument('--model', default='Cat')
    ap.add_argument('--B', type=int, default=30)
    ap.add_argument('--splits', type=int, default=5)
    ap.add_argument('--out', default='results_ptr_gate.csv')
    args = ap.parse_args()

    mask = (g >= BULK_GATE) if args.mask == 'bulk' else np.ones(len(y), bool)
    ym = y[mask]
    tr_list = [train_test_split(np.arange(len(ym)), test_size=0.3, stratify=ym,
                                random_state=s) for s in range(args.splits)]

    rows = []
    print(f"{'特征块':<12}{'Δ随机':>9}{'零分布 95%':>12}{'p':>8}{'通过':>6}")
    for name, fn in BLOCKS.items():
        block = fn(0)
        obs = delta_rand(args.model, mask, block, tr_list)
        null = []
        for b in range(args.B):
            yp = np.random.RandomState(1000 + b).permutation(ym)
            # 用置换后的标签重算，但划分索引保持不变（配对置换）
            X0m = Xb[mask]
            X1m = np.column_stack([X0m, block[mask]]).astype(np.float32)
            d = []
            for s, (tr, te) in enumerate(tr_list):
                a0, yy = SMOTE(random_state=s, k_neighbors=5).fit_resample(X0m[tr], yp[tr])
                a1, _ = SMOTE(random_state=s, k_neighbors=5).fit_resample(X1m[tr], yp[tr])
                p0 = mk(args.model, s).fit(a0, yy).predict_proba(X0m[te])[:, 1]
                p1 = mk(args.model, s).fit(a1, yy).predict_proba(X1m[te])[:, 1]
                d.append(average_precision_score(yp[te], p1)
                         - average_precision_score(yp[te], p0))
            null.append(np.mean(d))
        null = np.array(null)
        p = (1 + (null >= obs).sum()) / (args.B + 1)
        ok = p <= 0.05
        print(f'{name:<12}{obs:>+9.4f}{np.percentile(null,95):>12.4f}{p:>8.3f}'
              f'{"是" if ok else "否":>6}', flush=True)
        rows.append(dict(mask=args.mask, model=args.model, block=name, d_rand=obs,
                         null_mean=null.mean(), null_sd=null.std(),
                         null_p95=np.percentile(null, 95), p_value=p, pass_gate=ok,
                         B=args.B, n_splits=args.splits))
        pd.DataFrame(rows).to_csv(args.out, index=False)

    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f'\nsaved {args.out}')


if __name__ == '__main__':
    main()
