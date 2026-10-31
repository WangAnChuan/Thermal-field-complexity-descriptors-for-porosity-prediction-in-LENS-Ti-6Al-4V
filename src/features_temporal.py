"""Temporal and heat-history features (11 dimensions), used only in a supplementary ablation."""
import numpy as np, pandas as pd

def build_all(cache_dir='data/cache'):
    from src.features import build
    X0, names0, y, g = build(cache_dir)
    df = pd.read_parquet(f'{cache_dir}/labels.parquet')
    X_img = np.load(f'{cache_dir}/thermal.npy')
    base = pd.DataFrame(X0, columns=names0)

    # 逐帧温度统计（用于构造滚动特征）
    s_mean = X_img.mean((1, 2)); s_std = X_img.std((1, 2)); s_max = X_img.max((1, 2))
    s_a15 = (X_img > 1500).mean((1, 2)); s_a12 = (X_img > 1200).mean((1, 2))

    out = {c: base[c].values for c in names0}
    lay = df['Layer'].values; z = df['Z (mm)'].values
    pos_in_layer = np.zeros(len(df)); roll3 = {k: np.zeros(len(df)) for k in ('mean','std','a15')}
    dev = np.zeros(len(df)); cum = np.zeros(len(df))

    # 层内时序：按 Layer 分组，层内顺序即帧序
    start = 0
    for L, idx in df.groupby('Layer', sort=True).indices.items():
        idx = np.sort(idx); n = len(idx)
        pos_in_layer[idx] = np.arange(n) / max(n - 1, 1)
        for j, i in enumerate(idx):
            lo = max(0, j - 2)
            roll3['mean'][i] = s_mean[idx[lo:j+1]].mean()
            roll3['std'][i]  = s_std[idx[lo:j+1]].mean()
            roll3['a15'][i]  = s_a15[idx[lo:j+1]].mean()
            prev = s_std[idx[:j]].mean() if j > 0 else s_std[i]
            dev[i] = s_std[i] - prev                       # 相对本层历史均值的偏离
            cum[i] = s_a15[idx[:j+1]].sum()                # 层内累计高温面积（热积累代理）

    out.update({
        'roll3_mean': roll3['mean'], 'roll3_std': roll3['std'], 'roll3_a15': roll3['a15'],
        'dev_std': dev, 'cum_a15': cum, 'pos_in_layer': pos_in_layer,
        'layer_norm': df['Layer'].values / df['Layer'].max(),
        'std_x_pos': s_std * pos_in_layer,
    })
    cols = names0 + ['roll3_mean','roll3_std','roll3_a15','dev_std','cum_a15','pos_in_layer','layer_norm','std_x_pos']
    X = np.column_stack([out[c] for c in cols]).astype(np.float32)
    return X, cols, y, g
