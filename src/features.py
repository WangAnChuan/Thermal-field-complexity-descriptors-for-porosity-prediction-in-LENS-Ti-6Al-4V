"""Base feature set (18 dimensions): process metadata and global temperature statistics."""
import numpy as np, pandas as pd

def build(cache_dir='data/cache'):
    X_img = np.load(f'{cache_dir}/thermal.npy')          # (N,200,201)
    df = pd.read_parquet(f'{cache_dir}/labels.parquet')
    f = pd.DataFrame({
        'Time': df['Time (Sec)'], 'Y': df['Y (mm)'], 'Z': df['Z (mm)'],
        'MP_Area': df['MP Area'], 'MP_Ecc': df['MP Eccentricity'], 'Tmax': df['Tmax'],
    })
    f['mean'] = X_img.mean((1, 2)); f['std'] = X_img.std((1, 2))
    f['p50'] = np.percentile(X_img, 50, (1, 2))
    f['p95'] = np.percentile(X_img, 95, (1, 2))
    f['p99'] = np.percentile(X_img, 99, (1, 2))
    for th in (800, 1200, 1500):
        f[f'area_{th}'] = (X_img > th).mean((1, 2))
    w = (X_img > 1500).astype(np.float32)
    r = np.arange(200)[None, :, None]; c = np.arange(201)[None, None, :]
    tot = w.sum((1, 2)) + 1e-9
    cr = (w * r).sum((1, 2)) / tot; cc = (w * c).sum((1, 2)) / tot
    f['cr'] = cr; f['cc'] = cc
    f['sr'] = np.sqrt((w * (r - cr[:, None, None]) ** 2).sum((1, 2)) / tot)
    f['sc'] = np.sqrt((w * (c - cc[:, None, None]) ** 2).sum((1, 2)) / tot)
    return (f.values.astype(np.float32), f.columns.tolist(),
            df['Porosity Label'].values.astype(int), df['Layer'].values)
