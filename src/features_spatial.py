"""Physics-inspired spatial descriptors of the thermal field (21 dimensions)."""
import numpy as np, pandas as pd
from scipy import ndimage

def build_spatial(cache_dir='data/cache'):
    from src.features import build
    X0, names0, y, g = build(cache_dir)
    img = np.load(f'{cache_dir}/thermal.npy')
    df = pd.read_parquet(f'{cache_dir}/labels.parquet')
    N = img.shape[0]
    F = {}
    for th in (1000, 1200, 1400, 1600):
        area = []; nreg = []; big = []; circ = []
        for i in range(N):
            m = img[i] > th
            lab, n = ndimage.label(m)
            area.append(m.mean())
            nreg.append(n)
            if n > 0:
                sizes = ndimage.sum(m, lab, range(1, n + 1))
                big.append(sizes.max() / m.sum() if m.sum() > 0 else 0)
                # 最大区域的圆度 4*pi*A/P^2
                k = int(np.argmax(sizes)) + 1
                reg = lab == k
                per = ndimage.binary_dilation(reg).sum() - reg.sum()
                circ.append(4 * np.pi * sizes.max() / (per ** 2 + 1e-9))
            else:
                big.append(0.0); circ.append(0.0)
        F[f't{th}_area'] = area; F[f't{th}_nreg'] = nreg
        F[f't{th}_bigfrac'] = big; F[f't{th}_circ'] = circ

    # 梯度
    gy = np.abs(np.diff(img, axis=1)); gx = np.abs(np.diff(img, axis=2))
    F['grad_mean'] = gx.mean((1, 2)) + gy.mean((1, 2))
    F['grad_p99'] = np.percentile(np.concatenate([gx.reshape(N, -1), gy.reshape(N, -1)], 1), 99, axis=1)

    # 空间自相关 Moran's I（对 1500°C 二值图的滞后）
    m = (img > 1500).astype(np.float32)
    mu = m.mean((1, 2), keepdims=True)
    z = m - mu
    lag = (np.roll(z, 1, axis=1) + np.roll(z, -1, axis=1) + np.roll(z, 1, axis=2) + np.roll(z, -1, axis=2)) / 4
    F['moran'] = (z * lag).sum((1, 2)) / ((z ** 2).sum((1, 2)) + 1e-9)

    # 直方图熵
    ent = []
    for i in range(N):
        h, _ = np.histogram(img[i], bins=32, range=(0, 2000)); p = h / (h.sum() + 1e-9)
        ent.append(-(p[p > 0] * np.log(p[p > 0])).sum())
    F['hist_entropy'] = ent

    # 高温区与峰值的空间分离度
    r = np.arange(200)[None, :, None]; c = np.arange(201)[None, None, :]
    w = (img > 1500).astype(np.float32); tot = w.sum((1, 2)) + 1e-9
    cr = (w * r).sum((1, 2)) / tot; cc = (w * c).sum((1, 2)) / tot
    pk = np.array([np.unravel_index(img[i].argmax(), img[i].shape) for i in range(N)])
    F['peak_centroid_dist'] = np.hypot(pk[:, 0] - cr, pk[:, 1] - cc)

    S = pd.DataFrame(F)
    cols = names0 + list(S.columns)
    X = np.column_stack([X0, S.values]).astype(np.float32)
    return X, cols, y, g
