"""Are the three descriptors robust to field of view, resolution and background?

Variants (all computed from the same 1564 frames):
  crop80 / crop60 : central window only
  down2           : 2x2 block average, i.e. half the sampling resolution
  hotmask         : entropy and gradients computed on pixels above the frame's
                    80th percentile only, so the background cannot contribute

Output: Spearman correlation with the original descriptor, and the forward
chained increment on the aligned population for each variant.
"""
import os, sys, warnings
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
sys.path.insert(0, REPO)
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scipy import ndimage
from sklearn.metrics import average_precision_score
from src.ptr import Xb, y, g, BOUNDS, mk, base_scores

img = np.load('data/cache/thermal.npy').astype(np.float32)
N = len(img)


def entropy(a, lo=0.0, hi=2000.0, bins=32):
    h, _ = np.histogram(a, bins=bins, range=(lo, hi)); p = h / (h.sum() + 1e-9)
    return -(p[p > 0] * np.log(p[p > 0])).sum()


def desc_from(arrs):
    """arrs: list of frames (possibly cropped/downsampled)."""
    H, GM, GP = [], [], []
    for a in arrs:
        H.append(entropy(a))
        gy = np.abs(np.diff(a, axis=0)); gx = np.abs(np.diff(a, axis=1))
        GM.append(gx.mean() + gy.mean())
        GP.append(np.percentile(np.concatenate([gx.ravel(), gy.ravel()]), 99))
    return np.column_stack([H, GM, GP]).astype(np.float32)


def masked_desc(a):
    """Hot-region only: entropy over pixels above p80, gradients within that mask."""
    thr = np.percentile(a, 80)
    m = a >= thr
    vals = a[m]
    H = entropy(vals, lo=float(thr), hi=float(a.max()) + 1e-6)
    b = np.where(m, a, thr).astype(np.float32)          # background flattened
    gy = np.abs(np.diff(b, axis=0)); gx = np.abs(np.diff(b, axis=1))
    return H, gx.mean() + gy.mean(), np.percentile(np.concatenate([gx.ravel(), gy.ravel()]), 99)


def pool2(a):
    h, w = (a.shape[0] // 2) * 2, (a.shape[1] // 2) * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))


variants = {}
variants['base'] = desc_from([img[i] for i in range(N)])
variants['crop80'] = desc_from([img[i, 20:180, 20:181] for i in range(N)])
variants['crop60'] = desc_from([img[i, 40:160, 40:161] for i in range(N)])
variants['down2'] = desc_from([pool2(img[i]) for i in range(N)])
HM = np.array([masked_desc(img[i]) for i in range(N)], dtype=np.float32)
variants['hotmask'] = HM

# correlations with the original
rows = []
for k, V in variants.items():
    if k == 'base': continue
    for j, nm in enumerate(['entropy', 'grad_mean', 'grad_p99']):
        rho = spearmanr(variants['base'][:, j], V[:, j]).statistic
        rows.append(dict(variant=k, descriptor=nm, spearman=round(float(rho), 3)))
print(pd.DataFrame(rows).to_string(index=False))

# forward-chained increment on the aligned population
mask = g >= 12
out = []
for model in ('Cat', 'XGB'):
    _, bf = base_scores(model, mask)
    for k, V in variants.items():
        Vm = V[mask]
        per_seed = []
        for s in range(3):
            per = []
            for kk in BOUNDS:
                tr = np.where(g[mask] <= kk)[0]; te = np.where((g[mask] > kk) & (g[mask] <= kk + 10))[0]
                if (y[mask][te] == 1).sum() < 1: continue
                X1 = np.column_stack([Xb[mask], Vm]).astype(np.float32)
                p = mk(model, s).fit(X1[tr], y[mask][tr]).predict_proba(X1[te])[:, 1]
                per.append(average_precision_score(y[mask][te], p))
            per_seed.append(np.mean(per))
        ef = np.array(per_seed)
        out.append(dict(model=model, variant=k, base_fwd=round(float(bf.mean()), 3),
                        ext_fwd=round(float(ef.mean()), 3), delta=round(float((ef - bf).mean()), 3)))
        print(out[-1], flush=True)

pd.DataFrame(out).to_csv('results/results_variant_robustness.csv', index=False)
print('saved results/results_variant_robustness.csv')
