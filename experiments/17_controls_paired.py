"""Complexity descriptors vs generic distribution moments: 7 boundaries x 3 seeds, paired."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from src.features import build
from src.features_spatial import build_spatial

Xb, nb, y, g = build(); Xs, ns, _, _ = build_spatial()
idx = {n: i for i, n in enumerate(ns)}
EG = Xs[:, [idx['hist_entropy'], idx['grad_mean'], idx['grad_p99']]]
img = np.load('data/cache/thermal.npy'); flat = img.reshape(len(img), -1)
MOM = np.column_stack([skew(flat, axis=1), kurtosis(flat, axis=1),
                       np.sqrt((flat**2).mean(1))]).astype(np.float32)

SETS = {'base': Xb,
        'moments': np.column_stack([Xb, MOM]).astype(np.float32),
        'complexity': np.column_stack([Xb, EG]).astype(np.float32),
        'both': np.column_stack([Xb, MOM, EG]).astype(np.float32)}

def cat(seed):
    return CatBoostClassifier(iterations=600, depth=4, learning_rate=0.05,
                              auto_class_weights='Balanced', random_seed=seed,
                              verbose=0, thread_count=4)

BOUNDS = [15, 20, 25, 30, 35, 40, 45]
score = {k: [] for k in SETS}
keys = []
for seed in range(3):
    for k in BOUNDS:
        tr = np.where(g <= k)[0]; te = np.where((g > k) & (g <= k + 10))[0]
        if (y[te] == 1).sum() < 1: continue
        for name, X in SETS.items():
            p = cat(seed).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            score[name].append(average_precision_score(y[te], p))
        keys.append((seed, k))

n = len(keys)
print(f"{'set':<12}{'PR-AUC':>10}")
for name in SETS:
    print(f"{name:<12}{np.mean(score[name]):>10.4f}")

rng = np.random.RandomState(0)
print("\npaired bootstrap over splits (1000 resamples):")
for a, b in (('complexity', 'moments'), ('complexity', 'base'), ('moments', 'base'), ('both', 'complexity')):
    d = np.array(score[a]) - np.array(score[b])
    bs = np.array([rng.choice(d, n).mean() for _ in range(1000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    print(f"  {a:>10} - {b:<10} delta={d.mean():+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]  p={max(p,1e-4):.4f}")

pd.DataFrame({'seed': [k[0] for k in keys], 'k': [k[1] for k in keys],
              **{name: score[name] for name in SETS}}).to_csv('results/results_control_paired.csv', index=False)
print("\nsaved results_control_paired.csv")
