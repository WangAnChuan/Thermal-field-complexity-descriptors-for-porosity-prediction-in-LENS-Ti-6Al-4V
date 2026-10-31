"""Parse the 1564 pyrometer CSV frames into a cached NumPy tensor and a parquet label table."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os, sys

import numpy as np, pandas as pd, re, time
from pathlib import Path
D = Path('data'); out = D/'cache'; out.mkdir(exist_ok=True)
df = pd.read_excel(D/'thermal-porosity-table.xlsx')
files = [D/'cropped_thinwall_CSV_pyrometer'/f'Frame_{f}' for f in df['Frame']]
t=time.time()
imgs = np.empty((len(files),200,201), dtype=np.float32)
for i,p in enumerate(files):
    m = np.loadtxt(p, delimiter=',', skiprows=1, dtype=np.float32)
    imgs[i] = m[:,1:]
np.save(out/'thermal.npy', imgs)
df.to_parquet(out/'labels.parquet')
print(f"cached {imgs.shape} in {time.time()-t:.1f}s  -> {out/'thermal.npy'}")
