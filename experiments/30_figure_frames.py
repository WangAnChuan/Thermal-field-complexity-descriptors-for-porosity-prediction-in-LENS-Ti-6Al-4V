"""Real thermal frames from the public build."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from pathlib import Path
from figstyle import apply, PAL

apply()
OUT = Path('figures'); OUT.mkdir(exist_ok=True)

d = pd.read_parquet('data/cache/labels.parquet').reset_index(drop=True)
img = np.load('data/cache/thermal.npy', mmap_mode='r')

def pick(layer, pore):
    sub = d[(d.Layer == layer) & (d['Porosity Label'] == pore)]
    return int(sub.index[0]) if len(sub) else None

panels = [
    (pick(20, 1), 'layer 20, pore', True),
    (pick(20, 0), 'layer 20, no pore', True),
    (pick(1, 0), 'layer 1, interface', False),
    (pick(60, 0), 'layer 60', False),
]

fig = plt.figure(figsize=(3.35, 3.55))
gs = GridSpec(2, 2, left=0.02, right=0.98, top=0.94, bottom=0.16, wspace=0.08, hspace=0.16)
for k, (i, title, contours) in enumerate(panels):
    ax = fig.add_subplot(gs[k // 2, k % 2])
    a = np.asarray(img[i], dtype=float)
    im = ax.imshow(a, cmap='inferno', vmin=1000, vmax=1800, interpolation='nearest')
    if contours:
        ax.contour(a > 1600, levels=[0.5], colors='#7FE7E0', linewidths=0.6)
        ax.contour(a > 1660, levels=[0.5], colors='white', linewidths=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor('#BBBBBB'); s.set_linewidth(0.4)
    ax.set_xlabel(title, fontsize=6.2, labelpad=2)
    ax.text(-0.06, 1.02, f'({"abcd"[k]})', transform=ax.transAxes, fontsize=7.5,
            weight='bold', va='bottom', ha='left')

cax = fig.add_axes([0.22, 0.075, 0.56, 0.018])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.set_label('Temperature (°C)', fontsize=6.2, labelpad=2)
cb.ax.tick_params(labelsize=5.8, length=1.8, width=0.5)
cb.outline.set_linewidth(0.4)

fig.savefig(OUT / 'fig_frames.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT / 'fig_frames.png', bbox_inches='tight', pad_inches=0.02, dpi=600)
print('saved fig_frames')
