"""Dataset overview (Section 3.2) and the matched/unmatched comparison (Section 5.9)."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from pathlib import Path
from figstyle import apply, panel as panel_head, PAL

apply()
OUT = Path('figures'); OUT.mkdir(exist_ok=True)
W = 5.42

def save(fig, stem):
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{stem}.png', bbox_inches='tight', pad_inches=0.02, dpi=600)
    print(f'saved {stem}.svg/.pdf/.png')

# ---------------------------------------------------------------- dataset overview
d = pd.read_parquet('data/cache/labels.parquet')
pore = d[d['Porosity Label'] == 1]
per_layer = pore.groupby('Layer').size().reindex(range(1, 61), fill_value=0)

fig, axs = plt.subplots(1, 2, figsize=(W, 2.0), gridspec_kw={'wspace': 0.34})
ax = axs[0]
cols = [PAL['vermillion'] if L <= 2 else PAL['neutral_light'] for L in per_layer.index]
ax.bar(per_layer.index, per_layer.values, width=0.8, color=cols, edgecolor='none')
ax.set(xlabel='Layer index', ylabel='Pore frames', xlim=(0.5, 60.5))
ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
ax.set_ylim(0, 26)
ax.text(3.2, 22.5, '42 of 71 pore frames\nin layers 1–2', fontsize=6.0, color=PAL['ink'], va='top')
ax.yaxis.grid(True, color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
panel_head(ax, 'a', 'Pore frames by layer', x=-0.22)

ax = axs[1]
small = pore[pore['Porosity Size (mm)'] < 0.6]
large = pore[pore['Porosity Size (mm)'] >= 0.6]
ax.scatter(large.Layer, large['Porosity Size (mm)'], s=14, color=PAL['vermillion'],
           edgecolor='white', linewidth=0.3, zorder=3, label='large (≥ 0.6 mm)')
ax.scatter(small.Layer, small['Porosity Size (mm)'], s=14, color=PAL['blue'],
           edgecolor='white', linewidth=0.3, zorder=3, label='small (< 0.6 mm)')
ax.axvspan(0.5, 11.5, color='#000000', alpha=0.04, lw=0)
ax.text(6, 0.55, 'interface', fontsize=6.0, color=PAL['neutral'], ha='center')
ax.text(36, 0.55, 'bulk', fontsize=6.0, color=PAL['neutral'], ha='center')
ax.set(xlabel='Layer index', ylabel='Pore size (mm)', xlim=(0.5, 60.5), ylim=(0, 1.08))
ax.set_xticks([1, 10, 20, 30, 40, 50, 60]); ax.set_yticks([0, 0.3, 0.6, 0.9])
ax.legend(loc='lower right', fontsize=5.8, handletextpad=0.3)
ax.yaxis.grid(True, color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
panel_head(ax, 'b', 'Two pore populations', x=-0.22)
save(fig, 'fig_dataset_overview'); plt.close(fig)

# ---------------------------------------------------------------- matched vs unmatched
# Values from Table 9 (CatBoost and XGBoost, delta PR-AUC when the three
# descriptors are added, three protocols, two test populations).
data = {
    'CatBoost': {'all': [0.005, 0.021, 0.086], 'bulk': [0.060, 0.055, 0.151]},
    'XGBoost':  {'all': [0.017, 0.020, 0.054], 'bulk': [0.103, 0.092, 0.137]},
}
fig, axs = plt.subplots(1, 2, figsize=(W, 2.15), gridspec_kw={'wspace': 0.30}, sharey=True)
prot = ['Random', 'Grouped', 'Forward']
for k, (model, vals) in enumerate(data.items()):
    ax = axs[k]
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, vals['all'],  w, color=PAL['neutral_light'], edgecolor='none', label='All layers')
    ax.bar(x + w/2, vals['bulk'], w, color=PAL['vermillion'], edgecolor='none', label='Layers ≥ 12')
    ax.set_xticks(x); ax.set_xticklabels(prot)
    ax.set_ylim(0, 0.19)
    ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.yaxis.grid(True, color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
    r_all = vals['all'][2] / vals['all'][0]
    r_bulk = vals['bulk'][2] / vals['bulk'][0]
    ax.text(2 - w/2, vals['all'][2] + 0.006, f'{r_all:.1f}×', ha='center', fontsize=6.2,
            color=PAL['neutral'])
    ax.text(2 + w/2, vals['bulk'][2] + 0.006, f'{r_bulk:.1f}×', ha='center', fontsize=6.2,
            color=PAL['vermillion'], weight='bold')
    panel_head(ax, 'ab'[k], model, x=-0.30)
axs[0].set_ylabel('ΔPR-AUC from descriptors', fontsize=7.0)
axs[0].legend(loc='upper left', fontsize=5.8, handletextpad=0.3)
save(fig, 'fig_confound'); plt.close(fig)
