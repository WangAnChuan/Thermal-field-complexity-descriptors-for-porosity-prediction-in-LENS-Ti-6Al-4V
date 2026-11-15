"""Preview versions of five figures redrawn for a single column (8.15 cm)."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from figstyle import apply, panel as ph, PAL

apply()
OUT = Path('figures/preview'); OUT.mkdir(parents=True, exist_ok=True)
W = 3.21          # 8.15 cm
R = Path('results')

def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight', pad_inches=0.01)
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=0.01, dpi=600)
    print('saved', name)

# ---------------- F2 dataset: stacked panels
d = pd.read_parquet('data/cache/labels.parquet')
pore = d[d['Porosity Label'] == 1]
per_layer = pore.groupby('Layer').size().reindex(range(1, 61), fill_value=0)
fig, axs = plt.subplots(2, 1, figsize=(W, 3.5), gridspec_kw={'hspace': 0.55})
ax = axs[0]
cols = [PAL['vermillion'] if L <= 2 else PAL['neutral_light'] for L in per_layer.index]
ax.bar(per_layer.index, per_layer.values, width=0.85, color=cols, edgecolor='none')
ax.set(xlim=(0.5, 60.5), ylim=(0, 26), ylabel='Pore frames')
ax.set_xticks([1, 20, 40, 60])
ax.text(6, 23, '42 of 71 in layers 1–2', fontsize=5.8, va='top')
ax.yaxis.grid(True, color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
ph(ax, 'a', 'Pore frames per layer', x=-0.20, y=1.03, size=7.2)
ax = axs[1]
small = pore[pore['Porosity Size (mm)'] < 0.6]; large = pore[pore['Porosity Size (mm)'] >= 0.6]
ax.scatter(large.Layer, large['Porosity Size (mm)'], s=9, color=PAL['vermillion'], edgecolor='white', lw=0.25, zorder=3)
ax.scatter(small.Layer, small['Porosity Size (mm)'], s=9, color=PAL['blue'], edgecolor='white', lw=0.25, zorder=3)
ax.axvspan(0.5, 11.5, color='#000000', alpha=0.04, lw=0)
ax.set(xlim=(0.5, 60.5), ylim=(0, 1.1), xlabel='Layer index', ylabel='Pore size (mm)')
ax.set_xticks([1, 20, 40, 60]); ax.set_yticks([0, 0.3, 0.6, 0.9])
ax.text(5.5, 0.62, 'interface', fontsize=5.4, color=PAL['neutral'], ha='center')
ax.text(35, 0.62, 'bulk', fontsize=5.4, color=PAL['neutral'], ha='center')
ax.yaxis.grid(True, color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
ph(ax, 'b', 'Two pore populations', x=-0.20, y=1.03, size=7.2)
save(fig, 'p02_dataset'); plt.close(fig)

# ---------------- F5 feature sets
combo = pd.read_csv(R/'results_combo.csv'); abl = pd.read_csv(R/'results_ablation.csv')
lab6 = ['base', '+E', '+G', '+E+G', '+EGA', 'full']
fig, axs = plt.subplots(2, 1, figsize=(W, 3.9), gridspec_kw={'hspace': 0.72, 'height_ratios': [1.2, 1]})
ax = axs[0]; x = np.arange(6)
ax.errorbar(x, combo.cat_forward, yerr=combo.cat_forward_sd, fmt='s', ms=3.2, color=PAL['vermillion'], ecolor=PAL['vermillion'], elinewidth=0.6, capsize=1.2, label='Forward')
ax.plot(x, combo.cat_group5, 'o', ms=3.2, color=PAL['blue'], label='Grouped')
ax.set_xticks(x); ax.set_xticklabels(lab6, fontsize=5.4, rotation=35, ha='right', rotation_mode='anchor')
ax.set(ylabel='PR-AUC', ylim=(0.78, 1.02))
ax.yaxis.grid(True, color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
ax.legend(loc='lower right', fontsize=5.4, handletextpad=0.3)
ph(ax, 'a', 'PR-AUC by feature set', x=-0.26, y=1.04, size=7.2)
ax = axs[1]; xx = np.arange(5); w = 0.36
names = ['Entropy', 'Gradient', 'Topology', 'Moran', 'Area']
ax.barh(xx - w/2, abl.forward - 0.845, w, color=PAL['vermillion'], label='Forward')
ax.barh(xx + w/2, abl.group5 - 0.938, w, color=PAL['blue'], label='Grouped')
ax.axvline(0, color=PAL['ink'], lw=0.6)
ax.set_yticks(xx); ax.set_yticklabels(names, fontsize=5.8)
ax.set(xlabel='Gain over own baseline', xlim=(-0.07, 0.10))
ax.xaxis.grid(True, color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
ax.legend(loc='upper right', fontsize=5.0, handletextpad=0.3)
ph(ax, 'b', 'Descriptor groups', x=-0.30, y=1.04, size=7.2)
save(fig, 'p05_feature_sets'); plt.close(fig)

# ---------------- F7 forest, short labels
rows = [('Complexity, Cat', 2.51, 1.64, 4.95, True),
        ('Complexity, XGB', 1.33, 0.81, 2.85, True),
        ('Gradients, Cat', 3.51, 1.38, 26.76, True),
        ('Entropy, Cat', 1.48, 0.82, 3.10, True),
        ('Entropy, XGB', 1.19, 0.73, 2.55, True),
        ('Moments, Cat', 5.71, None, None, False),
        ('Rand. proj., Cat', 2.74, None, None, False),
        ('Permuted, Cat', None, None, None, False)]
fig, ax = plt.subplots(figsize=(W, 2.5))
y = np.arange(len(rows))[::-1]
for i, (lab, p, lo, hi, ok) in enumerate(rows):
    c = PAL['blue'] if 'Cat' in lab else PAL['vermillion']
    if ok:
        ax.plot([lo, hi], [y[i], y[i]], lw=0.8, color=c)
        ax.plot(p, y[i], 'o', ms=3.2, color=c, mec='white', mew=0.3)
    elif p:
        ax.plot(p, y[i], 'o', ms=3.2, mfc='white', mec=PAL['neutral'], mew=0.8)
    else:
        ax.text(1.3, y[i], 'n.d.', fontsize=5.2, va='center', ha='center', color=PAL['neutral'])
ax.axhline(2.5, color='#DDDDDD', lw=0.5)
for i,(lab,p_,lo,hi,ok) in enumerate(rows):
    if ok: ax.text(hi*1.12, y[i], f'{p_:.1f}', fontsize=5.2, va='center', color=PAL['blue'] if 'Cat' in lab else PAL['vermillion'])
ax.axvline(1, color=PAL['grey'], lw=0.7, ls=(0, (2, 2)))
ax.set_xscale('log'); ax.set_xlim(0.6, 40)
ax.set_xticks([1, 2, 5, 10, 20]); ax.set_xticklabels(['1', '2', '5', '10', '20'], fontsize=6)
ax.minorticks_off()
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=5.6)
ax.set_ylim(-0.7, 7.6)
ax.set_xlabel('PTR (log scale)')
ph(ax, None, 'Protocol transfer ratio', x=-0.52, y=1.03, size=7.2)
save(fig, 'p07_forest'); plt.close(fig)

# ---------------- F10 confound, stacked models
data = {'CatBoost': {'all': [0.005, 0.021, 0.086], 'bulk': [0.060, 0.055, 0.151]},
        'XGBoost': {'all': [0.017, 0.020, 0.054], 'bulk': [0.103, 0.092, 0.137]}}
fig, axs = plt.subplots(2, 1, figsize=(W, 3.4), gridspec_kw={'hspace': 0.55}, sharex=True)
for k, (m, v) in enumerate(data.items()):
    ax = axs[k]; x = np.arange(3); w = 0.36
    ax.bar(x - w/2, v['all'], w, color=PAL['neutral_light'], label='All layers')
    ax.bar(x + w/2, v['bulk'], w, color=PAL['vermillion'], label='Layers ≥ 12')
    ax.set_ylim(0, 0.20); ax.set_yticks([0, 0.05, 0.10, 0.15])
    ax.yaxis.grid(True, color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
    ax.text(2 - w/2, v['all'][2] + 0.006, f"{v['all'][2]/v['all'][0]:.1f}×", ha='center', fontsize=5.6, color=PAL['neutral'])
    ax.text(2 + w/2, v['bulk'][2] + 0.006, f"{v['bulk'][2]/v['bulk'][0]:.1f}×", ha='center', fontsize=5.6, color=PAL['vermillion'], weight='bold')
    ph(ax, 'ab'[k], m, x=-0.24, y=1.03, size=7.2)
    if k == 0: ax.legend(loc='upper left', fontsize=5.2, handletextpad=0.3)
axs[1].set_xticks(np.arange(3)); axs[1].set_xticklabels(['Random', 'Grouped', 'Forward'], fontsize=6)
axs[0].set_ylabel('ΔPR-AUC'); axs[1].set_ylabel('ΔPR-AUC')
fig.text(0.5, -0.02, '', ha='center')
save(fig, 'p10_confound'); plt.close(fig)
