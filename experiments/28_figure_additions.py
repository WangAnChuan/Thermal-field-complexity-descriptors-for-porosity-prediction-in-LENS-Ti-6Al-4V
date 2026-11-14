"""Figures added in revision: protocol schematic, PTR forest, calibration curve."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle
import matplotlib as mpl
from pathlib import Path

PAL = {'neutral': '#6E6E6E', 'neutral_light': '#B8B8B8', 'blue': '#0072B2',
       'vermillion': '#D55E00', 'green': '#009E73', 'ink': '#222222', 'grey': '#8C8C8C'}
mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42, 'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.8,
    'axes.spines.right': False, 'axes.spines.top': False, 'axes.linewidth': 0.7,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'legend.frameon': False, 'figure.dpi': 300, 'savefig.dpi': 600, 'axes.axisbelow': True,
})
OUT = Path('figures'); OUT.mkdir(exist_ok=True)
W = 5.42

def save(fig, stem):
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{stem}.png', bbox_inches='tight', pad_inches=0.02, dpi=600)
    print(f'saved {stem}.svg/.pdf/.png')

def panel_label(ax, text, x=-0.10, y=1.10):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, weight='bold',
            va='bottom', ha='left', color=PAL['ink'])

# ---------------------------------------------------------------- protocol schematic
N = 60
from sklearn.model_selection import GroupKFold
lbl = pd.read_parquet('data/cache/labels.parquet')
g_layers = lbl['Layer'].values
fold_of = {}
for f, (_, te) in enumerate(GroupKFold(n_splits=5).split(np.zeros((len(g_layers), 1)), groups=g_layers)):
    for L in np.unique(g_layers[te]):
        fold_of[int(L)] = f

INK2, EDGE = '#333333', '#8C2D04'
fig = plt.figure(figsize=(W, 3.45))
gs = GridSpec(3, 1, height_ratios=[1.0, 2.0, 2.3], hspace=0.75,
              left=0.10, right=0.71, bottom=0.17, top=0.93)

def channel_note(ax, adjacency, positional):
    for y, name, state in ((0.64, 'adjacency', adjacency), (0.34, 'position', positional)):
        ax.text(1.04, y, name, fontsize=6.4, color=INK2, transform=ax.transAxes, va='center')
        mark = r'$\checkmark$' if state == 'closed' else r'$\times$'
        ax.text(1.245, y, f'{mark} {state}', fontsize=6.4, transform=ax.transAxes, va='center',
                color=INK2 if state == 'closed' else PAL['vermillion'])

# (a) random split: five frames per layer, all sixty layers
ax = fig.add_subplot(gs[0])
rng = np.random.RandomState(7)
per_layer = 5
cells = {}
for L in range(1, N + 1):
    n_test = 1 if rng.rand() < 0.6 else 2
    idx = set(rng.choice(per_layer, n_test, replace=False))
    cells[L] = idx
    for k in range(per_layer):
        test = k in idx
        ax.add_patch(Rectangle((L - 0.42 + k * 0.84 / per_layer, 0.05), 0.84 / per_layer - 0.012, 0.80,
                               facecolor=PAL['vermillion'] if test else PAL['blue'],
                               edgecolor=EDGE if test else 'none', lw=0.3))
ax.set_xlim(0.3, N + 0.7); ax.set_ylim(-0.60, 1.10)
ax.set_yticks([]); ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
ax.tick_params(axis='x', length=2, pad=1, labelsize=6.6)
for sp in ax.spines.values(): sp.set_visible(False)
ax.text(1, 1.02, 'Random split: five frames per layer, all 60 layers', fontsize=7.0,
        weight='bold', va='bottom')
# bracket over one train/test boundary inside a layer
Lb, ks = 6, sorted(cells[6])
for k in ks:
    if k + 1 < per_layer and (k + 1) not in ks:
        x0 = Lb - 0.42 + k * 0.84 / per_layer
        x1 = x0 + 2 * 0.84 / per_layer - 0.012
        ax.plot([x0, x0, x1, x1], [-0.02, -0.10, -0.10, -0.02], lw=0.6, color=INK2, clip_on=False)
        ax.text((x0 + x1) / 2, -0.14, 'adjacent train/test frames', ha='center', va='top',
                fontsize=5.8, color=INK2)
        break
panel_label(ax, '(a)', x=-0.10, y=1.02)
channel_note(ax, 'open', 'open')

# (b) grouped 5-fold, actual assignment
ax = fig.add_subplot(gs[1])
for f in range(5):
    y0 = 4 - f
    for L in range(1, N + 1):
        test = fold_of[L] == f
        ax.add_patch(Rectangle((L - 0.42, y0), 0.84, 0.72,
                               facecolor=PAL['vermillion'] if test else PAL['blue'],
                               edgecolor=EDGE if test else 'none', lw=0.3))
    ax.text(0.2, y0 + 0.36, f'fold {f+1}', fontsize=5.8, ha='right', va='center', color=INK2)
ax.set_xlim(0.3, N + 0.7); ax.set_ylim(-0.1, 5.75)
ax.set_yticks([]); ax.set_xticks([])
for sp in ax.spines.values(): sp.set_visible(False)
ax.text(1, 5.68, 'Layer-grouped 5-fold (actual assignment)', fontsize=7.0, weight='bold', va='top')
panel_label(ax, '(b)', x=-0.10, y=1.06)
channel_note(ax, 'closed', 'open')

# (c) forward chaining
ax = fig.add_subplot(gs[2])
for r, k in enumerate((20, 30, 40, 50)):
    y0 = 3 - r
    for L in range(1, N + 1):
        if L <= k:
            fc, ec = PAL['blue'], 'none'
        elif L <= k + 10:
            fc, ec = PAL['vermillion'], EDGE
        else:
            fc, ec = '#EDEDED', 'none'
        ax.add_patch(Rectangle((L - 0.42, y0), 0.84, 0.72, facecolor=fc, edgecolor=ec, lw=0.3))
    ax.text(0.2, y0 + 0.36, f'k = {k}', fontsize=5.8, ha='right', va='center', color=INK2)
ax.set_xlim(0.3, N + 0.7); ax.set_ylim(-0.1, 4.72)
ax.set_yticks([]); ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
ax.set_xlabel('Layer index', fontsize=7.0, labelpad=2)
ax.tick_params(axis='x', length=2, labelsize=6.6)
for sp in ax.spines.values(): sp.set_visible(False)
ax.text(1, 4.66, 'Forward chaining', fontsize=7.0, weight='bold', va='top')
panel_label(ax, '(c)', x=-0.10, y=1.06)
channel_note(ax, 'closed', 'closed')

fig.legend(handles=[Patch(color=PAL['blue'], label='Training'),
                    Patch(facecolor=PAL['vermillion'], edgecolor=EDGE, label='Test'),
                    Patch(color='#EDEDED', label='Not used')],
           loc='lower center', ncol=3, fontsize=6.4, bbox_to_anchor=(0.40, 0.005),
           handlelength=1.1, columnspacing=1.6)
save(fig, 'fig_protocol_schematic'); plt.close(fig)

# ---------------------------------------------------------------- PTR forest
ptr = pd.read_csv('results/results_ptr.csv')
b = ptr[ptr['mask'] == '仅≥12层'].copy()
want = [('Cat', 'complexity', 'Complexity, 3 dims (CatBoost)'),
        ('XGB', 'complexity', 'Complexity, 3 dims (XGBoost)'),
        ('Cat', 'grad', 'Gradients, 2 dims (CatBoost)'),
        ('Cat', 'entropy', 'Entropy, 1 dim (CatBoost)'),
        ('XGB', 'entropy', 'Entropy, 1 dim (XGBoost)')]
rows = []
for m, blk, lab in want:
    r = b[(b.model == m) & (b.block == blk)].iloc[0]
    rows.append(dict(label=lab, ptr=r.d_fwd / r.d_rand, lo=r.ptr_ci_lo, hi=r.ptr_ci_hi))
d = pd.DataFrame(rows)

fig, ax = plt.subplots(figsize=(W * 0.78, 2.05))
y = np.arange(len(d))[::-1]
for i, r in d.iterrows():
    c = PAL['blue'] if 'CatBoost' in r.label else PAL['vermillion']
    ax.plot([r.lo, r.hi], [y[i], y[i]], lw=1.0, color=c, solid_capstyle='butt')
    ax.plot(r.ptr, y[i], 'o', ms=4.5, color=c, mec='white', mew=0.4, zorder=3)
    ax.text(r.hi * 1.12, y[i], f'{r.ptr:.2f}', fontsize=6.0, va='center', color=c)
ax.axvline(1.0, color=PAL['grey'], lw=0.8, ls=(0, (2, 2)))
ax.set_xscale('log')
ax.set_xlim(0.5, 45)
ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(['0.5', '1', '2', '5', '10', '20'])
ax.minorticks_off()
ax.set_yticks(y); ax.set_yticklabels(d.label, fontsize=6.2)
ax.set_xlabel('Protocol transfer ratio (log scale)', fontsize=7.0)
ax.text(1.06, -0.55, 'no transfer', fontsize=5.8, color=PAL['neutral'], va='center')
ax.set_ylim(-0.9, len(d) - 0.4)
panel_label(ax, '', x=0, y=0)
save(fig, 'fig_ptr_forest'); plt.close(fig)

# ---------------------------------------------------------------- calibration
cal = pd.read_csv('results/results_ptr_calibration.csv')
g = cal.groupby('omega').ptr.agg(['mean', 'std', 'count']).reset_index()
fig, ax = plt.subplots(figsize=(W * 0.62, 1.95))
ax.errorbar(g.omega, g['mean'], yerr=g['std'], fmt='o-', ms=3.6, lw=0.9,
            color=PAL['green'], ecolor=PAL['green'], elinewidth=0.7, capsize=2, capthick=0.6)
ax.plot([0.8, 5.2], [0.8, 5.2], ls=':', lw=0.7, color=PAL['grey'])
ax.text(4.6, 4.6, 'identity', fontsize=5.8, color=PAL['neutral'], ha='right', va='bottom')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xticks([1, 2, 3, 5]); ax.set_xticklabels(['1', '2', '3', '5'])
ax.set_yticks([1, 2, 3, 5]); ax.set_yticklabels(['1', '2', '3', '5'])
ax.minorticks_off()
ax.set_xlim(0.85, 5.5); ax.set_ylim(0.85, 5.5)
ax.set(xlabel='True effect growth factor', ylabel='Estimated PTR')
ax.text(5.3, 1.02, 'PTR rises only to 1.11\nfor a five-fold effect', fontsize=5.8, color=PAL['neutral'],
        ha='right', va='top')
save(fig, 'fig_calibration'); plt.close(fig)
print('additions done')
