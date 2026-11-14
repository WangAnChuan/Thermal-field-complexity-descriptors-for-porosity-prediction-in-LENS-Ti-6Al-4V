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
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
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
fig = plt.figure(figsize=(W, 3.55))
gs = GridSpec(3, 1, height_ratios=[1.0, 2.0, 2.3], hspace=0.62,
              left=0.10, right=0.78, bottom=0.16, top=0.94)

# (a) random split: every layer on both sides
ax = fig.add_subplot(gs[0])
for i in range(1, N + 1):
    ax.add_patch(Rectangle((i - 0.42, 0.0), 0.84, 0.70, facecolor=PAL['blue'], lw=0))
    ax.add_patch(Rectangle((i - 0.42, 0.70), 0.84, 0.30, facecolor=PAL['vermillion'], lw=0))
ax.set_xlim(0.3, N + 0.7); ax.set_ylim(0, 1.25)
ax.set_yticks([]); ax.set_xticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.text(1, 1.12, 'Random split', fontsize=7.0, weight='bold', va='bottom')
ax.text(N + 1.0, 0.5, 'every layer\ncontributes\nto both sets', fontsize=6.0,
        va='center', ha='left', color=PAL['neutral'])
panel_label(ax, '(a)', x=-0.09, y=1.02)

# (b) grouped 5-fold
ax = fig.add_subplot(gs[1])
for f in range(5):
    y0 = 4 - f
    for i in range(1, N + 1):
        test = (i - 1) % 5 == f
        ax.add_patch(Rectangle((i - 0.42, y0), 0.84, 0.72,
                               facecolor=PAL['vermillion'] if test else PAL['blue'], lw=0))
    ax.text(0.2, y0 + 0.36, f'fold {f+1}', fontsize=5.8, ha='right', va='center', color=PAL['neutral'])
ax.set_xlim(0.3, N + 0.7); ax.set_ylim(-0.1, 5.35)
ax.set_yticks([]); ax.set_xticks([])
for s in ax.spines.values(): s.set_visible(False)
ax.text(1, 5.18, 'Layer-grouped 5-fold', fontsize=7.0, weight='bold', va='bottom')
ax.text(N + 1.0, 2.5, 'whole layers\nheld out, but\ntraining layers\nbracket the test', fontsize=6.0,
        va='center', ha='left', color=PAL['neutral'])
panel_label(ax, '(b)', x=-0.09, y=1.02)

# (c) forward chaining
ax = fig.add_subplot(gs[2])
for r, k in enumerate((20, 30, 40, 50)):
    y0 = 3 - r
    for i in range(1, N + 1):
        if i <= k: c = PAL['blue']
        elif i <= k + 10: c = PAL['vermillion']
        else: c = '#E8E8E8'
        ax.add_patch(Rectangle((i - 0.42, y0), 0.84, 0.72, facecolor=c, lw=0))
    ax.text(0.2, y0 + 0.36, f'k = {k}', fontsize=5.8, ha='right', va='center', color=PAL['neutral'])
ax.set_xlim(0.3, N + 0.7); ax.set_ylim(-0.1, 4.35)
ax.set_yticks([]); ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
ax.set_xlabel('Layer index', fontsize=7.0)
ax.tick_params(axis='x', length=2)
for s in ax.spines.values(): s.set_visible(False)
ax.text(1, 4.18, 'Forward chaining', fontsize=7.0, weight='bold', va='bottom')
ax.text(N + 1.0, 1.5, 'test lies beyond\nevery training\nlayer: both\nchannels closed', fontsize=6.0,
        va='center', ha='left', color=PAL['neutral'])
panel_label(ax, '(c)', x=-0.09, y=1.02)

fig.legend(handles=[Patch(color=PAL['blue'], label='Training'),
                    Patch(color=PAL['vermillion'], label='Test'),
                    Patch(color='#E8E8E8', label='Not used')],
           loc='lower center', ncol=3, fontsize=6.4, bbox_to_anchor=(0.38, -0.005),
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
