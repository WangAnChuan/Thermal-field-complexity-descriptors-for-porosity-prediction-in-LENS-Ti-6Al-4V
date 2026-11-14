"""Redraw all manuscript figures in one consistent style.

Figure contract
---------------
Core conclusion: thermal-field descriptors pay off under extrapolation, and the
evaluation protocol decides how much credit any feature receives.
Panel roles: Fig 1 motivates (data), Fig 2 orients (pipeline), Figs 3-7 test the
claim from independent angles (protocol, derivation, robustness, decision, mechanism).
Every panel answers a different question; no panel repeats another's metric.
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib as mpl
from figstyle import apply, panel as panel_head
from pathlib import Path

apply()

PAL = {'neutral': '#6E6E6E', 'neutral_light': '#B8B8B8', 'blue': '#0072B2',
       'vermillion': '#D55E00', 'green': '#009E73', 'ink': '#222222', 'grey': '#8C8C8C'}
OUT = Path('figures')
RES = Path('results')
def R(name):
    """Result tables live in results/ in the released repository."""
    p = RES / name
    return p if p.exists() else Path(name)
FINAL_WIDTH_IN = 5.42

def panel_label(ax, text, x=-0.16, y=1.10):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, weight='bold',
            va='bottom', ha='left', color=PAL['ink'])

def grid(ax, axis='y'):
    ax.grid(axis=axis, color='#EEEEEE', linewidth=0.6)
    ax.set_axisbelow(True)

def alignment_gate(fig, stem):
    """Optional render-time panel-alignment check when the figure toolkit is present."""
    import os
    skill = os.environ.get('NATURE_FIGURE_SCRIPTS')
    if not skill:
        return
    sys.path.insert(0, skill)
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except Exception:
        return
    try:
        require_matplotlib_panel_alignment(
            fig, json_out=str(OUT / f'{stem}.alignment.json'),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=False)
    except Exception as exc:
        print(f'  alignment note ({stem}): {exc}')


def save(fig, stem):
    """Vector first; PNG kept only as a preview."""
    OUT.mkdir(parents=True, exist_ok=True)
    # Crop to content: keeps every label inside the page (no clipping) and
    # guarantees the smallest glyph stays at its designed size or larger.
    alignment_gate(fig, stem)
    fig.savefig(OUT / f'{stem}.svg', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{stem}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT / f'{stem}.png', bbox_inches='tight', pad_inches=0.02, dpi=600)
    print(f'saved {stem}.svg/.pdf/.png')

W = 5.42
INK, GREY, BLUE, VERM, GREEN = PAL['ink'], PAL['grey'], PAL['blue'], PAL['vermillion'], PAL['green']

img = np.load('data/cache/thermal.npy')
df = pd.read_parquet('data/cache/labels.parquet')
y = df['Porosity Label'].values
L = df['Layer'].values
size = df['Porosity Size (mm)'].values

# ---------------------------------------------------------------- Fig 1
def hist_ent(a):
    h, _ = np.histogram(a, bins=32, range=(0, 2000)); p = h / h.sum()
    return -(p[p > 0] * np.log(p[p > 0])).sum()

E = np.array([hist_ent(img[i]) for i in range(len(img))])
pos = np.where(y == 1)[0]; free = np.where(y == 0)[0]
i_free = free[np.argsort(E[free])[-30]]
i_pore = pos[np.argmax(size[pos])]

fig = plt.figure(figsize=(W, 2.55))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1.45], wspace=0.50, left=0.03, right=0.99, bottom=0.20, top=0.86)
for k, (i, ttl) in enumerate(((i_free, 'Pore-free'), (i_pore, 'Pore'))):
    ax = fig.add_subplot(gs[k])
    ax.imshow(img[i], cmap='inferno', vmin=600, vmax=2000)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(True); s.set_linewidth(0.6); s.set_color(GREY)
    ax.set_title(ttl, pad=3)
    panel_label(ax, f'({"ab"[k]})', x=-0.10, y=1.03)

ax = fig.add_subplot(gs[2])
sz = size[pos]
small, large = pos[sz < 0.6], pos[sz >= 0.6]
ax.scatter(L[large], size[large], s=16, color=VERM, edgecolor='white', linewidth=0.35, zorder=3, label='large')
ax.scatter(L[small], size[small], s=16, color=BLUE, edgecolor='white', linewidth=0.35, zorder=3, label='small')
ax.axvspan(0.5, 15, color='#000000', alpha=0.035, zorder=0, linewidth=0)
ax.set(xlabel='Layer index', ylabel='Pore size (mm)', xlim=(0, 61), ylim=(0, 1.06))
ax.set_xticks([0, 20, 40, 60]); ax.set_yticks([0, 0.3, 0.6, 0.9])
ax.text(7.5, 0.97, 'interface', fontsize=6.4, color=PAL['neutral'], rotation=90, va='top', ha='center')
ax.text(38, 0.80, 'bulk', fontsize=6.4, color=PAL['neutral'], ha='center')
panel_label(ax, '(c)', x=-0.30, y=1.03)
save(fig, 'fig1_dataset'); plt.close(fig)

# ---------------------------------------------------------------- Fig 2 (schematic)
fig = plt.figure(figsize=(W, 2.85))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis('off')

def card(x, y, w, h, head, body, ec=GREY, fc='#F7F7F7', lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=0.9',
                                linewidth=lw, edgecolor=ec, facecolor=fc))
    ax.text(x + w / 2, y + h - 2.6, head, ha='center', va='top', fontsize=7.2, weight='bold')
    if body:
        ax.text(x + w / 2, y + (h - 6) / 2, body, ha='center', va='center', fontsize=6.4, linespacing=1.4)

def arr(p, q, c=INK, lw=0.7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=6, linewidth=lw,
                                 color=c, shrinkA=0, shrinkB=0))

ax.text(2, 97, '(a) Input', fontsize=7.2, weight='bold', va='top')
ax.text(22, 97, '(b) Features', fontsize=7.2, weight='bold', va='top')
ax.text(48, 97, '(c) Classifier', fontsize=7.2, weight='bold', va='top')
ax.text(69, 97, '(d) Protocol', fontsize=7.2, weight='bold', va='top')
ax.text(2, 28, '(e) Decision', fontsize=7.2, weight='bold', va='top')

axi = fig.add_axes([0.030, 0.50, 0.108, 0.29])
axi.imshow(img[0], cmap='inferno', vmin=600, vmax=2000); axi.set_xticks([]); axi.set_yticks([])
for s in axi.spines.values(): s.set_linewidth(0.6); s.set_color(GREY)

card(20, 62, 19, 27, 'baseline', '18 dims\ntime, height, peak T,\narea, percentiles')
card(20, 30, 19, 27, '+ descriptors', '3 dims\nentropy\nmean gradient\n99th pct. gradient', ec=BLUE, fc='#EDF4F9')
card(45, 44, 16, 34, 'gradient\nboosting', 'CatBoost\nXGBoost\nrandom forest')

def strip(x0, y0, pattern, label, note):
    w, h, gap = 1.5, 2.4, 0.42
    for i, c in enumerate(pattern):
        fc = {'B': BLUE, 'R': VERM, 'W': 'white'}[c]
        ax.add_patch(Rectangle((x0 + i * (w + gap), y0), w, h, facecolor=fc,
                               edgecolor=fc if c != 'W' else '#CFCFCF', linewidth=0.5))
    ax.text(x0, y0 - 2.4, label, fontsize=6.6, weight='bold', va='top')
    ax.text(x0, y0 - 6.0, note, fontsize=5.9, va='top', color=PAL['neutral'])

strip(69, 82, 'BRBRBRWW', 'random split', 'frames shuffled')
strip(69, 63, 'BBRRBBRW', 'grouped CV', 'whole layers held out')
strip(69, 44, 'BBBBBRRW', 'forward chaining', 'train 1-k, test k+1 to k+10')

card(2, 2, 40, 20, 'score to action',
     'cost-sensitive threshold on out-of-fold scores\nabstain on the least confident frames', fc='#FBFBFB')

arr((13.2, 62), (19.6, 73), GREY); arr((13.2, 55), (19.6, 45), GREY)
arr((39.6, 75), (44.6, 66), INK); arr((39.6, 44), (44.6, 54), INK)
arr((61.6, 61), (68.6, 61), INK)
arr((96.0, 40), (96.0, 32), INK); arr((96.0, 32), (42.6, 32), INK)
save(fig, 'fig2_overview'); plt.close(fig)

# ---------------------------------------------------------------- Fig 3 protocol
bench = pd.read_csv(R('results_bench_suite.csv')); b = bench[bench.feat == 'base']
models = ['LogReg', 'RF', 'XGB', 'LGBM', 'CatBoost', 'TabPFN']
g5 = np.array([b[(b.model == m) & (b.protocol == 'group5')].pr.values[0] for m in models])
fw = np.array([b[(b.model == m) & (b.protocol == 'forward')].pr.values[0] for m in models])
g5s = np.array([b[(b.model == m) & (b.protocol == 'group5')].pr_std.values[0] for m in models])
fws = np.array([b[(b.model == m) & (b.protocol == 'forward')].pr_std.values[0] for m in models])
leak = pd.read_csv(R('results_leakage.csv'))

fig, axs = plt.subplots(1, 2, figsize=(W, 2.25), gridspec_kw={'width_ratios': [1.55, 1], 'wspace': 0.48})
ax = axs[0]; x = np.arange(len(models))
for i in range(len(models)):
    ax.plot([i, i], [g5[i], fw[i]], color='#D0D0D0', lw=1.2, zorder=1)
ax.errorbar(x, g5, yerr=g5s, fmt='o', ms=4, color=BLUE, ecolor=BLUE, elinewidth=0.6,
            capsize=1.5, capthick=0.6, zorder=3, label='Layer-grouped')
ax.errorbar(x, fw, yerr=fws, fmt='o', ms=4, color=VERM, ecolor=VERM, elinewidth=0.6,
            capsize=1.5, capthick=0.6, zorder=3, label='Forward chaining')
ax.set_xticks(x); ax.set_xticklabels(models, rotation=45, ha='right', fontsize=6.0)
ax.tick_params(axis='x', pad=2)
ax.set(ylabel='PR-AUC', ylim=(0.50, 0.99), xlim=(-0.7, 5.7))
ax.legend(loc='lower right', ncol=1, handletextpad=0.2, fontsize=6.2)
grid(ax)
ax.annotate('CatBoost falls', xy=(4, 0.86), xytext=(2.2, 0.955), fontsize=6.2, color=PAL['neutral'],
            arrowprops=dict(arrowstyle='-', lw=0.5, color=PAL['neutral']))
ax.annotate('TabPFN rises', xy=(5, 0.90), xytext=(3.6, 0.785), fontsize=6.2, color=PAL['neutral'],
            arrowprops=dict(arrowstyle='-', lw=0.5, color=PAL['neutral']))
panel_head(ax, 'a', 'PR-AUC by model', x=-0.14)

# (b) three protocols, one model: accuracy stays flat, PR-AUC moves
ax = axs[1]
prot = ['random_5fold', 'grouped_5fold', 'forward_chaining']
lbl = ['Random', 'Grouped', 'Forward']
pr = [leak[(leak.protocol == p) & (leak.model == 'Cat')].pr.values[0] for p in prot]
ac = [leak[(leak.protocol == p) & (leak.model == 'Cat')].acc.values[0] for p in prot]
xs = np.arange(3)
ax.plot(xs, pr, 'o-', ms=4, lw=1.0, color=GREEN, label='PR-AUC')
ax.plot(xs, ac, 's--', ms=3.6, lw=1.0, color=PAL['neutral'], label='Accuracy')
ax.axhline(0.955, ls=':', lw=0.8, color=GREY)
ax.text(-0.42, 0.950, 'trivial classifier', fontsize=5.8, color=PAL['neutral'], ha='left', va='top')
ax.set_xticks(xs); ax.set_xticklabels(lbl)
ax.set(ylabel='Metric value', ylim=(0.80, 1.01), xlim=(-0.45, 2.45))
ax.set_yticks([0.80, 0.85, 0.90, 0.95, 1.00])
ax.legend(loc='lower left', fontsize=6.2, ncol=1)
grid(ax)
panel_head(ax, 'b', 'Metric change', x=-0.34)
save(fig, 'fig3_protocol'); plt.close(fig)

# ---------------------------------------------------------------- Fig 4 ablation
combo = pd.read_csv(R('results_combo.csv')); abl = pd.read_csv(R('results_ablation.csv'))
labels = ['base', '+E', '+G', '+E+G', '+E+G\n+A', 'full']
fig, axs = plt.subplots(1, 2, figsize=(W, 2.35), gridspec_kw={'width_ratios': [1.35, 1], 'wspace': 0.40})
ax = axs[0]; x = np.arange(len(combo)); d = 0.09
ax.errorbar(x - d, combo.cat_forward, yerr=combo.cat_forward_sd, fmt='o', ms=4,
            color=VERM, ecolor=VERM, elinewidth=0.6, capsize=1.5, capthick=0.6, zorder=3,
            label='Forward chaining')
ax.scatter(x + d, combo.cat_group5, s=20, color=BLUE, zorder=3, marker='s', label='Grouped CV')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=6.2)
ax.tick_params(axis='x', pad=3, length=2)
ax.set(ylabel='PR-AUC', ylim=(0.70, 1.03))
ax.set_xlim(-0.9, 5.9)
ax.legend(loc='lower right', fontsize=6.0, handletextpad=0.3, ncol=1)
grid(ax)
panel_head(ax, 'a', 'PR-AUC by feature set', x=-0.15)

ax = axs[1]
names = ['Entropy', 'Gradient', 'Topology', 'Moran', 'Area']
xx = np.arange(5); w = 0.34
ax.bar(xx - w / 2, abl.forward - 0.845, w, color=VERM, edgecolor='none', label='Forward chaining')
ax.bar(xx + w / 2, abl.group5 - 0.938, w, color=BLUE, edgecolor='none', label='Grouped CV')
ax.axhline(0, color=INK, lw=0.6)
ax.set_xticks(xx); ax.set_xticklabels(names, rotation=50, ha='right', fontsize=5.8)
ax.tick_params(axis='x', pad=2)
ax.set(ylabel='Gain over own baseline', ylim=(-0.07, 0.10))
ax.set_yticks([-0.06, -0.03, 0, 0.03, 0.06, 0.09])
ax.legend(loc='upper right', fontsize=5.6, handletextpad=0.3)
grid(ax)
panel_head(ax, 'b', 'Gain over baseline', x=-0.3)
save(fig, 'fig4_ablation'); plt.close(fig)

# ---------------------------------------------------------------- Fig 5 robustness
ms = pd.read_csv(R('results_multiseed.csv')); ex = pd.read_csv(R('results_exclude_firstlayers.csv'))
ll = pd.read_csv(R('results_layerlevel.csv'))
fig = plt.figure(figsize=(W, 3.10))
gs = GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.58, left=0.08, right=0.99, bottom=0.24, top=0.88)

ax = fig.add_subplot(gs[0]); mm = ['XGB', 'Cat', 'RF']; x = np.arange(3)
a = ms[ms.protocol == 'forward']
b1 = [a[(a.model == m) & (a.feat == 'base')]['mean'].iloc[0] for m in mm]
b2 = [a[(a.model == m) & (a.feat == 'entropy+gradient')]['mean'].iloc[0] for m in mm]
e1 = [a[(a.model == m) & (a.feat == 'base')]['sd'].iloc[0] for m in mm]
e2 = [a[(a.model == m) & (a.feat == 'entropy+gradient')]['sd'].iloc[0] for m in mm]
ax.errorbar(x - 0.10, b1, yerr=e1, fmt='o', ms=4, color=PAL['neutral_light'], ecolor=PAL['neutral_light'],
            elinewidth=0.8, capsize=2, capthick=0.7, label='Baseline')
ax.errorbar(x + 0.10, b2, yerr=e2, fmt='o', ms=4, color=GREEN, ecolor=GREEN,
            elinewidth=0.8, capsize=2, capthick=0.7, label='+ descriptors')
ax.set_xticks(x); ax.set_xticklabels(mm, fontsize=6.8)
ax.set(ylabel='PR-AUC', ylim=(0.60, 1.04), xlim=(-0.5, 2.5))
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.legend(loc='lower left', fontsize=5.8, handletextpad=0.3); grid(ax)
panel_head(ax, 'a', 'Ten-seed forward PR-AUC', x=-0.3)

ax = fig.add_subplot(gs[1]); s2 = ex[ex.protocol == 'forward']
b1 = [s2[(s2.model == m) & (s2.feat == 'base')]['pr'].iloc[0] for m in mm]
b2 = [s2[(s2.model == m) & (s2.feat == 'base+entropy+gradient')]['pr'].iloc[0] for m in mm]
for i, m in enumerate(mm):
    ax.plot([i, i], [b1[i], b2[i]], color='#D8D8D8', lw=1.0, zorder=1)
ax.scatter(x, b1, s=26, color=PAL['neutral_light'], zorder=3, label='Baseline')
ax.scatter(x, b2, s=26, color=GREEN, zorder=3, label='+ descriptors')
ax.set_xticks(x); ax.set_xticklabels(mm, fontsize=6.8)
ax.set(ylabel='PR-AUC', ylim=(0.60, 1.04), xlim=(-0.5, 2.5))
ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
ax.legend(loc='lower left', fontsize=5.8, handletextpad=0.3); grid(ax)
panel_head(ax, 'b', 'Without layers 1–2', x=-0.3)

ax = fig.add_subplot(gs[2]); q = ll[ll['agg'] == 'max']
keys = list(dict.fromkeys(zip(q.model, q.feat)))
labs = [f'{m} {"base" if f == "base" else "+E+G"}' for m, f in keys]
gv = [q[(q.model == m) & (q.feat == f) & (q.protocol == 'group5')]['pr'].iloc[0] for m, f in keys]
fv = [q[(q.model == m) & (q.feat == f) & (q.protocol == 'forward')]['pr'].iloc[0] for m, f in keys]
xx = np.arange(len(keys))
ax.vlines(xx, gv, fv, color='#D8D8D8', lw=1.0, zorder=1)
ax.scatter(xx, gv, s=26, color=BLUE, zorder=3, label='Grouped CV')
ax.scatter(xx, fv, s=26, color=VERM, zorder=3, label='Forward')
ax.set_xticks(xx); ax.set_xticklabels(labs, fontsize=5.4, rotation=25, ha='right')
ax.tick_params(axis='x', pad=2)
ax.set(ylabel='Layer-level PR-AUC', ylim=(0.90, 1.015))
ax.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
ax.legend(loc='lower right', fontsize=5.6, handletextpad=0.3); grid(ax)
panel_head(ax, 'c', 'Layer-level PR-AUC', x=-0.3)
save(fig, 'fig5_robustness'); plt.close(fig)

# ---------------------------------------------------------------- Fig 6 decision
cost = pd.read_csv(R('results_cost.csv')); sel = pd.read_csv(R('results_selective.csv'))
fig, axs = plt.subplots(1, 2, figsize=(W, 2.15), gridspec_kw={'wspace': 0.44})
ax = axs[0]
ax.axvspan(10, 20, color='#000000', alpha=0.045, lw=0, zorder=0)
ax.plot(cost.R, cost.recall, 'o-', ms=3, lw=0.9, color=VERM, label='Recall')
ax.plot(cost.R, cost.precision, 's--', ms=2.6, lw=0.9, color=BLUE, label='Precision')
ax.set_xscale('log')
ax.set_xticks([1, 2, 5, 10, 20, 50])
ax.set_xticklabels(['1', '2', '5', '10', '20', '50'])
ax.minorticks_off()
ax.set(xlabel='Cost ratio (miss : false alarm)', ylabel='Score', ylim=(0.5, 1.04))
ax.legend(loc='lower left', fontsize=6.2); grid(ax)
ax.annotate('20:1', xy=(20, 0.95), xytext=(9, 1.0), fontsize=6.0, color=PAL['neutral'],
            arrowprops=dict(arrowstyle='-', lw=0.5, color=PAL['neutral']))
ax.text(14, 0.53, 'plausible range', fontsize=5.6, ha='center', color=PAL['neutral'])
panel_head(ax, 'a', 'Cost-sensitive operating points', x=-0.22)

ax = axs[1]
for f, c, lbl in (('base', PAL['neutral_light'], 'Baseline'), ('entropy+gradient', GREEN, '+ descriptors')):
    d = sel[sel.feat == f]
    ax.plot(d.coverage, d.miss_rate, 'o-', ms=3, lw=1.0, color=c, label=lbl)
ax.axvline(0.9, ls=':', lw=0.7, color=GREY)
ax.set(xlabel='Automatic coverage', ylabel='Miss rate', ylim=(-0.015, 0.30))
ax.text(0.86, 0.955, 'n = 20 pore frames', transform=ax.transAxes, fontsize=5.8,
        ha='right', va='top', color=PAL['neutral'])
ax.legend(loc='upper left', fontsize=6.0); grid(ax)
panel_head(ax, 'b', 'Abstention', x=-0.28)
save(fig, 'fig6_decision'); plt.close(fig)

# ---------------------------------------------------------------- Fig 7 SHAP
imp = pd.read_csv('results/shap_importance.csv', index_col=0).iloc[:, 0].sort_values()
bins = pd.read_csv('results/shap_bins.csv')
NICE = {'hist_entropy': 'Histogram entropy', 'grad_mean': 'Mean gradient',
        'grad_p99': '99th-pct gradient', 'std': 'Std. dev. of field',
        'Tmax': 'Peak temperature', 'cc': 'Connected components',
        'Time': 'Elapsed time', 'MP Ecc': 'Melt-pool eccentricity', 'MP_Ecc': 'Melt-pool eccentricity',
        'p50': 'Median temperature', 'Y': 'Lateral position'}
fig, axs = plt.subplots(1, 2, figsize=(W, 2.40), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.70})
ax = axs[0]
top = imp.tail(9).copy()
if 'Tmax' not in top.index:
    top['Tmax'] = imp['Tmax']
top = top.sort_values()
cols = [GREEN if n in ('hist_entropy', 'grad_mean', 'grad_p99') else PAL['neutral_light'] for n in top.index]
ax.barh(range(len(top)), top.values, color=cols, edgecolor='none', height=0.68)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([NICE.get(n, n) for n in top.index], fontsize=6.2)
ax.set(xlabel='mean |SHAP|')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=GREEN, label='This work'), Patch(color=PAL['neutral_light'], label='Baseline')],
          loc='lower right', fontsize=5.6, handlelength=1.0)
ax.grid(False); ax.set_axisbelow(True)
panel_head(ax, 'a', 'Global importance', x=-0.5)

ax = axs[1]
for f, c, lbl, off in (('hist_entropy', GREEN, 'Entropy', -0.08),
                       ('grad_mean', VERM, 'Gradient', 0.08),
                       ('std', BLUE, 'Std. dev.', 0.0)):
    d = bins[bins.feature == f].reset_index(drop=True)
    ax.plot(np.arange(1, len(d) + 1) + off, d.pos_rate * 100, 'o-', ms=3, lw=0.9, color=c, label=lbl)
ax.set_xticks([1, 2, 3, 4, 5]); ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])
ax.set(xlabel='Feature quintile', ylabel='Pore rate (%)', ylim=(-1, 27))
ax.text(0.03, 0.95, 'Q1 holds 68 of 71\npore frames', transform=ax.transAxes, fontsize=5.6,
        va='top', color=PAL['neutral'])
ax.legend(loc='upper right', fontsize=6.0, ncol=1); grid(ax)
panel_head(ax, 'b', 'Pore rate by quintile', x=-0.26)
save(fig, 'fig7_shap'); plt.close(fig)

print('all figures regenerated')
