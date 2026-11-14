"""Figure: melt-pool geometry at the solidification isotherms."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from skimage.measure import regionprops, label as sk_label
from src.features import build

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

def panel_label(ax, text, x=-0.16, y=1.10):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, weight='bold',
            va='bottom', ha='left', color=PAL['ink'])

def save(fig, stem):
    fig.savefig(f'figures/{stem}.pdf', bbox_inches='tight', pad_inches=0.02)
    fig.savefig(f'figures/{stem}.png', bbox_inches='tight', pad_inches=0.02)
    print(f'saved {stem}.pdf/.png')
Xb, nb, y, g = build()
img = np.load('data/cache/thermal.npy')

uni = pd.read_csv('results/results_geometry_univariate.csv')
ev = pd.read_csv('results/results_geometry_eval.csv')

fig = plt.figure(figsize=(7.1, 2.75))
gs = GridSpec(1, 4, width_ratios=[0.68, 0.68, 1.14, 1.04], wspace=0.95,
              left=0.03, right=0.985, bottom=0.19, top=0.85)

# ---------------- (a) two frames with solidification isotherms ----------------
ip = int(np.where((y == 1) & (g == 42))[0][0])
inn = int(np.where((y == 0) & (g == 42))[0][0])
axes_a = []
for col, (k, ttl) in enumerate(((ip, 'pore frame'), (inn, 'no-pore frame'))):
    ax = fig.add_subplot(gs[col]); axes_a.append(ax)
    im = ax.imshow(img[k], cmap='inferno', vmin=900, vmax=2000, interpolation='nearest')
    ax.contour(img[k] > 1600, levels=[0.5], colors='#7FE7E0', linewidths=0.7)
    ax.contour(img[k] > 1660, levels=[0.5], colors='white', linewidths=0.7)
    p = max(regionprops(sk_label(img[k] > 1660)), key=lambda q: q.area)
    ax.plot(p.centroid[1], p.centroid[0], marker='+', ms=5, mew=0.8, color='white')
    ax.text(0.5, -0.09, ttl, transform=ax.transAxes, ha='center', va='top', fontsize=6.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
fig.text(0.005, 0.90, '(a)', fontsize=8, weight='bold')
fig.text(0.035, 0.90, 'liquidus 1660 °C  ·  solidus 1600 °C', fontsize=6.4, color=PAL['neutral'])
p1, p2 = axes_a[0].get_position(), axes_a[1].get_position()
cax = fig.add_axes([p1.x0 + 0.02, p1.y0 - 0.135, (p2.x1 - p1.x0) * 0.72, 0.018])
cb = fig.colorbar(im, cax=cax, orientation='horizontal')
cb.set_label('Temperature (°C)', fontsize=6.0, labelpad=1.5)
cb.ax.tick_params(labelsize=5.6, length=1.6, width=0.5)
cb.outline.set_linewidth(0.4)

# ---------------- (b) univariate discriminability ----------------
ax = fig.add_subplot(gs[2])
geom = uni[uni.feature.str.startswith('geo_')].copy()
desc = uni[uni.feature.isin(['hist_entropy', 'grad_mean', 'grad_p99'])].copy()
base = uni[uni.feature.isin(['MP_Area', 'MP_Ecc', 'Tmax', 'p99'])].copy()
for d in (geom, desc, base):
    d['s'] = (d.auc - 0.5).abs()
sel = pd.concat([desc.sort_values('s'), geom.sort_values('s').tail(5), base.sort_values('s').tail(2)])
colmap = {**{k: PAL['vermillion'] for k in desc.feature},
          **{k: PAL['neutral'] for k in geom.feature},
          **{k: PAL['neutral_light'] for k in base.feature}}
yy = np.arange(len(sel))
ax.hlines(yy, 0, sel.s.values, color=[colmap[f] for f in sel.feature], lw=1.0, alpha=0.55)
NICE = {'hist_entropy': 'Histogram entropy', 'grad_mean': 'Mean gradient', 'grad_p99': '99th-pct gradient',
        'geo_len': 'Pool length', 'geo_wid': 'Pool width', 'geo_ar': 'Aspect ratio',
        'geo_area': 'Pool area', 'geo_perim': 'Perimeter', 'geo_circ': 'Circularity',
        'geo_solidity': 'Solidity', 'geo_mushy': 'Mushy-zone ratio',
        'geo_G_mushy': 'Mushy grad.', 'geo_asym': 'Trailing asymmetry',
        'geo_peak_off': 'Peak offset', 'MP_Area': 'MP area (released)', 'MP_Ecc': 'MP eccentricity',
        'Tmax': 'Peak temperature', 'p99': '99th-pct temperature'}
DESC_SET = set(desc.feature); GEO_SET = set(geom.feature)
def shape(f):
    if f in DESC_SET: return 'o'
    if f in GEO_SET: return 's'
    return '^'
for i, (_, r) in enumerate(sel.iterrows()):
    ax.plot(r.s, i, shape(r.feature), ms=4.2, color=colmap[r.feature], mec='white', mew=0.4, zorder=3)
ax.set_yticks(yy)
ax.set_yticklabels([NICE.get(f, f) for f in sel.feature], fontsize=6.3)
for t, f in zip(ax.get_yticklabels(), sel.feature):
    t.set_color(colmap[f])
ax.set_ylim(-0.7, len(sel) - 0.3)
ax.set_xlabel('single-feature discriminability  |AUC − 0.5|', fontsize=6.8)
ax.set_xlim(0, 0.5)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)
panel_label(ax, '(b)', x=-0.60, y=1.02)

# ---------------- (c) forward-chaining PR-AUC ----------------
ax = fig.add_subplot(gs[3])
order = ['base', 'geo', 'ours', 'geo+ours']
lab = ['Baseline', '+ geometry', '+ descriptors', '+ both']
x = np.arange(len(order))
for m, c, off, mk_ in (('Cat', PAL['blue'], -0.13, 'o'), ('XGB', PAL['vermillion'], 0.13, 's')):
    v = [ev[(ev.model == m) & (ev.feats == f)].forward.values[0] for f in order]
    ax.plot(x + off, v, mk_, ms=4.6, color=c, mec='white', mew=0.4, label=m, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(lab, fontsize=6.2, rotation=24, ha='right', rotation_mode='anchor')
ax.set_xlabel('added to the baseline set', fontsize=6.8, labelpad=1)
ax.set_ylim(0.78, 1.0)
ax.set_ylabel('PR-AUC, forward chaining', fontsize=6.8)
base_v = ev[(ev.model == 'Cat') & (ev.feats == 'base')].forward.values[0]
ax.axhline(base_v, color=PAL['grey'], lw=0.7, ls=(0, (1, 2)), zorder=1)
ax.text(3.45, base_v - 0.006, 'baseline', fontsize=5.4, color=PAL['neutral'], ha='right', va='top')
ax.legend(loc='upper left', ncol=2, fontsize=5.8, handletextpad=0.25, columnspacing=0.6,
          bbox_to_anchor=(0.0, 1.02))
panel_label(ax, '(c)', x=-0.24, y=1.02)

save(fig, 'fig_geometry')
print('saved fig_geometry')
