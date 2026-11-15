"""Figure 4: model ranking under two protocols, and CatBoost across all three.

Single-column figure (8.15 cm). Panel (b) carries both PR-AUC and accuracy so
that the flat-accuracy claim is drawn rather than only stated in the text.
"""
import warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from figstyle import apply, panel as ph, PAL

apply()
OUT = Path('figures'); OUT.mkdir(exist_ok=True)
R = Path('results')
W = 3.21          # 8.15 cm

b = pd.read_csv(R / 'results_bench_suite.csv'); b = b[b.feat == 'base']
models = ['LogReg', 'RF', 'XGB', 'LGBM', 'CatBoost', 'TabPFN']
g = [b[(b.model == m) & (b.protocol == 'group5')].pr.values[0] for m in models]
gs = [b[(b.model == m) & (b.protocol == 'group5')].pr_std.values[0] for m in models]
f = [b[(b.model == m) & (b.protocol == 'forward')].pr.values[0] for m in models]
fs = [b[(b.model == m) & (b.protocol == 'forward')].pr_std.values[0] for m in models]
leak = pd.read_csv(R / 'results_leakage.csv')

fig, axs = plt.subplots(2, 1, figsize=(W, 3.7),
                        gridspec_kw={'hspace': 0.62, 'height_ratios': [1.25, 1]})

ax = axs[0]; y = np.arange(len(models))[::-1]
ax.errorbar(g, y - 0.12, xerr=gs, fmt='o', ms=3.4, color=PAL['blue'], ecolor=PAL['blue'],
            elinewidth=0.6, capsize=1.2, label='Grouped')
ax.errorbar(f, y + 0.12, xerr=fs, fmt='s', ms=3.2, color=PAL['vermillion'],
            ecolor=PAL['vermillion'], elinewidth=0.6, capsize=1.2, label='Forward')
ax.set_yticks(y); ax.set_yticklabels(models, fontsize=6.0)
ax.set(xlim=(0.6, 1.04), xlabel='PR-AUC')
ax.xaxis.grid(True, color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
ax.legend(loc='lower left', fontsize=5.6, handletextpad=0.3)
ph(ax, 'a', 'PR-AUC by model', x=-0.34, y=1.02, size=7.2)

ax = axs[1]
prot = ['random_5fold', 'grouped_5fold', 'forward_chaining']
lab = ['Random', 'Grouped', 'Forward']
pr = [leak[(leak.protocol == p) & (leak.model == 'Cat')].pr.values[0] for p in prot]
ac = [leak[(leak.protocol == p) & (leak.model == 'Cat')].acc.values[0] for p in prot]
x = np.arange(3)
ax.plot(x, pr, 'o-', ms=3.4, lw=0.9, color=PAL['green'], label='PR-AUC')
ax.plot(x, ac, 's--', ms=3.0, lw=0.9, color=PAL['neutral'], label='Accuracy')
ax.axhline(0.955, ls=':', lw=0.7, color=PAL['grey'])
ax.text(2.18, 0.947, 'trivial', fontsize=5.2, ha='right', va='top', color=PAL['neutral'])
ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=6.0)
ax.set(ylim=(0.8, 1.01), ylabel='Metric', xlim=(-0.45, 2.45))
ax.yaxis.grid(True, color='#EEEEEE', lw=0.6); ax.set_axisbelow(True)
ax.legend(loc='lower left', fontsize=5.4, handletextpad=0.3)
ph(ax, 'b', 'CatBoost, three protocols', x=-0.28, y=1.04, size=7.2)

fig.savefig(OUT / 'fig_model_ranking.pdf', bbox_inches='tight', pad_inches=0.01)
fig.savefig(OUT / 'fig_model_ranking.png', bbox_inches='tight', pad_inches=0.01, dpi=600)
print('saved fig_model_ranking.pdf/.png')
