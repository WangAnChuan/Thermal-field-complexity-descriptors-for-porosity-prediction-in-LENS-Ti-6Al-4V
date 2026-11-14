"""House style for manuscript figures.

Matches Figure 1: Times, a bold "(a) Short title" heading on each panel,
light horizontal rules instead of a full grid, and one palette.
"""
import matplotlib as mpl

PAL = {'neutral': '#6E6E6E', 'neutral_light': '#B8B8B8', 'blue': '#0072B2',
       'vermillion': '#D55E00', 'green': '#009E73', 'ink': '#222222',
       'grey': '#8C8C8C', 'test_edge': '#7A3300', 'off': '#E8E8E8'}


def apply():
    mpl.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Liberation Serif', 'Nimbus Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 7.5, 'axes.titlesize': 7.5, 'axes.labelsize': 7.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.6,
        'axes.spines.right': False, 'axes.spines.top': False,
        'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'legend.frameon': False, 'figure.dpi': 300, 'savefig.dpi': 600,
        'axes.axisbelow': True,
    })


def panel(ax, tag=None, title=None, x=-0.12, y=1.04, size=8.0):
    """Bold '(a) Title' heading at the top-left of a set of axes.

    Pass tag=None for a single-panel figure to print the title alone.
    """
    text = (f'({tag})  ' if tag else '') + (title or '')
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, weight='bold',
            va='bottom', ha='left', color=PAL['ink'])


def grid(ax, axis='y'):
    """Light rules instead of a full grid; no vertical rules by default."""
    ax.grid(axis=axis, color='#EEEEEE', linewidth=0.6)
    ax.set_axisbelow(True)
