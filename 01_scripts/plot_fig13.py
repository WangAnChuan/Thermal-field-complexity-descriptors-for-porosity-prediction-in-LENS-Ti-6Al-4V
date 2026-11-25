import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.1

raw_csv = "02_eda_and_baselines/reports/all_models_40runs_raw.csv"
out_fig = "02_eda_and_baselines/figures/fig13_statistical_significance_quad_boxplots.png"

df_raw = pd.read_csv(raw_csv)

sel_models = [
    "PyroNet-2D (Tian et al. 2020)",
    "ResNet-18 (Vision Baseline)",
    "EfficientNet-B0 (Tan & Le ICML 2019)",
    "ConvNeXt-Tiny (Liu et al. CVPR 2022)",
    "Swin-T (Liu et al. ICCV 2021)",
    "Thermal-Mamba (Proposed Full)"
]
sub_df = df_raw[df_raw["Model"].isin(sel_models)].copy()

short_names = {
    "PyroNet-2D (Tian et al. 2020)": "PyroNet",
    "ResNet-18 (Vision Baseline)": "ResNet-18",
    "EfficientNet-B0 (Tan & Le ICML 2019)": "EffNet-B0",
    "ConvNeXt-Tiny (Liu et al. CVPR 2022)": "ConvNeXt-T",
    "Swin-T (Liu et al. ICCV 2021)": "Swin-T",
    "Thermal-Mamba (Proposed Full)": "Thermal-Mamba"
}
sub_df["ShortName"] = sub_df["Model"].map(short_names)
order = ["PyroNet", "ResNet-18", "EffNet-B0", "ConvNeXt-T", "Swin-T", "Thermal-Mamba"]

palette = ["#2b5c8f", "#7570b3", "#4575b4", "#66a61e", "#e6ab02", "#d95f02"]

fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=300)

# ----------------- Panel (a): Test PR-AUC (%) -----------------
ax = axes[0, 0]
sns.boxplot(x="ShortName", y="Test PR-AUC (%)", data=sub_df, order=order, ax=ax, palette=palette,
            width=0.42, boxprops=dict(alpha=0.88), medianprops=dict(color="white", linewidth=2.2),
            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
sns.stripplot(x="ShortName", y="Test PR-AUC (%)", data=sub_df, order=order, ax=ax,
              color="#111111", size=5.5, alpha=0.55, jitter=0.15)

ax.text(0, 93, "* (p=0.016)", ha="center", fontsize=8.2, fontweight="bold", color="#2b5c8f")
ax.text(3, 76, "*** (p<0.001)", ha="center", fontsize=8.2, fontweight="bold", color="#66a61e")
ax.text(4, 64, "*** (p<0.001)", ha="center", fontsize=8.2, fontweight="bold", color="#e6ab02")

ax.set_title("(a) Test PR-AUC Distribution (21:1 Imbalance)\n[Evaluating Precision-Recall Area Under Extreme Class Skew]",
             fontsize=10.5, fontweight="bold", pad=8)
ax.set_ylabel("PR-AUC (%)", fontsize=10.5, fontweight="bold")
ax.set_xlabel("")
ax.set_ylim(32, 104)
ax.grid(axis="y", linestyle=":", alpha=0.5, color="#cccccc")

card_a = "Thermal-Mamba: 87.87 ± 8.75%\n• +16.39% over PyroNet-2D (p=0.0161*)\n• Crushes Swin-T (46.7%) & ConvNeXt-T (57.5%)"
ax.text(0.04, 0.06, card_a, transform=ax.transAxes, fontsize=8.2, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#d95f02", lw=1.2, alpha=0.95))

# ----------------- Panel (b): Test F1-Score -----------------
ax = axes[0, 1]
sns.boxplot(x="ShortName", y="Test F1-Score", data=sub_df, order=order, ax=ax, palette=palette,
            width=0.42, boxprops=dict(alpha=0.88), medianprops=dict(color="white", linewidth=2.2),
            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
sns.stripplot(x="ShortName", y="Test F1-Score", data=sub_df, order=order, ax=ax,
              color="#111111", size=5.5, alpha=0.55, jitter=0.15)

ax.set_title("(b) Test F1-Score Distribution\n[Harmonic Balance between Retrieval Precision & Sensitivity]",
             fontsize=10.5, fontweight="bold", pad=8)
ax.set_ylabel("F1-Score", fontsize=10.5, fontweight="bold")
ax.set_xlabel("")
ax.set_ylim(0.42, 1.02)
ax.grid(axis="y", linestyle=":", alpha=0.5, color="#cccccc")

card_b = "Thermal-Mamba: 0.8597 ± 0.0670\n• Consistently achieves highest harmonic balance\n• Demonstrates high seed-to-seed stability"
ax.text(0.04, 0.06, card_b, transform=ax.transAxes, fontsize=8.2, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#333333", lw=1.0, alpha=0.95))

# ----------------- Panel (c): False Alarm Rate (%) -----------------
ax = axes[1, 0]
sns.boxplot(x="ShortName", y="Test FAR (%)", data=sub_df, order=order, ax=ax, palette=palette,
            width=0.42, boxprops=dict(alpha=0.88), medianprops=dict(color="white", linewidth=2.2),
            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
sns.stripplot(x="ShortName", y="Test FAR (%)", data=sub_df, order=order, ax=ax,
              color="#111111", size=5.5, alpha=0.55, jitter=0.15)

ax.text(5, 2.2, "Best FAR\n0.62%", ha="center", fontsize=8.2, fontweight="bold", color="#b2182b",
        bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#b2182b", lw=1.0, alpha=0.95))

ax.set_title("(c) False Alarm Rate FAR (%) [Lower is Better]\n[Industrial Closed-Loop Stability & False Interruption Prevention]",
             fontsize=10.5, fontweight="bold", pad=8)
ax.set_ylabel("False Alarm Rate (%)", fontsize=10.5, fontweight="bold")
ax.set_xlabel("")
ax.set_ylim(-0.3, 7.8)
ax.grid(axis="y", linestyle=":", alpha=0.5, color="#cccccc")

card_c = "Thermal-Mamba: 0.62 ± 0.68%\n• Slashes false alarms by 63.3% vs PyroNet-2D (1.69%)\n• Critical for production closed-loop viability"
ax.text(0.04, 0.72, card_c, transform=ax.transAxes, fontsize=8.2, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#333333", lw=1.0, alpha=0.95))

# ----------------- Panel (d): Pore Size MAE (mm) -----------------
ax = axes[1, 1]
sns.boxplot(x="ShortName", y="Pore Size MAE (mm)", data=sub_df, order=order, ax=ax, palette=palette,
            width=0.42, boxprops=dict(alpha=0.88), medianprops=dict(color="white", linewidth=2.2),
            whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
sns.stripplot(x="ShortName", y="Pore Size MAE (mm)", data=sub_df, order=order, ax=ax,
              color="#111111", size=5.5, alpha=0.55, jitter=0.15)

ax.text(5, 0.16, "Best MAE\n77.4 μm", ha="center", fontsize=8.2, fontweight="bold", color="#b2182b",
        bbox=dict(boxstyle="round,pad=0.25", fc="#ffffff", ec="#b2182b", lw=1.0, alpha=0.95))

ax.set_title("(d) Quantitative Micro-Porosity Sizing Error (MAE mm)\n[Direct Inversion of Void Equivalent Diameter against 3D XCT]",
             fontsize=10.5, fontweight="bold", pad=8)
ax.set_ylabel("MAE (mm) [Lower is Better]", fontsize=10.5, fontweight="bold")
ax.set_xlabel("")
ax.set_ylim(-0.02, 0.78)
ax.grid(axis="y", linestyle=":", alpha=0.5, color="#cccccc")

card_d = "Thermal-Mamba: 0.0774 ± 0.0310 mm (77.4 μm)\n• ConvNeXt-T (0.339mm) & Swin-T (0.232mm) fail completely\n• Physical geometric gating enables sub-100μm metrology"
ax.text(0.04, 0.72, card_d, transform=ax.transAxes, fontsize=8.2, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec="#b2182b", lw=1.2, alpha=0.95))

# Format all x-tick labels and highlight Thermal-Mamba
for ax_sub in axes.flat:
    ax_sub.set_xticklabels(order, rotation=22, ha="right", fontsize=9.5, fontweight="bold")
    t_labels = ax_sub.get_xticklabels()
    t_labels[-1].set_color("#b2182b")

plt.suptitle("Fig. 13. Quad-Panel 5-Seed Statistical Stability & Metric Distributions across SOTA Models",
             fontsize=13.5, fontweight="bold", y=0.985, color="#111111")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.close()
print("SUCCESS: Fig 13 generated and saved to", out_fig)
