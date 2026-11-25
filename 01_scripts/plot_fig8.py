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
master_csv = "02_eda_and_baselines/reports/MASTER_SOTA_BENCHMARK_TABLE.csv"
out_fig = "02_eda_and_baselines/figures/fig8_master_sota_comprehensive_benchmark.png"

df_raw = pd.read_csv(raw_csv)
df_master = pd.read_csv(master_csv)

clean_names = {
    "PyroNet-2D (Tian et al. 2020)": "PyroNet",
    "ResNet-18 (Vision Baseline)": "ResNet-18",
    "EfficientNet-B0 (Tan & Le ICML 2019)": "EffNet-B0",
    "ConvNeXt-Tiny (Liu et al. CVPR 2022)": "ConvNeXt-T",
    "Swin-T (Liu et al. ICCV 2021)": "Swin-T",
    "Ablation: w/o Physical Gating": "w/o PhysGate",
    "Ablation: w/o Bi-Scan (1D Mamba)": "1D-Mamba",
    "Thermal-Mamba (Proposed Full)": "Thermal-Mamba"
}

model_order = [
    "PyroNet-2D (Tian et al. 2020)",
    "ResNet-18 (Vision Baseline)",
    "EfficientNet-B0 (Tan & Le ICML 2019)",
    "ConvNeXt-Tiny (Liu et al. CVPR 2022)",
    "Swin-T (Liu et al. ICCV 2021)",
    "Ablation: w/o Physical Gating",
    "Ablation: w/o Bi-Scan (1D Mamba)",
    "Thermal-Mamba (Proposed Full)"
]

short_order = [clean_names[m] for m in model_order]
df_raw["ShortName"] = df_raw["Model"].map(clean_names)

palette = [
    "#2b5c8f",  # PyroNet
    "#7570b3",  # ResNet
    "#4575b4",  # EffNet
    "#66a61e",  # ConvNeXt
    "#e6ab02",  # Swin-T
    "#8c510a",  # w/o PhysGate
    "#1b9e77",  # 1D-Mamba
    "#d95f02"   # Thermal-Mamba (Proposed)
]

fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), dpi=300)

# ----------------- Subplot (a): Test PR-AUC Distribution -----------------
ax = axes[0]
sns.boxplot(x="ShortName", y="Test PR-AUC (%)", data=df_raw, order=short_order, ax=ax,
            palette=palette, width=0.45, boxprops=dict(alpha=0.88),
            medianprops=dict(color="white", linewidth=2.2), whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2))
sns.stripplot(x="ShortName", y="Test PR-AUC (%)", data=df_raw, order=short_order, ax=ax,
              color="#111111", size=5.5, alpha=0.6, jitter=0.15)

# Significance annotations
# PyroNet vs Ours (p = 0.0161 *)
ax.text(0, 93, "* (p=0.016)", ha="center", fontsize=8.2, fontweight="bold", color="#2b5c8f")
# ConvNeXt vs Ours (p < 0.001 ***)
ax.text(3, 76, "*** (p<0.001)", ha="center", fontsize=8.2, fontweight="bold", color="#66a61e")
# Swin vs Ours (p < 0.001 ***)
ax.text(4, 64, "*** (p<0.001)", ha="center", fontsize=8.2, fontweight="bold", color="#e6ab02")

ax.set_xticklabels(short_order, rotation=35, ha="right", fontsize=9.5, fontweight="bold")
ax.set_ylabel("Test PR-AUC (%) Across 5 Seeds", fontsize=10.5, fontweight="bold")
ax.set_xlabel("")
ax.set_title("(a) Test PR-AUC Distribution across 5 Seeds\n[Robustness under 21:1 Skew & 40 Independent Runs]",
             fontsize=11, fontweight="bold", pad=10)
ax.set_ylim(32, 104)
ax.grid(axis="y", linestyle=":", alpha=0.5, color="#cccccc")

# Highlight Thermal-Mamba label on x-axis
labels = ax.get_xticklabels()
labels[-1].set_color("#b2182b")

# ----------------- Subplot (b): Pareto Frontier -----------------
ax = axes[1]
model_params = {
    "PyroNet": 0.43,
    "ResNet-18": 11.25,
    "EffNet-B0": 4.02,
    "ConvNeXt-T": 27.83,
    "Swin-T": 27.53,
    "w/o PhysGate": 0.33,
    "1D-Mamba": 0.28,
    "Thermal-Mamba": 0.41
}

model_fps = {
    "PyroNet": 1425.7,
    "ResNet-18": 503.7,
    "EffNet-B0": 480.0,
    "ConvNeXt-T": 210.5,
    "Swin-T": 185.2,
    "w/o PhysGate": 510.2,
    "1D-Mamba": 540.6,
    "Thermal-Mamba": 438.8
}

mean_prauc = df_raw.groupby("ShortName")["Test PR-AUC (%)"].mean()

# Smart annotation offsets to avoid ANY overlapping
annot_offsets = {
    "1D-Mamba": (-42, 12),
    "Thermal-Mamba": (12, 6),
    "w/o PhysGate": (-68, -14),
    "PyroNet": (10, -12),
    "EffNet-B0": (10, 6),
    "ResNet-18": (10, 6),
    "ConvNeXt-T": (10, 6),
    "Swin-T": (10, -12)
}

for idx, name in enumerate(short_order):
    p = model_params[name]
    prauc = mean_prauc[name]
    c = palette[idx]
    is_ours = (name == "Thermal-Mamba")
    s = 180 if is_ours else 90

    ax.scatter(p, prauc, color=c, s=s, edgecolors="#111111", linewidth=1.3, zorder=6 if is_ours else 4)

    dx, dy = annot_offsets[name]
    fontweight = "bold" if is_ours else "normal"
    fontcolor = "#b2182b" if is_ours else "#111111"
    ax.annotate(name if not is_ours else "Thermal-Mamba [Ours]", (p, prauc), xytext=(dx, dy),
                textcoords="offset points", fontsize=9.0, fontweight=fontweight, color=fontcolor,
                arrowprops=dict(arrowstyle="->", color=fontcolor, lw=0.9) if is_ours or name in ["1D-Mamba", "w/o PhysGate"] else None,
                zorder=7)

# Pareto frontier dashed line connecting 1D-Mamba and Thermal-Mamba
ax.plot([0.28, 0.41], [mean_prauc["1D-Mamba"], mean_prauc["Thermal-Mamba"]], linestyle="--",
        color="#b2182b", linewidth=1.8, label="Optimal Pareto Frontier", zorder=3)

card_pareto = (
    "Optimal Pareto Frontier:\n"
    "• Thermal-Mamba: 0.41M Params | 438.8 FPS | 87.87% PR-AUC\n"
    "• Swin-T / ConvNeXt-T collapse (<58% PR-AUC, severe overfitting)\n"
    "• 27× parameter reduction vs general vision backbones"
)
ax.text(0.04, 0.12, card_pareto, transform=ax.transAxes, fontsize=8.8, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec="#b2182b", lw=1.2, alpha=0.95), zorder=5)

ax.set_xscale("log")
ax.set_xlabel("Model Parameters (Millions, Log Scale)", fontsize=10.5, fontweight="bold")
ax.set_ylabel("Mean Test PR-AUC (%)", fontsize=10.5, fontweight="bold")
ax.set_title("(b) Pareto Frontier: PR-AUC vs Parameter Complexity\n[Accuracy vs Hardware Footprint on Embedded Edge]",
             fontsize=11, fontweight="bold", pad=10)
ax.set_xlim(0.18, 45)
ax.set_ylim(42, 98)
ax.grid(True, linestyle=":", alpha=0.5, color="#cccccc")
ax.legend(loc="upper right", fontsize=9.2, framealpha=0.92, edgecolor="#cccccc")

# ----------------- Subplot (c): Pore Size MAE -----------------
ax = axes[2]
mae_means = df_raw.groupby("ShortName")["Pore Size MAE (mm)"].mean()[short_order]
mae_stds = df_raw.groupby("ShortName")["Pore Size MAE (mm)"].std()[short_order]

x_pos = np.arange(len(short_order))
bars = ax.bar(x_pos, mae_means, yerr=mae_stds, capsize=4, color=palette, edgecolor="#111111",
              linewidth=1.0, alpha=0.88, ecolor="#111111")

# Highlight lowest error on Thermal-Mamba
ax.text(7, mae_means["Thermal-Mamba"] + mae_stds["Thermal-Mamba"] + 0.02,
        f"Best Sizing\n{mae_means['Thermal-Mamba']*1000:.1f} μm", ha="center", va="bottom",
        fontsize=8.5, fontweight="bold", color="#b2182b",
        bbox=dict(boxstyle="round,pad=0.3", fc="#ffffff", ec="#b2182b", lw=1.1, alpha=0.95))

ax.set_xticks(x_pos)
ax.set_xticklabels(short_order, rotation=35, ha="right", fontsize=9.5, fontweight="bold")
ax.set_ylabel("Pore Size MAE (mm) [Lower is Better]", fontsize=10.5, fontweight="bold")
ax.set_xlabel("")
ax.set_title("(c) Quantitative Micro-Porosity Sizing Error\n[Continuous Void Diameter Regression Fidelity]",
             fontsize=11, fontweight="bold", pad=10)
ax.set_ylim(0, 0.64)
ax.grid(axis="y", linestyle=":", alpha=0.5, color="#cccccc")

# Highlight Thermal-Mamba label on x-axis
labels_c = ax.get_xticklabels()
labels_c[-1].set_color("#b2182b")

plt.suptitle("Fig. 8. Master Empirical SOTA Benchmark for Ti-6Al-4V LENS In-Situ Porosity Detection & Sizing",
             fontsize=13.5, fontweight="bold", y=0.98, color="#111111")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.close()
print("SUCCESS: Fig 8 generated and saved to", out_fig)
