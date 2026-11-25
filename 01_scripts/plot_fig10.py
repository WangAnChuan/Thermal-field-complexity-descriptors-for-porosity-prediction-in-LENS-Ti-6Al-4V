import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.1

master_csv = "02_eda_and_baselines/reports/MASTER_SOTA_BENCHMARK_TABLE.csv"
out_fig = "02_eda_and_baselines/figures/fig10_multimetric_radar_comparison.png"

df_master = pd.read_csv(master_csv)

models_to_plot = [
    ("PyroNet-2D (Tian et al. 2020)", "PyroNet-2D (2020)", "#2b5c8f", 1.8, 0.05, 3),
    ("ResNet-18 (Vision Baseline)", "ResNet-18", "#7570b3", 1.8, 0.05, 3),
    ("ConvNeXt-Tiny (Liu et al. CVPR 2022)", "ConvNeXt-Tiny", "#66a61e", 1.8, 0.05, 3),
    ("Swin-T (Liu et al. ICCV 2021)", "Swin-T (ICCV 2021)", "#e6ab02", 1.8, 0.05, 3),
    ("Thermal-Mamba (Proposed Full)", "Thermal-Mamba [Ours]", "#d95f02", 3.0, 0.22, 6)
]

categories = [
    "Defect Retrieval\n(PR-AUC %)",
    "Balanced Detection\n(F1-Score %)",
    "False Alarm Suppression\n(100 - FAR %)",
    "Pore Sizing Fidelity\n(1 - MAE/0.35 %)",
    "Parameter Compactness\n(log-scaled %)",
    "Inference Speed\n(FPS / 15 %)"
]

N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

params_map = {
    "PyroNet-2D (Tian et al. 2020)": 0.43,
    "ResNet-18 (Vision Baseline)": 11.25,
    "ConvNeXt-Tiny (Liu et al. CVPR 2022)": 27.83,
    "Swin-T (Liu et al. ICCV 2021)": 27.53,
    "Thermal-Mamba (Proposed Full)": 0.41
}

fps_map = {
    "PyroNet-2D (Tian et al. 2020)": 1425.7,
    "ResNet-18 (Vision Baseline)": 503.7,
    "ConvNeXt-Tiny (Liu et al. CVPR 2022)": 210.5,
    "Swin-T (Liu et al. ICCV 2021)": 185.2,
    "Thermal-Mamba (Proposed Full)": 438.8
}

fig = plt.figure(figsize=(10.5, 10.5), dpi=300)
ax = fig.add_axes([0.20, 0.12, 0.65, 0.65], polar=True)

# Radial labels
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

plt.xticks(angles[:-1], categories, color="#111111", size=10.5, fontweight="bold")
ax.tick_params(axis='x', pad=22)

ax.set_rlabel_position(25)
plt.yticks([20, 40, 60, 80, 100], ["20%", "40%", "60%", "80%", "100%"], color="#666666", size=9.0, fontweight="bold")
plt.ylim(0, 105)
ax.grid(True, linestyle=":", alpha=0.6, color="#bbbbbb")

for m_key, m_label, color, lw, alpha, zorder in models_to_plot:
    row = df_master[df_master["Model Architecture"] == m_key].iloc[0]

    pr = float(row["Test PR-AUC (%)"].split("±")[0].strip())
    f1 = float(row["Test F1-Score"].split("±")[0].strip()) * 100.0
    far = float(row["False Alarm Rate (%)"].split("±")[0].strip())
    spec = 100.0 - far

    mae = float(row["Pore Size MAE (mm)"].split("±")[0].strip())
    size_score = np.clip((1.0 - mae / 0.35) * 100.0, 10.0, 100.0)

    p_val = params_map[m_key]
    compact = np.clip((1.0 - np.log10(p_val + 0.1) / np.log10(35.0)) * 100.0, 10.0, 100.0)

    fps = fps_map[m_key]
    speed_score = np.clip((fps / 1500.0) * 100.0, 15.0, 100.0)

    values = [pr, f1, spec, size_score, compact, speed_score]
    values += values[:1]

    ax.plot(angles, values, color=color, linewidth=lw, linestyle="solid", label=m_label, zorder=zorder)
    ax.fill(angles, values, color=color, alpha=alpha, zorder=zorder-1)

plt.suptitle("Fig. 10. Multi-Metric Engineering Radar Benchmarking across 6 Dimensions",
             size=13.5, fontweight="bold", y=0.97, color="#111111")

# Clean Legend
ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.15), fontsize=9.8, framealpha=0.92,
          edgecolor="#cccccc")

# Inset Callout Card (Upper Left of Figure)
card_radar = (
    "Holistic Engineering Superiority:\n"
    "• Thermal-Mamba forms the widest convex polygon\n"
    "• Dominates Defect Retrieval (87.9%), F1 (0.86), and FAR (0.6%)\n"
    "• 438.8 FPS ultra-real-time throughput with 0.41M compactness"
)
fig.text(0.03, 0.11, card_radar, fontsize=8.8, fontweight="bold", color="#111111",
         bbox=dict(boxstyle="round,pad=0.45", fc="#ffffff", ec="#d95f02", lw=1.3, alpha=0.95))

plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.close()
print("SUCCESS: Fig 10 generated and saved to", out_fig)
