import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_project"
RUNS_5SEED_CSV = os.path.join(PROJECT_DIR, "5seed_rigorous_results", "rigorous_5seed_all_runs.csv")
SOTA_CSV = os.path.join(PROJECT_DIR, "sota_expanded_results", "sota_expanded_5seed_runs.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "master_sota_benchmark")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0

MODEL_ORDER = [
    # 1. Classical Baselines
    "PyroNet-2D (Tian et al. 2020)",
    "ResNet-18 (Vision Baseline)",
    # 2. Modern Vision SOTA
    "EfficientNet-B0 (Tan & Le ICML 2019)",
    "ConvNeXt-Tiny (Liu et al. CVPR 2022)",
    "Swin-T (Liu et al. ICCV 2021)",
    # 3. Ablation Variants
    "Ablation: w/o Bi-Scan (1D Mamba)",
    "Ablation: w/o Physical Gating",
    # 4. Proposed Architecture
    "Thermal-Mamba (Proposed Full)"
]

MODEL_PARAMS = {
    "PyroNet-2D (Tian et al. 2020)": 0.43,
    "ResNet-18 (Vision Baseline)": 11.25,
    "EfficientNet-B0 (Tan & Le ICML 2019)": 4.02,
    "ConvNeXt-Tiny (Liu et al. CVPR 2022)": 27.83,
    "Swin-T (Liu et al. ICCV 2021)": 27.53,
    "Ablation: w/o Bi-Scan (1D Mamba)": 0.28,
    "Ablation: w/o Physical Gating": 0.33,
    "Thermal-Mamba (Proposed Full)": 0.41
}

def main():
    print("=" * 80)
    print("MERGING ALL RIGOROUS 5-SEED RUNS: MAMBA, BASELINES & EXPANDED SOTA")
    print("=" * 80)

    if not os.path.exists(RUNS_5SEED_CSV) or not os.path.exists(SOTA_CSV):
        print(f"[WAIT] One or both files missing: {RUNS_5SEED_CSV}, {SOTA_CSV}")
        sys.exit(1)

    df1 = pd.read_csv(RUNS_5SEED_CSV)
    df2 = pd.read_csv(SOTA_CSV)

    all_df = pd.concat([df1, df2], ignore_index=True)
    all_runs_out = os.path.join(OUTPUT_DIR, "all_models_40runs_raw.csv")
    all_df.to_csv(all_runs_out, index=False)
    print(f"[SAVED] Combined 40 runs saved to: {all_runs_out}")

    # Compute Statistics
    mamba_pr = all_df[all_df["Model"] == "Thermal-Mamba (Proposed Full)"]["Test PR-AUC (%)"].values

    summary_rows = []
    for m in MODEL_ORDER:
        sub = all_df[all_df["Model"] == m]
        if len(sub) == 0:
            continue

        pr_vals = sub["Test PR-AUC (%)"].values
        roc_vals = sub["Test ROC-AUC (%)"].values
        f1_vals = sub["Test F1-Score"].values
        prec_vals = sub["Test Precision (%)"].values
        rec_vals = sub["Test Recall (%)"].values
        spec_vals = sub["Test Specificity (%)"].values
        far_vals = sub["Test FAR (%)"].values
        mae_vals = sub["Pore Size MAE (mm)"].values
        r2_vals = sub["Pore Size R2"].values

        if m != "Thermal-Mamba (Proposed Full)":
            _, p_pr = stats.ttest_rel(mamba_pr, pr_vals)
            sig_pr = "***" if p_pr < 0.001 else ("**" if p_pr < 0.01 else ("*" if p_pr < 0.05 else "n.s."))
            p_val_str = f"p = {p_pr:.4f} ({sig_pr})"
        else:
            p_val_str = "Ref (Ours)"

        row = {
            "Model Architecture": m,
            "Params (M)": MODEL_PARAMS.get(m, 0.0),
            "Test PR-AUC (%)": f"{np.mean(pr_vals):.2f} ± {np.std(pr_vals, ddof=1):.2f}",
            "Paired t-test (vs Ours)": p_val_str,
            "Test ROC-AUC (%)": f"{np.mean(roc_vals):.2f} ± {np.std(roc_vals, ddof=1):.2f}",
            "Test F1-Score": f"{np.mean(f1_vals):.4f} ± {np.std(f1_vals, ddof=1):.4f}",
            "Precision (%)": f"{np.mean(prec_vals):.2f} ± {np.std(prec_vals, ddof=1):.2f}",
            "Recall (%)": f"{np.mean(rec_vals):.2f} ± {np.std(rec_vals, ddof=1):.2f}",
            "Specificity (%)": f"{np.mean(spec_vals):.2f} ± {np.std(spec_vals, ddof=1):.2f}",
            "False Alarm Rate (%)": f"{np.mean(far_vals):.2f} ± {np.std(far_vals, ddof=1):.2f}",
            "Pore Size MAE (mm)": f"{np.mean(mae_vals):.4f} ± {np.std(mae_vals, ddof=1):.4f}",
            "Pore Size R2": f"{np.mean(r2_vals):.4f} ± {np.std(r2_vals, ddof=1):.4f}"
        }
        summary_rows.append(row)

    sum_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(OUTPUT_DIR, "MASTER_SOTA_BENCHMARK_TABLE.csv")
    sum_df.to_csv(sum_csv, index=False)
    print("\n" + "=" * 80)
    print("MASTER SOTA BENCHMARK TABLE (MEAN ± STD, 5 SEEDS):")
    print("=" * 80)
    print(sum_df.to_string(index=False))

    # ---------------------------------------------------------
    # Comprehensive Publication Figure (Fig 8)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), dpi=300)

    # Subplot 1: All-Model PR-AUC Boxplot
    clean_names = {
        "PyroNet-2D (Tian et al. 2020)": "PyroNet",
        "ResNet-18 (Vision Baseline)": "ResNet-18",
        "EfficientNet-B0 (Tan & Le ICML 2019)": "EffNet-B0",
        "ConvNeXt-Tiny (Liu et al. CVPR 2022)": "ConvNeXt-T",
        "Swin-T (Liu et al. ICCV 2021)": "Swin-T",
        "Ablation: w/o Bi-Scan (1D Mamba)": "1D-Mamba",
        "Ablation: w/o Physical Gating": "Mamba-w/o-Gate",
        "Thermal-Mamba (Proposed Full)": "Thermal-Mamba"
    }
    all_df["ShortName"] = all_df["Model"].map(clean_names)
    short_order = [clean_names[m] for m in MODEL_ORDER if m in clean_names]

    pal = sns.color_palette("muted", len(short_order))
    pal[-1] = "#d95f02" # Highlight Thermal-Mamba in bold orange

    sns.boxplot(x="ShortName", y="Test PR-AUC (%)", data=all_df, order=short_order, ax=axes[0], palette=pal, width=0.45)
    sns.stripplot(x="ShortName", y="Test PR-AUC (%)", data=all_df, order=short_order, ax=axes[0], color="#222222", size=5, jitter=0.15)
    axes[0].set_xticklabels(short_order, rotation=35, ha="right", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Test PR-AUC (%) Across 5 Seeds", fontsize=11, fontweight="bold")
    axes[0].set_title("PR-AUC Comparison across SOTA & Baselines", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Subplot 2: Pareto Frontier (PR-AUC vs Parameters)
    pareto_data = []
    for m in MODEL_ORDER:
        sub = all_df[all_df["Model"] == m]
        if len(sub) > 0:
            pareto_data.append({
                "Model": clean_names[m],
                "Params": MODEL_PARAMS[m],
                "PR_AUC": np.mean(sub["Test PR-AUC (%)"].values),
                "MAE": np.mean(sub["Pore Size MAE (mm)"].values)
            })
    p_df = pd.DataFrame(pareto_data)

    for _, row in p_df.iterrows():
        is_ours = "Thermal-Mamba" in row["Model"]
        c = "#d95f02" if is_ours else "#2b5c8f"
        s = 140 if is_ours else 70
        axes[1].scatter(row["Params"], row["PR_AUC"], color=c, s=s, edgecolors="#000000", lw=1.2, zorder=5)
        offset = (5, 5) if not is_ours else (-30, 8)
        axes[1].annotate(row["Model"], (row["Params"], row["PR_AUC"]), xytext=offset, textcoords="offset points",
                         fontsize=8.5, fontweight="bold" if is_ours else "normal")

    axes[1].set_xscale("log")
    axes[1].set_xlabel("Model Parameters (Millions, Log Scale)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Mean Test PR-AUC (%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Pareto Efficiency: PR-AUC vs Model Complexity", fontsize=11, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    # Subplot 3: Pore Size MAE (mm) Inversion Accuracy
    sns.barplot(x="ShortName", y="Pore Size MAE (mm)", data=all_df, order=short_order, ax=axes[2], palette=pal, capsize=0.1)
    axes[2].set_xticklabels(short_order, rotation=35, ha="right", fontsize=9, fontweight="bold")
    axes[2].set_ylabel("Pore Size MAE (mm) [Lower is Better]", fontsize=11, fontweight="bold")
    axes[2].set_title("Quantitative Micro-Porosity Sizing Error", fontsize=11, fontweight="bold")
    axes[2].grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Master Empirical SOTA Benchmark for Ti-6Al-4V LENS In-Situ Porosity Detection & Sizing", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_out = os.path.join(FIGURES_DIR, "fig8_master_sota_comprehensive_benchmark.png")
    plt.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE SAVED] Master SOTA Figure saved to: {fig_out}")

if __name__ == "__main__":
    main()
