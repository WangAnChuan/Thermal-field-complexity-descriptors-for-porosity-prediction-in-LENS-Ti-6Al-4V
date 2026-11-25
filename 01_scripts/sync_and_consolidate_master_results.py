#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Results Synchronization & Benchmark Consolidation Script
==============================================================
Downloads all generated CSVs and publication figures from the remote GPU server,
merges all runs into an updated MASTER SOTA & ABLATION BENCHMARK TABLE,
computes paired t-tests, and prepares updated Markdown & LaTeX tables for the paper.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

from ssh_client import download_file

REMOTE_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_project/advanced_experimental_results"
LOCAL_REPORTS = "02_eda_and_baselines/reports"
LOCAL_FIGURES = "02_eda_and_baselines/figures"
os.makedirs(LOCAL_REPORTS, exist_ok=True)
os.makedirs(LOCAL_FIGURES, exist_ok=True)

FILES_TO_SYNC = [
    ("advanced_suite_all_runs.csv", os.path.join(LOCAL_REPORTS, "advanced_suite_all_runs.csv")),
    ("advanced_suite_summary.csv", os.path.join(LOCAL_REPORTS, "advanced_suite_summary.csv")),
    ("xct_pore_severity_fracture_mechanics_summary.csv", os.path.join(LOCAL_REPORTS, "xct_pore_severity_fracture_mechanics_summary.csv")),
    ("hardware_efficiency_profiling.csv", os.path.join(LOCAL_REPORTS, "hardware_efficiency_profiling.csv")),
    ("figures/fig14_stem_loss_temporal_ablation.png", os.path.join(LOCAL_FIGURES, "fig14_stem_loss_temporal_ablation.png")),
    ("figures/fig15_xct_pore_severity_recall.png", os.path.join(LOCAL_FIGURES, "fig15_xct_pore_severity_recall.png"))
]

def sync_remote_assets():
    print("=" * 80)
    print("SYNCING ASSETS FROM REMOTE SERVER TO LOCAL WORKSPACE")
    print("=" * 80)
    for rem_rel, loc_path in FILES_TO_SYNC:
        rem_path = f"{REMOTE_DIR}/{rem_rel}"
        try:
            download_file(rem_path, loc_path)
            print(f"[SYNC OK] {rem_rel} -> {loc_path}")
        except Exception as e:
            print(f"[SYNC FAILED / PENDING] {rem_rel}: {e}")

def consolidate_master_benchmark():
    print("\n" + "=" * 80)
    print("CONSOLIDATING COMPREHENSIVE SOTA & ABLATION MASTER TABLE")
    print("=" * 80)

    # 1. Load original 40 runs
    orig_runs_p = os.path.join(LOCAL_REPORTS, "all_models_40runs_raw.csv")
    adv_runs_p = os.path.join(LOCAL_REPORTS, "advanced_suite_all_runs.csv")

    if not os.path.exists(orig_runs_p) or not os.path.exists(adv_runs_p):
        print(f"[ERROR] Missing runs files: {orig_runs_p} or {adv_runs_p}")
        return

    df_orig = pd.read_csv(orig_runs_p)
    df_adv = pd.read_csv(adv_runs_p)

    # Ensure consistent column naming
    combined_df = pd.concat([df_orig, df_adv], ignore_index=True)
    combined_csv = os.path.join(LOCAL_REPORTS, "all_models_expanded_runs_raw.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"[SAVE] Raw combined runs ({len(combined_df)} records) saved to: {combined_csv}")

    # Extract reference Proposed Full runs for paired t-tests
    ref_sub = combined_df[combined_df["Model"] == "Thermal-Mamba (Proposed Full)"]
    if ref_sub.empty:
        # Fallback to 3-task or existing
        ref_sub = combined_df[combined_df["Model"].str.contains("Thermal-Mamba")].head(5)
    
    ref_seeds = ref_sub["Seed"].values
    ref_pr = ref_sub.set_index("Seed")["Test PR-AUC (%)"]

    MODEL_DISPLAY_ORDER = [
        # Domain Baselines
        "PyroNet-2D (Tian et al. 2020)",
        "CNN-BiLSTM (LRCN Temporal Baseline)",
        # Vision Baselines & Modern SOTA
        "ResNet-18 (Vision Baseline)",
        "EfficientNet-B0 (Tan & Le ICML 2019)",
        "ConvNeXt-Tiny (Liu et al. CVPR 2022)",
        "Swin-T (Liu et al. ICCV 2021)",
        # Fine-grained Ablations
        "Ablation: Single-Scale Stem",
        "Ablation: Naive Concat Fusion",
        "Ablation: Standard BCE Loss",
        "Ablation: Weighted BCE Loss",
        "Ablation: w/o Bi-Scan (1D Mamba)",
        "Ablation: w/o Physical Gating",
        # Proposed Architectures
        "Thermal-Mamba (3-Task with Localization)",
        "Thermal-Mamba (Proposed Full)"
    ]

    MODEL_PARAMS_MAP = {
        "PyroNet-2D (Tian et al. 2020)": 0.43,
        "CNN-BiLSTM (LRCN Temporal Baseline)": 0.58,
        "ResNet-18 (Vision Baseline)": 11.25,
        "EfficientNet-B0 (Tan & Le ICML 2019)": 4.02,
        "ConvNeXt-Tiny (Liu et al. CVPR 2022)": 27.83,
        "Swin-T (Liu et al. ICCV 2021)": 27.53,
        "Ablation: Single-Scale Stem": 0.39,
        "Ablation: Naive Concat Fusion": 0.42,
        "Ablation: Standard BCE Loss": 0.41,
        "Ablation: Weighted BCE Loss": 0.41,
        "Ablation: w/o Bi-Scan (1D Mamba)": 0.28,
        "Ablation: w/o Physical Gating": 0.33,
        "Thermal-Mamba (3-Task with Localization)": 0.42,
        "Thermal-Mamba (Proposed Full)": 0.41
    }

    summary_rows = []
    for m in MODEL_DISPLAY_ORDER:
        sub = combined_df[combined_df["Model"] == m]
        if len(sub) == 0:
            continue

        pr_vals = sub["Test PR-AUC (%)"].values
        roc_vals = sub["Test ROC-AUC (%)"].values
        f1_vals = sub["Test F1-Score"].values
        prec_vals = sub["Test Precision (%)"].values
        rec_vals = sub["Test Recall (%)"].values
        far_vals = sub["Test FAR (%)"].values
        mae_vals = sub["Pore Size MAE (mm)"].values
        r2_vals = sub["Pore Size R2"].values

        # Paired t-test
        if m != "Thermal-Mamba (Proposed Full)":
            sub_indexed = sub.set_index("Seed")["Test PR-AUC (%)"]
            common_seeds = list(set(ref_pr.index) & set(sub_indexed.index))
            if len(common_seeds) >= 3:
                v1 = [ref_pr.loc[s] for s in common_seeds]
                v2 = [sub_indexed.loc[s] for s in common_seeds]
                _, p_pr = stats.ttest_rel(v1, v2)
                sig_pr = "***" if p_pr < 0.001 else ("**" if p_pr < 0.01 else ("*" if p_pr < 0.05 else "n.s."))
                p_val_str = f"p = {p_pr:.4f} ({sig_pr})"
            else:
                p_val_str = "n/a"
        else:
            p_val_str = "Ref (Ours)"

        row = {
            "Model Architecture": m,
            "Params (M)": MODEL_PARAMS_MAP.get(m, 0.41),
            "Test PR-AUC (%)": f"{np.mean(pr_vals):.2f} ± {np.std(pr_vals, ddof=1):.2f}",
            "Paired t-test (vs Ours)": p_val_str,
            "Test ROC-AUC (%)": f"{np.mean(roc_vals):.2f} ± {np.std(roc_vals, ddof=1):.2f}",
            "Test F1-Score": f"{np.mean(f1_vals):.4f} ± {np.std(f1_vals, ddof=1):.4f}",
            "Precision (%)": f"{np.mean(prec_vals):.2f} ± {np.std(prec_vals, ddof=1):.2f}",
            "Recall (%)": f"{np.mean(rec_vals):.2f} ± {np.std(rec_vals, ddof=1):.2f}",
            "False Alarm Rate (%)": f"{np.mean(far_vals):.2f} ± {np.std(far_vals, ddof=1):.2f}",
            "Pore Size MAE (mm)": f"{np.mean(mae_vals):.4f} ± {np.std(mae_vals, ddof=1):.4f}",
            "Pore Size R2": f"{np.mean(r2_vals):.4f} ± {np.std(r2_vals, ddof=1):.4f}"
        }
        summary_rows.append(row)

    sum_df = pd.DataFrame(summary_rows)
    master_table_p = os.path.join(LOCAL_REPORTS, "COMPREHENSIVE_MASTER_SOTA_ABLATION_TABLE.csv")
    sum_df.to_csv(master_table_p, index=False)
    print(f"\n[SAVED] Comprehensive Master Table saved to: {master_table_p}")
    print(sum_df.to_string(index=False))

    # Also output formatted Markdown table
    md_table = sum_df.to_markdown(index=False)
    md_path = os.path.join(LOCAL_REPORTS, "MASTER_TABLE.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Master SOTA & Ablation Benchmark Table\n\n")
        f.write(md_table + "\n")
    print(f"[SAVED] Markdown Table saved to: {md_path}")

if __name__ == "__main__":
    sync_remote_assets()
    consolidate_master_benchmark()
