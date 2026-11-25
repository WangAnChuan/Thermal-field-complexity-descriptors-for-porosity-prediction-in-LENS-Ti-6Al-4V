import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve, 
                             precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb

DATA_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_datasets/2023_Ti6Al4V_Thermal_Porosity_Cropped"
OUTPUT_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_project/eda_and_tabular_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0

def load_data():
    excel_path = os.path.join(DATA_DIR, "thermal-porosity-table.xlsx")
    print(f"[LOAD] Loading data from {excel_path}...")
    df = pd.read_excel(excel_path)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    return df

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return (np.mean(x) - np.mean(y)) / (pooled_std + 1e-8)

def run_physical_eda(df):
    print("\n" + "=" * 80)
    print("STEP A.1: STATISTICAL HYPOTHESIS TESTING ON MELT POOL PHYSICAL SIGNATURES")
    print("=" * 80)

    normal_df = df[df["Porosity Label"] == 0]
    porous_df = df[df["Porosity Label"] == 1]
    
    n_norm = len(normal_df)
    n_pore = len(porous_df)
    print(f"Normal Samples (Label=0): {n_norm} ({n_norm/len(df)*100:.2f}%)")
    print(f"Porous Samples (Label=1): {n_pore} ({n_pore/len(df)*100:.2f}%)")
    print(f"Imbalance Ratio: {n_norm / n_pore:.1f} : 1")

    features = ["MP Area", "MP Eccentricity", "Tmax", "Z (mm)", "Y (mm)", "Layer"]
    stat_records = []

    for feat in features:
        val_norm = normal_df[feat].dropna().values
        val_pore = porous_df[feat].dropna().values

        # Descriptives
        m_norm, s_norm = np.mean(val_norm), np.std(val_norm, ddof=1)
        med_norm, iqr_norm = np.median(val_norm), stats.iqr(val_norm)

        m_pore, s_pore = np.mean(val_pore), np.std(val_pore, ddof=1)
        med_pore, iqr_pore = np.median(val_pore), stats.iqr(val_pore)

        # Statistical tests
        t_stat, t_pval = stats.ttest_ind(val_pore, val_norm, equal_var=False)
        u_stat, u_pval = stats.mannwhitneyu(val_pore, val_norm, alternative="two-sided")
        d = cohen_d(val_pore, val_norm)

        sig = "***" if u_pval < 0.001 else ("**" if u_pval < 0.01 else ("*" if u_pval < 0.05 else "n.s."))

        rec = {
            "Feature": feat,
            "Normal Mean±Std": f"{m_norm:.2f} ± {s_norm:.2f}",
            "Normal Median(IQR)": f"{med_norm:.2f} ({iqr_norm:.2f})",
            "Porous Mean±Std": f"{m_pore:.2f} ± {s_pore:.2f}",
            "Porous Median(IQR)": f"{med_pore:.2f} ({iqr_pore:.2f})",
            "Mann-Whitney p-val": f"{u_pval:.2e} ({sig})",
            "Welch t-val": f"{t_pval:.2e}",
            "Cohen's d": round(d, 3),
            "Direction": "Elevated in Pore" if d > 0 else "Suppressed in Pore"
        }
        stat_records.append(rec)
        print(f"  [{feat:<15}] Pore: {m_pore:8.2f} vs Norm: {m_norm:8.2f} | p-val: {u_pval:.2e} | Cohen d: {d:6.3f} | {sig}")

    stat_df = pd.DataFrame(stat_records)
    stat_csv = os.path.join(OUTPUT_DIR, "physical_features_statistical_tests.csv")
    stat_df.to_csv(stat_csv, index=False)
    print(f"\n[SAVED] Statistical hypothesis report saved to {stat_csv}")

    # Plot Fig 1: Boxplots of physical features
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=300)
    colors = ["#2b5c8f", "#d95f02"]
    
    plot_feats = [
        ("Tmax", "Peak Melt Pool Temp Tmax (°C)", axes[0]),
        ("MP Area", "Melt Pool Area (pixels)", axes[1]),
        ("MP Eccentricity", "Melt Pool Eccentricity", axes[2])
    ]

    for feat, label, ax in plot_feats:
        sns.boxplot(x="Porosity Label", y=feat, data=df, ax=ax, palette=colors, width=0.45, fliersize=2.5, boxprops=dict(alpha=0.85))
        ax.set_xticklabels(["Non-Porous (1,493)", "Porous (71)"], fontsize=11, fontweight="bold")
        ax.set_ylabel(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Annotate p-value
        u_p = float(stat_df[stat_df["Feature"] == feat]["Mann-Whitney p-val"].values[0].split()[0])
        sig_str = "*** (p < 0.001)" if u_p < 0.001 else f"p = {u_p:.3f}"
        y_max = df[feat].max()
        ax.text(0.5, 0.92, sig_str, transform=ax.transAxes, ha="center", fontsize=11, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#888888", alpha=0.9))

    plt.suptitle("Ti-6Al-4V LENS Melt Pool Thermal & Geometric Signatures by Porosity Status", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "fig1_physical_distributions.png")
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE 1] Saved: {fig1_path}")

    # Plot Fig 2: Spatial Distribution of Porosity across Layers (Y vs Z)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.scatter(normal_df["Y (mm)"], normal_df["Z (mm)"], c="#cccccc", alpha=0.4, s=15, label="Normal Deposition (N=1,493)")
    sc = ax.scatter(porous_df["Y (mm)"], porous_df["Z (mm)"], c=porous_df["Porosity Size (mm)"], 
                    cmap="plasma", s=porous_df["Porosity Size (mm)"] * 120 + 30, alpha=0.95, edgecolors="#000000", linewidth=0.8,
                    label="Porous Frame (N=71)")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("XCT Measured Pore Equivalent Diameter (mm)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Scan Position Y (mm)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Deposition Height Z (mm)", fontsize=11, fontweight="bold")
    ax.set_title("Spatial Distribution of Micro-Porosity Across 60 Deposited Layers (XCT Ground Truth)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", framealpha=0.9)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "fig2_spatiotemporal_porosity_map.png")
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE 2] Saved: {fig2_path}")

    return stat_df

def run_tabular_machine_learning_benchmarks(df):
    print("\n" + "=" * 80)
    print("STEP A.2: STRATIFIED 5-FOLD ML BENCHMARKS ON IMBALANCED PROCESS FEATURES")
    print("=" * 80)

    feature_cols = ["Layer", "Time (Sec)", "Y (mm)", "Z (mm)", "MP Area", "MP Eccentricity", "Tmax"]
    X = df[feature_cols].values
    y = df["Porosity Label"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pos_weight = (len(y) - sum(y)) / sum(y)  # ~21.0

    models = {
        "Logistic Regression (Balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000))
        ]),
        "Support Vector Machine (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, class_weight="balanced", random_state=42))
        ]),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=150, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=42
        ),
        "XGBoost (Cost-Sensitive)": xgb.XGBClassifier(
            n_estimators=150, max_depth=4, scale_pos_weight=pos_weight, learning_rate=0.05,
            random_state=42, eval_metric="logloss", n_jobs=-1
        ),
        "LightGBM (Cost-Sensitive)": lgb.LGBMClassifier(
            n_estimators=150, max_depth=4, scale_pos_weight=pos_weight, learning_rate=0.05,
            random_state=42, verbose=-1, n_jobs=-1
        )
    }

    benchmark_results = []
    curves_data = {}

    for name, model in models.items():
        oof_probs = np.zeros(len(y))
        oof_preds = np.zeros(len(y))
        
        fold_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_val = X[val_idx], y[val_idx]

            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = (probs >= 0.5).astype(int)

        # Global out-of-fold metrics
        roc = roc_auc_score(y, oof_probs)
        pr_auc = average_precision_score(y, oof_probs)
        rec = recall_score(y, oof_preds)
        prec = precision_score(y, oof_preds, zero_division=0)
        f1 = f1_score(y, oof_preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, oof_preds).ravel()
        spec = tn / (tn + fp)
        far = fp / (fp + tn)

        # Optimal F1 threshold search
        precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_th = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]

        rec_entry = {
            "Model": name,
            "ROC-AUC": round(roc, 4),
            "PR-AUC (AP)": round(pr_auc, 4),
            "Recall (Sens)": round(rec, 4),
            "Precision": round(prec, 4),
            "F1-Score (Th=0.5)": round(f1, 4),
            "Optimal F1 (Th)": f"{best_f1:.4f} ({best_th:.2f})",
            "Specificity": round(spec, 4),
            "False Alarm Rate (FAR)": round(far, 4)
        }
        benchmark_results.append(rec_entry)
        curves_data[name] = (y, oof_probs)
        print(f"  [{name:<30}] ROC-AUC: {roc*100:5.2f}% | PR-AUC: {pr_auc*100:5.2f}% | Recall: {rec*100:5.2f}% | Prec: {prec*100:5.2f}% | F1: {f1:.3f}")

    res_df = pd.DataFrame(benchmark_results)
    res_csv = os.path.join(OUTPUT_DIR, "tabular_machine_learning_benchmarks.csv")
    res_df.to_csv(res_csv, index=False)
    print(f"\n[SAVED] Benchmark metrics saved to: {res_csv}")

    # Plot Fig 3: ROC & PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    palette = sns.color_palette("tab10", len(models))

    for idx, (name, (y_true, y_prob)) in enumerate(curves_data.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, color=palette[idx], lw=2, label=f"{name} (AUC={roc:.3f})")

        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        axes[1].plot(r_curve, p_curve, color=palette[idx], lw=2, label=f"{name} (AP={pr_auc:.3f})")

    # Chance lines
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Random Guess (0.500)")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    axes[0].set_title("ROC Curves (Process-Feature Baselines)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=8.5)

    base_pr = sum(y) / len(y)
    axes[1].axhline(y=base_pr, color="k", linestyle="--", lw=1.2, alpha=0.7, label=f"Random Chance ({base_pr*100:.1f}%)")
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    axes[1].set_title("Precision-Recall Curves (21:1 Imbalance)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=8.5)

    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "fig3_baseline_roc_pr_curves.png")
    plt.savefig(fig3_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE 3] Saved: {fig3_path}")

    # Feature Importance from Random Forest & XGBoost
    rf = models["Random Forest (Balanced)"]
    xgb_m = models["XGBoost (Cost-Sensitive)"]
    
    rf_imp = rf.feature_importances_
    xgb_imp = xgb_m.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Random Forest Importance": rf_imp,
        "XGBoost Importance": xgb_imp
    }).sort_values(by="Random Forest Importance", ascending=False)
    
    imp_csv = os.path.join(OUTPUT_DIR, "feature_importance_ranking.csv")
    imp_df.to_csv(imp_csv, index=False)
    print(f"[FEATURE IMPORTANCE RANKING]:\n{imp_df.to_string(index=False)}")

    # Plot Fig 4: Feature Importance & Correlation Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    
    # Heatmap
    corr = df[feature_cols + ["Porosity Label", "Porosity Size (mm)"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=axes[0], cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
    axes[0].set_title("Thermal-Process-Defect Correlation Matrix", fontsize=11, fontweight="bold")

    # Importance bars
    imp_plot = imp_df.melt(id_vars="Feature", value_vars=["Random Forest Importance", "XGBoost Importance"], 
                           var_name="Model", value_name="Importance")
    sns.barplot(x="Importance", y="Feature", hue="Model", data=imp_plot, ax=axes[1], palette=["#2b5c8f", "#d95f02"])
    axes[1].set_title("Physical Feature Predictor Importance Ranking", fontsize=11, fontweight="bold")
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "fig4_feature_importance_and_correlation.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE 4] Saved: {fig4_path}")

    return res_df

def run_pore_size_regression_benchmark(df):
    print("\n" + "=" * 80)
    print("STEP A.3: QUANTITATIVE DEFECT SIZE (mm) REGRESSION BENCHMARK")
    print("=" * 80)

    # Filter to porous samples
    pos_df = df[df["Porosity Label"] == 1].copy()
    feature_cols = ["Layer", "Time (Sec)", "Y (mm)", "Z (mm)", "MP Area", "MP Eccentricity", "Tmax"]
    X = pos_df[feature_cols].values
    y = pos_df["Porosity Size (mm)"].values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    reg_models = {
        "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    }

    reg_results = []
    for name, model in reg_models.items():
        oof_preds = np.zeros(len(y))
        for train_idx, val_idx in kf.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            oof_preds[val_idx] = model.predict(X[val_idx])

        mae = mean_absolute_error(y, oof_preds)
        rmse = np.sqrt(mean_squared_error(y, oof_preds))
        r2 = r2_score(y, oof_preds)
        
        reg_results.append({
            "Regressor Model": name,
            "MAE (mm)": round(mae, 4),
            "RMSE (mm)": round(rmse, 4),
            "R2 Score": round(r2, 4),
            "Mean Target Size (mm)": round(float(np.mean(y)), 4)
        })
        print(f"  [{name:<30}] MAE: {mae:.4f} mm | RMSE: {rmse:.4f} mm | R2: {r2:.4f}")

    reg_df = pd.DataFrame(reg_results)
    reg_csv = os.path.join(OUTPUT_DIR, "pore_size_regression_benchmarks.csv")
    reg_df.to_csv(reg_csv, index=False)
    print(f"\n[SAVED] Pore size regression benchmark saved to: {reg_csv}")
    return reg_df

def main():
    df = load_data()
    stat_df = run_physical_eda(df)
    ml_df = run_tabular_machine_learning_benchmarks(df)
    reg_df = run_pore_size_regression_benchmark(df)

    print("\n" + "=" * 80)
    print("STEP A: CPU PHYSICAL EDA & MACHINE LEARNING BENCHMARK SUITE COMPLETED!")
    print(f"Results Directory: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
