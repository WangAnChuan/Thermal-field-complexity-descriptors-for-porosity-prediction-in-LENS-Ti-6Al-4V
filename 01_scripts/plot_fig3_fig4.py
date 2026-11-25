import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.1

df = pd.read_excel('02_eda_and_baselines/thermal-porosity-table.xlsx')
feature_cols = ['Time (Sec)', 'Y (mm)', 'Z (mm)', 'Layer', 'MP Area', 'MP Eccentricity', 'Tmax']
X = df[feature_cols].values
y = df['Porosity Label'].values

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    'Support Vector Machine': SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42),
    'XGBoost (Cost-Sensitive)': XGBClassifier(scale_pos_weight=21, random_state=42, eval_metric='logloss'),
    'LightGBM (Cost-Sensitive)': LGBMClassifier(scale_pos_weight=21, random_state=42, verbose=-1)
}

color_map = {
    'Logistic Regression': '#7f7f7f',
    'Support Vector Machine': '#7570b3',
    'Random Forest': '#2b5c8f',
    'HistGradientBoosting': '#4575b4',
    'XGBoost (Cost-Sensitive)': '#d95f02',
    'LightGBM (Cost-Sensitive)': '#1b9e77'
}

curves = {}
for name, clf in models.items():
    oof_probs = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        clf.fit(X_train, y_train)
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
    curves[name] = oof_probs

# ----------------- Fig 3: ROC & PR -----------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6.0), dpi=300)

# (a) ROC
ax = axes[0]
for name, probs in curves.items():
    fpr, tpr, _ = roc_curve(y, probs)
    auc = roc_auc_score(y, probs)
    ax.plot(fpr, tpr, color=color_map[name], lw=2.2, label=f'{name} (AUC = {auc*100:.1f}%)')

ax.plot([0, 1], [0, 1], linestyle='--', color='#666666', lw=1.2, label='Random Guess (50.0%)')
ax.set_title('(a) Receiver Operating Characteristic (ROC)\n[5-Fold Cross-Validation on In-Situ Process Features]', fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10.5, fontweight='bold')
ax.set_ylabel('True Positive Rate (Recall / Sensitivity)', fontsize=10.5, fontweight='bold')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.04)
ax.grid(True, linestyle=':', alpha=0.5, color='#cccccc')
ax.legend(loc='lower right', fontsize=9.2, framealpha=0.92, edgecolor='#cccccc')

# (b) PR
ax = axes[1]
for name, probs in curves.items():
    prec, rec, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    ax.plot(rec, prec, color=color_map[name], lw=2.2, label=f'{name} (AP = {ap*100:.1f}%)')

prev = np.mean(y)
ax.axhline(prev, linestyle='--', color='#666666', lw=1.2, label=f'Random Chance ({prev*100:.1f}%)')

card_pr = (
    'Class Skew: 21:1 (Prevalence: 4.54%)\n'
    '• Tree Ensembles (LGBM/XGB/HGB) reach AP 80~82%\n'
    '• In-situ scalar features show strong initial discrimination\n'
    '• 2D spatial gradients required to resolve ambiguity band'
)
ax.text(0.04, 0.15, card_pr, transform=ax.transAxes, fontsize=8.8, fontweight='bold', color='#111111',
        bbox=dict(boxstyle='round,pad=0.5', fc='#ffffff', ec='#d95f02', lw=1.2, alpha=0.95))

ax.set_title('(b) Precision-Recall Curves (21:1 Extreme Imbalance)\n[Evaluating Defect Retrieval under Severe Class Skew]', fontsize=11, fontweight='bold', pad=10)
ax.set_xlabel('Recall (True Positive Rate)', fontsize=10.5, fontweight='bold')
ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=10.5, fontweight='bold')
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.04)
ax.grid(True, linestyle=':', alpha=0.5, color='#cccccc')
ax.legend(loc='upper right', fontsize=9.2, framealpha=0.92, edgecolor='#cccccc')

plt.suptitle('Fig. 3. 5-Fold Stratified Machine Learning Baselines on In-Situ Process Features',
             fontsize=13.5, fontweight='bold', y=0.98, color='#111111')
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_p3 = '02_eda_and_baselines/figures/fig3_baseline_roc_pr_curves.png'
plt.savefig(out_p3, dpi=300, bbox_inches='tight')
plt.close()
print('SUCCESS: Fig 3 generated at', out_p3)

# ----------------- Fig 4: Correlation & Feature Importance -----------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), dpi=300)

# (a) Correlation Heatmap
corr_cols = ['Tmax', 'MP Area', 'MP Eccentricity', 'Layer', 'Z (mm)', 'Y (mm)', 'Porosity Label', 'Porosity Size (mm)']
corr_labels = ['Tmax', 'MP Area', 'MP Eccent.', 'Layer', 'Z (mm)', 'Y (mm)', 'Defect Label', 'Pore Size (mm)']
corr = df[corr_cols].corr()

ax = axes[0]
cax = ax.matshow(corr, cmap='vlag', vmin=-1.0, vmax=1.0)
cbar = fig.colorbar(cax, ax=ax, shrink=0.82, pad=0.03)
cbar.set_label('Pearson Correlation Coefficient $r$', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

for i in range(len(corr_cols)):
    for j in range(len(corr_cols)):
        val = corr.iloc[i, j]
        txt_color = 'white' if abs(val) > 0.45 else '#111111'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=9.2, fontweight='bold', color=txt_color)

ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_labels, rotation=35, ha='left', fontsize=9.5, fontweight='bold')
ax.set_yticklabels(corr_labels, fontsize=9.5, fontweight='bold')
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
ax.set_title('(a) Thermal-Process-Porosity Correlation Matrix\n[Pairwise Linear Dependencies across Process Descriptors]',
             fontsize=11, fontweight='bold', pad=15)

# (b) Feature Importance
ax = axes[1]
imp_df = pd.read_csv('02_eda_and_baselines/reports/feature_importance_ranking.csv')
imp_df = imp_df.sort_values(by='Random Forest Importance', ascending=True)

y_pos = np.arange(len(imp_df))
bar_h = 0.35
ax.barh(y_pos - bar_h/2, imp_df['Random Forest Importance'], height=bar_h, color='#2b5c8f', edgecolor='#111111', lw=0.9, label='Random Forest (Balanced)')
ax.barh(y_pos + bar_h/2, imp_df['XGBoost Importance'], height=bar_h, color='#d95f02', edgecolor='#111111', lw=0.9, label='XGBoost (Cost-Sensitive)')

ax.set_yticks(y_pos)
ax.set_yticklabels(imp_df['Feature'], fontsize=9.5, fontweight='bold')
ax.set_xlabel('Predictor Gini Feature Importance', fontsize=10.5, fontweight='bold')
ax.set_title('(b) Physical Predictor Relative Importance Ranking\n[Tree-Based Feature Sensitivity to Void Nucleation]',
             fontsize=11, fontweight='bold', pad=15)
ax.grid(axis='x', linestyle=':', alpha=0.5, color='#cccccc')
ax.legend(loc='lower right', fontsize=9.5, framealpha=0.92, edgecolor='#cccccc')

card_imp = (
    'Dominant Predictors: Time (Sec) & MP Eccentricity\n'
    '• Time captures cumulative heat accumulation across build\n'
    '• Eccentricity captures melt pool tear during rapid moves\n'
    '• Layer & Z exhibit collinear cold-baseplate conductive sink'
)
ax.text(0.04, 0.45, card_imp, transform=ax.transAxes, fontsize=8.8, fontweight='bold', color='#111111',
        bbox=dict(boxstyle='round,pad=0.5', fc='#ffffff', ec='#2b5c8f', lw=1.2, alpha=0.95))

plt.suptitle('Fig. 4. Multimodal Process-Feature Interaction & Predictor Importance Landscape',
             fontsize=13.5, fontweight='bold', y=0.98, color='#111111')
plt.tight_layout(rect=[0, 0, 1, 0.95])
out_p4 = '02_eda_and_baselines/figures/fig4_feature_importance_and_correlation.png'
plt.savefig(out_p4, dpi=300, bbox_inches='tight')
plt.close()
print('SUCCESS: Fig 4 generated at', out_p4)
