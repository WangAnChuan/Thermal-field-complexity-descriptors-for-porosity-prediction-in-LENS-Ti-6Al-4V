import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve, 
                             precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score)

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_datasets/2023_Ti6Al4V_Thermal_Porosity_Cropped"
FRAMES_DIR = os.path.join(DATA_DIR, "extracted_frames", "cropped_thinwall_CSV_pyrometer")
OUTPUT_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_project/stageB_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Dataset & In-Memory Cache
# ---------------------------------------------------------
class ThermalPorosityDataset(Dataset):
    def __init__(self, images, physical_feats, labels, sizes):
        self.images = images
        self.physical_feats = physical_feats
        self.labels = labels
        self.sizes = sizes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "physical": self.physical_feats[idx],
            "label": self.labels[idx],
            "size": self.sizes[idx]
        }

def load_all_data():
    xlsx_path = os.path.join(DATA_DIR, "thermal-porosity-table.xlsx")
    df = pd.read_excel(xlsx_path)
    n_samples = len(df)
    print(f"[DATA LOAD] Loading {n_samples} frames from {FRAMES_DIR} into GPU/RAM cache...")

    all_images = np.zeros((n_samples, 1, 200, 201), dtype=np.float32)
    t0 = time.time()
    for idx, row in df.iterrows():
        f_num = int(row["Frame"])
        csv_file = os.path.join(FRAMES_DIR, f"Frame_{f_num}")
        # Read matrix skipping header and index column
        mat = pd.read_csv(csv_file, index_col=0).values.astype(np.float32)
        all_images[idx, 0] = mat
        if (idx + 1) % 500 == 0 or (idx + 1) == n_samples:
            print(f"  Loaded [{idx + 1}/{n_samples}] frames... ({time.time() - t0:.1f} s)")

    # Temperature normalization (0 to 1875 °C -> [0, 1])
    all_images = all_images / 2000.0

    # Physical features
    phys_cols = ["Layer", "Time (Sec)", "Y (mm)", "Z (mm)", "MP Area", "MP Eccentricity", "Tmax"]
    phys_raw = df[phys_cols].values.astype(np.float32)
    # Standardize physical features
    phys_mean = np.mean(phys_raw, axis=0)
    phys_std = np.std(phys_raw, axis=0) + 1e-6
    phys_norm = (phys_raw - phys_mean) / phys_std

    labels = df["Porosity Label"].values.astype(np.float32)
    sizes = df["Porosity Size (mm)"].values.astype(np.float32)

    images_t = torch.from_numpy(all_images)
    phys_t = torch.from_numpy(phys_norm)
    labels_t = torch.from_numpy(labels)
    sizes_t = torch.from_numpy(sizes)

    print(f"[DATA READY] Total: {n_samples} | Positives: {int(sum(labels))} ({sum(labels)/len(labels)*100:.2f}%)")
    return df, images_t, phys_t, labels_t, sizes_t

# ---------------------------------------------------------
# 2. Model Architectures
# ---------------------------------------------------------
class PyroNet2DBaseline(nn.Module):
    """
    PyroNet-style 2D CNN baseline inspired by Tian et al. (ASME JMSE 2020)
    """
    def __init__(self, in_channels=1, phys_dim=7):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))

        self.phys_fc = nn.Sequential(nn.Linear(phys_dim, 32), nn.ReLU())
        self.fc_shared = nn.Sequential(nn.Linear(256 + 32, 128), nn.ReLU(), nn.Dropout(0.3))

        self.cls_head = nn.Linear(128, 1)
        self.reg_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, phys):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv4(feat).flatten(1)

        p_feat = self.phys_fc(phys)
        fusion = torch.cat([feat, p_feat], dim=1)
        h = self.fc_shared(fusion)

        logits = self.cls_head(h).squeeze(-1)
        pred_size = self.reg_head(h).squeeze(-1)
        return logits, pred_size

class ResNetThermalBaseline(nn.Module):
    """
    Standard ResNet-18 adapted for 1-channel thermal input
    """
    def __init__(self, in_channels=1, phys_dim=7):
        super().__init__()
        from torchvision.models import resnet18
        base = resnet18(weights=None)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.phys_fc = nn.Sequential(nn.Linear(phys_dim, 64), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(512 + 64, 128), nn.ReLU(), nn.Dropout(0.3))
        self.cls_head = nn.Linear(128, 1)
        self.reg_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, phys):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        feat = self.avgpool(x).flatten(1)

        p_feat = self.phys_fc(phys)
        h = self.fc(torch.cat([feat, p_feat], dim=1))
        logits = self.cls_head(h).squeeze(-1)
        pred_size = self.reg_head(h).squeeze(-1)
        return logits, pred_size

class ThermalBiMambaBlock(nn.Module):
    """
    2D Bi-Directional State Space Block with Mamba
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        if MAMBA_AVAILABLE:
            self.mamba_fwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
            self.mamba_bwd = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.mamba_fwd = nn.GRU(dim, dim, batch_first=True)
            self.mamba_bwd = nn.GRU(dim, dim, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        # x: (B, L, D)
        norm_x = self.norm(x)
        if MAMBA_AVAILABLE:
            out_f = self.mamba_fwd(norm_x)
            out_b = self.mamba_bwd(torch.flip(norm_x, dims=[1]))
            out_b = torch.flip(out_b, dims=[1])
        else:
            out_f, _ = self.mamba_fwd(norm_x)
            out_b, _ = self.mamba_bwd(torch.flip(norm_x, dims=[1]))
            out_b = torch.flip(out_b, dims=[1])
        fused = self.proj(torch.cat([out_f, out_b], dim=-1))
        return x + fused

class ThermalMambaNet(nn.Module):
    """
    Proposed Physics-Guided Multi-Scale Visual Mamba Network (Thermal-Mamba)
    """
    def __init__(self, in_channels=1, phys_dim=7, embed_dim=96):
        super().__init__()
        # Multi-scale patch embedding
        self.stem_local = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=4, stride=4)
        self.stem_dilated = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=4, padding=1)
        self.stem_norm = nn.BatchNorm2d(embed_dim)

        # Stage 1: Conv downsample
        self.conv_stage = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # Stage 2: Bi-Directional Mamba Sequence Modeling
        self.mamba1 = ThermalBiMambaBlock(embed_dim)
        self.mamba2 = ThermalBiMambaBlock(embed_dim)

        # Physical feature cross-gating
        self.phys_proj = nn.Sequential(nn.Linear(phys_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        
        # Dual prediction heads
        self.head_norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, 1)
        self.reg_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.ReLU()  # Pore size is strictly non-negative
        )

    def forward(self, x, phys):
        B = x.shape[0]
        # Align spatial dimensions to symmetric (200, 200)
        x = F.interpolate(x, size=(200, 200), mode="bilinear", align_corners=False)
        # Multi-scale patch stem
        x1 = self.stem_local(x)
        x2 = self.stem_dilated(x)
        x = self.stem_norm(torch.cat([x1, x2], dim=1)) # (B, 96, 50, 50)
        x = self.conv_stage(x)                         # (B, 96, 25, 25)

        # Flatten to sequence: (B, 625, 96)
        B, C, H, W = x.shape
        seq = x.flatten(2).transpose(1, 2)

        # Pass through Bi-Mamba state space blocks
        seq = self.mamba1(seq)
        seq = self.mamba2(seq)

        # Global average representation
        feat = seq.mean(dim=1) # (B, 96)

        # Physical feature gating (Mechanism-Guided Gate)
        gate = self.phys_proj(phys)
        feat_gated = feat * gate + feat

        norm_feat = self.head_norm(feat_gated)
        logits = self.cls_head(norm_feat).squeeze(-1)
        pred_size = self.reg_head(norm_feat).squeeze(-1)

        return logits, pred_size

# ---------------------------------------------------------
# 3. Cost-Sensitive & Multi-Task Loss
# ---------------------------------------------------------
class MultiTaskFocalLoss(nn.Module):
    def __init__(self, pos_weight=21.0, alpha=0.8, gamma=2.0, lambda_reg=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_reg = lambda_reg

    def forward(self, logits, pred_sizes, target_labels, target_sizes):
        # 1. Focal loss for 21:1 extreme imbalance
        bce_loss = F.binary_cross_entropy_with_logits(logits, target_labels, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * target_labels + (1.0 - probs) * (1.0 - target_labels)
        alpha_t = self.alpha * target_labels + (1.0 - self.alpha) * (1.0 - target_labels)
        focal_loss = (alpha_t * (1.0 - p_t) ** self.gamma * bce_loss).mean()

        # 2. Smooth L1 regression loss only on true positive pore frames
        pos_mask = (target_labels == 1)
        if pos_mask.sum() > 0:
            reg_loss = F.smooth_l1_loss(pred_sizes[pos_mask], target_sizes[pos_mask])
        else:
            reg_loss = torch.tensor(0.0, device=logits.device)

        total_loss = focal_loss + self.lambda_reg * reg_loss
        return total_loss, focal_loss.item(), reg_loss.item()

# ---------------------------------------------------------
# 4. Stratified 5-Fold Evaluation Loop
# ---------------------------------------------------------
def train_and_eval_model(model_cls, model_name, images_t, phys_t, labels_t, sizes_t, epochs=35, batch_size=32):
    print("\n" + "=" * 75)
    print(f"BENCHMARKING MODEL: {model_name} (5-Fold Stratified Cross-Validation)")
    print("=" * 75)

    y_np = labels_t.numpy()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_probs = np.zeros(len(y_np))
    oof_sizes = np.zeros(len(y_np))
    fold_train_times = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(images_t, y_np)):
        train_ds = ThermalPorosityDataset(images_t[tr_idx], phys_t[tr_idx], labels_t[tr_idx], sizes_t[tr_idx])
        val_ds = ThermalPorosityDataset(images_t[val_idx], phys_t[val_idx], labels_t[val_idx], sizes_t[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)

        model = model_cls().to(device)
        optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        criterion = MultiTaskFocalLoss()

        best_pr_auc = 0.0
        best_fold_probs = None
        best_fold_sizes = None

        t0_fold = time.time()
        for epoch in range(1, epochs + 1):
            model.train()
            for batch in train_loader:
                imgs = batch["image"].to(device)
                phys = batch["physical"].to(device)
                lbls = batch["label"].to(device)
                szs = batch["size"].to(device)

                optimizer.zero_grad()
                logits, pred_sz = model(imgs, phys)
                loss, _, _ = criterion(logits, pred_sz, lbls, szs)
                loss.backward()
                optimizer.step()

            scheduler.step()

            # Validation
            model.eval()
            val_logits_all = []
            val_sizes_all = []
            with torch.no_grad():
                for batch in val_loader:
                    imgs = batch["image"].to(device)
                    phys = batch["physical"].to(device)
                    logits, pred_sz = model(imgs, phys)
                    val_logits_all.append(logits.cpu())
                    val_sizes_all.append(pred_sz.cpu())

            val_probs = torch.sigmoid(torch.cat(val_logits_all)).numpy()
            val_sizes = torch.cat(val_sizes_all).numpy()

            val_pr_auc = average_precision_score(y_np[val_idx], val_probs)
            if val_pr_auc > best_pr_auc:
                best_pr_auc = val_pr_auc
                best_fold_probs = val_probs
                best_fold_sizes = val_sizes
                # Save best fold 1 model as exemplar
                if fold == 0:
                    torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"{model_name}_best.pth"))

        oof_probs[val_idx] = best_fold_probs
        oof_sizes[val_idx] = best_fold_sizes
        elapsed = time.time() - t0_fold
        fold_train_times.append(elapsed)
        print(f"  Fold [{fold+1}/5] finished in {elapsed:.1f} s | Best PR-AUC: {best_pr_auc*100:.2f}%")

    # Overall Out-of-Fold Evaluation
    roc = roc_auc_score(y_np, oof_probs)
    pr_auc = average_precision_score(y_np, oof_probs)

    # Threshold optimization for F1
    precisions, recalls, thresholds = precision_recall_curve(y_np, oof_probs)
    f1_arr = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_arr)
    best_th = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_arr[best_idx]
    best_rec = recalls[best_idx]
    best_prec = precisions[best_idx]

    tn, fp, fn, tp = confusion_matrix(y_np, (oof_probs >= best_th).astype(int)).ravel()
    spec = tn / (tn + fp)
    far = fp / (fp + tn)

    # Size regression metrics on actual positive pore frames
    pos_mask = (y_np == 1)
    true_sizes = sizes_t.numpy()[pos_mask]
    pred_pos_sizes = oof_sizes[pos_mask]
    mae = mean_absolute_error(true_sizes, pred_pos_sizes)
    rmse = np.sqrt(mean_squared_error(true_sizes, pred_pos_sizes))
    r2 = r2_score(true_sizes, pred_pos_sizes)

    # Complexity profiling
    dummy_img = torch.randn(1, 1, 200, 201).to(device)
    dummy_phys = torch.randn(1, 7).to(device)
    eval_model = model_cls().to(device).eval()
    
    n_params = sum(p.numel() for p in eval_model.parameters()) / 1e6
    if THOP_AVAILABLE:
        try:
            flops, _ = profile(eval_model, inputs=(dummy_img, dummy_phys), verbose=False)
            flops_g = flops / 1e9
        except Exception:
            flops_g = 0.0
    else:
        flops_g = 0.0

    # Latency test (100 runs)
    with torch.no_grad():
        for _ in range(20):
            _ = eval_model(dummy_img, dummy_phys)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0_lat = time.time()
        for _ in range(100):
            _ = eval_model(dummy_img, dummy_phys)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        lat_ms = ((time.time() - t0_lat) / 100.0) * 1000.0
        fps = 1000.0 / lat_ms

    report = {
        "Model": model_name,
        "ROC-AUC (%)": round(roc * 100, 2),
        "PR-AUC (%)": round(pr_auc * 100, 2),
        "F1-Score": round(best_f1, 4),
        "Optimal Th": round(float(best_th), 3),
        "Recall (%)": round(best_rec * 100, 2),
        "Precision (%)": round(best_prec * 100, 2),
        "Specificity (%)": round(spec * 100, 2),
        "False Alarm Rate (FAR) (%)": round(far * 100, 2),
        "Pore Size MAE (mm)": round(mae, 4),
        "Pore Size R2": round(r2, 4),
        "Params (M)": round(n_params, 2),
        "Latency (ms)": round(lat_ms, 2),
        "Throughput (FPS)": round(fps, 1),
        "Total 5-Fold Time (s)": round(sum(fold_train_times), 1)
    }

    print(f"\n[SUMMARY {model_name}]")
    print(f"  PR-AUC: {report['PR-AUC (%)']}% | F1: {report['F1-Score']} | Recall: {report['Recall (%)']}% | Prec: {report['Precision (%)']}%")
    print(f"  Pore Size MAE: {report['Pore Size MAE (mm)']} mm | R2: {report['Pore Size R2']} | Params: {report['Params (M)']}M | FPS: {report['Throughput (FPS)']}")
    return report, (y_np, oof_probs), (true_sizes, pred_pos_sizes)

# ---------------------------------------------------------
# 5. Master Pipeline Execution
# ---------------------------------------------------------
def main():
    print("=" * 80)
    print("STAGE B: DEEP LEARNING & STATE-SPACE MAMBA BENCHMARK FOR IN-SITU POROSITY")
    print(f"Device: {device} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    df, images_t, phys_t, labels_t, sizes_t = load_all_data()

    models_to_test = [
        (PyroNet2DBaseline, "PyroNet-2D (Tian et al. 2020)"),
        (ResNetThermalBaseline, "ResNet-18 (Standard Vision Baseline)"),
        (ThermalMambaNet, "Thermal-Mamba (Proposed Architecture)")
    ]

    all_reports = []
    curves_dict = {}
    sizes_dict = {}

    for model_cls, model_name in models_to_test:
        rep, curve_data, size_data = train_and_eval_model(
            model_cls, model_name, images_t, phys_t, labels_t, sizes_t, epochs=35, batch_size=32
        )
        all_reports.append(rep)
        curves_dict[model_name] = curve_data
        sizes_dict[model_name] = size_data

    # Save summary table
    res_df = pd.DataFrame(all_reports)
    csv_out = os.path.join(OUTPUT_DIR, "stageB_dl_benchmarks.csv")
    res_df.to_csv(csv_out, index=False)
    print(f"\n" + "=" * 80)
    print("STAGE B COMPLETE! SUMMARY BENCHMARK TABLE:")
    print("=" * 80)
    print(res_df.to_string(index=False))

    # Plot Comparison Curves (Fig B1: ROC, PR, and Size Inversion Parity)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=300)
    colors = ["#2b5c8f", "#7570b3", "#d95f02"]

    # 1. ROC Curves
    for idx, (m_name, (y_true, y_prob)) in enumerate(curves_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_val = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, color=colors[idx], lw=2.2, label=f"{m_name} (AUC={roc_val*100:.1f}%)")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7)
    axes[0].set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    axes[0].set_title("ROC Curves Comparison", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=8.5)

    # 2. PR Curves
    base_pr = sum(labels_t.numpy()) / len(labels_t)
    for idx, (m_name, (y_true, y_prob)) in enumerate(curves_dict.items()):
        p_c, r_c, _ = precision_recall_curve(y_true, y_prob)
        ap_val = average_precision_score(y_true, y_prob)
        axes[1].plot(r_c, p_c, color=colors[idx], lw=2.2, label=f"{m_name} (AP={ap_val*100:.1f}%)")
    axes[1].axhline(y=base_pr, color="k", linestyle="--", lw=1.2, alpha=0.7, label=f"Random Chance ({base_pr*100:.1f}%)")
    axes[1].set_xlabel("Recall", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Precision", fontsize=11, fontweight="bold")
    axes[1].set_title("PR Curves (21:1 Extreme Imbalance)", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="upper right", fontsize=8.5)

    # 3. Size Inversion Parity Plot (Predicted vs True Pore Size mm)
    mamba_true, mamba_pred = sizes_dict["Thermal-Mamba (Proposed Architecture)"]
    axes[2].scatter(mamba_true, mamba_pred, color="#d95f02", s=45, alpha=0.85, edgecolors="#333333", label="Thermal-Mamba Predictions")
    max_val = max(np.max(mamba_true), np.max(mamba_pred)) * 1.05
    axes[2].plot([0, max_val], [0, max_val], "k--", lw=1.5, label="Perfect Parity Line (y = x)")
    m_mae = mean_absolute_error(mamba_true, mamba_pred)
    m_r2 = r2_score(mamba_true, mamba_pred)
    axes[2].text(0.05, 0.88, f"MAE = {m_mae:.4f} mm\n$R^2$ = {m_r2:.4f}", transform=axes[2].transAxes,
                 fontsize=11, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffffff", edgecolor="#aaaaaa", alpha=0.9))
    axes[2].set_xlabel("XCT Measured True Pore Size (mm)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Model Inferred Pore Size (mm)", fontsize=11, fontweight="bold")
    axes[2].set_title("Micro-Porosity Size Quantitative Inversion", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(loc="lower right", fontsize=9)

    plt.suptitle("Deep Learning & State-Space Mamba Benchmarks for In-Situ Porosity Detection & Sizing", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "fig5_stageB_deep_learning_benchmarks.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE SAVED] Comprehensive Deep Learning Figure: {fig_path}")

if __name__ == "__main__":
    main()
