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

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve, 
                             precision_recall_curve, mean_absolute_error, mean_squared_error, r2_score)

import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_datasets/2023_Ti6Al4V_Thermal_Porosity_Cropped"
FRAMES_DIR = os.path.join(DATA_DIR, "extracted_frames", "cropped_thinwall_CSV_pyrometer")
OUTPUT_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_project/sota_expanded_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Dataset & In-Memory Loading
# ---------------------------------------------------------
class ThermalDataset(Dataset):
    def __init__(self, images, physical, labels, sizes):
        self.images = images
        self.physical = physical
        self.labels = labels
        self.sizes = sizes

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "physical": self.physical[idx],
            "label": self.labels[idx],
            "size": self.sizes[idx]
        }

def load_data():
    xlsx_p = os.path.join(DATA_DIR, "thermal-porosity-table.xlsx")
    df = pd.read_excel(xlsx_p)
    n = len(df)
    print(f"[DATA] Loading {n} frames from {FRAMES_DIR} into RAM...")

    all_imgs = np.zeros((n, 1, 200, 201), dtype=np.float32)
    for i, row in df.iterrows():
        f_num = int(row["Frame"])
        csv_file = os.path.join(FRAMES_DIR, f"Frame_{f_num}")
        mat = pd.read_csv(csv_file, index_col=0).values.astype(np.float32)
        all_imgs[i, 0] = mat

    all_imgs = all_imgs / 2000.0

    phys_cols = ["Layer", "Time (Sec)", "Y (mm)", "Z (mm)", "MP Area", "MP Eccentricity", "Tmax"]
    phys_raw = df[phys_cols].values.astype(np.float32)
    phys_norm = (phys_raw - np.mean(phys_raw, axis=0)) / (np.std(phys_raw, axis=0) + 1e-6)

    labels = df["Porosity Label"].values.astype(np.float32)
    sizes = df["Porosity Size (mm)"].values.astype(np.float32)

    return df, torch.from_numpy(all_imgs), torch.from_numpy(phys_norm), torch.from_numpy(labels), torch.from_numpy(sizes)

# ---------------------------------------------------------
# 2. SOTA Model Architectures
# ---------------------------------------------------------
class SwinThermalSOTA(nn.Module):
    """
    Swin Transformer Tiny (Liu et al., ICCV 2021) adapted for In-Situ Melt Pool Monitoring
    """
    def __init__(self, phys_dim=7):
        super().__init__()
        base = models.swin_t(weights=None)
        self.patch_embed = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        self.features = base.features[1:]
        self.norm = base.norm
        self.phys_fc = nn.Sequential(nn.Linear(phys_dim, 64), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(768 + 64, 128), nn.ReLU(), nn.Dropout(0.3))
        self.cls_head = nn.Linear(128, 1)
        self.reg_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.ReLU())

    def forward(self, x, phys):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.patch_embed(x).permute(0, 2, 3, 1) # (B, 56, 56, 96)
        x = self.features(x)
        x = self.norm(x)
        feat = x.mean(dim=[1, 2])
        p = self.phys_fc(phys)
        h = self.fc(torch.cat([feat, p], dim=1))
        return self.cls_head(h).squeeze(-1), self.reg_head(h).squeeze(-1)

class ConvNeXtThermalSOTA(nn.Module):
    """
    ConvNeXt Tiny (Liu et al., CVPR 2022) adapted for In-Situ Melt Pool Monitoring
    """
    def __init__(self, phys_dim=7):
        super().__init__()
        base = models.convnext_tiny(weights=None)
        self.stem = nn.Conv2d(1, 96, kernel_size=4, stride=4)
        self.features = base.features[1:]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.phys_fc = nn.Sequential(nn.Linear(phys_dim, 64), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(768 + 64, 128), nn.ReLU(), nn.Dropout(0.3))
        self.cls_head = nn.Linear(128, 1)
        self.reg_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.ReLU())

    def forward(self, x, phys):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.stem(x)
        x = self.features(x)
        feat = self.avgpool(x).flatten(1)
        p = self.phys_fc(phys)
        h = self.fc(torch.cat([feat, p], dim=1))
        return self.cls_head(h).squeeze(-1), self.reg_head(h).squeeze(-1)

class EfficientNetThermalSOTA(nn.Module):
    """
    EfficientNet-B0 (Tan & Le, ICML 2019) adapted for In-Situ Melt Pool Monitoring
    """
    def __init__(self, phys_dim=7):
        super().__init__()
        base = models.efficientnet_b0(weights=None)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = base.features[1:]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.phys_fc = nn.Sequential(nn.Linear(phys_dim, 32), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(1280 + 32, 128), nn.ReLU(), nn.Dropout(0.3))
        self.cls_head = nn.Linear(128, 1)
        self.reg_head = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1), nn.ReLU())

    def forward(self, x, phys):
        x = self.conv1(x)
        x = self.features(x)
        feat = self.avgpool(x).flatten(1)
        p = self.phys_fc(phys)
        h = self.fc(torch.cat([feat, p], dim=1))
        return self.cls_head(h).squeeze(-1), self.reg_head(h).squeeze(-1)

# ---------------------------------------------------------
# 3. Training & Evaluation Core
# ---------------------------------------------------------
class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0, lambda_reg=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_reg = lambda_reg

    def forward(self, logits, pred_sizes, target_labels, target_sizes):
        bce = F.binary_cross_entropy_with_logits(logits, target_labels, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * target_labels + (1.0 - probs) * (1.0 - target_labels)
        alpha_t = self.alpha * target_labels + (1.0 - self.alpha) * (1.0 - target_labels)
        focal = (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()

        pos_mask = (target_labels == 1)
        if pos_mask.sum() > 0:
            reg = F.smooth_l1_loss(pred_sizes[pos_mask], target_sizes[pos_mask])
        else:
            reg = torch.tensor(0.0, device=logits.device)

        return focal + self.lambda_reg * reg

def evaluate_loader(model, loader):
    model.eval()
    logits_list, sizes_list, lbls_list, szs_list = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            phys = batch["physical"].to(device)
            lg, ps = model(imgs, phys)
            logits_list.append(lg.cpu())
            sizes_list.append(ps.cpu())
            lbls_list.append(batch["label"])
            szs_list.append(batch["size"])

    probs = torch.sigmoid(torch.cat(logits_list)).numpy()
    pred_sz = torch.cat(sizes_list).numpy()
    y_true = torch.cat(lbls_list).numpy()
    sz_true = torch.cat(szs_list).numpy()

    roc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)

    p_c, r_c, ths = precision_recall_curve(y_true, probs)
    f1s = 2 * (p_c * r_c) / (p_c + r_c + 1e-8)
    b_idx = np.argmax(f1s)
    b_th = ths[b_idx] if b_idx < len(ths) else 0.5
    f1 = f1s[b_idx]
    prec = p_c[b_idx]
    rec = r_c[b_idx]

    tn, fp, fn, tp = confusion_matrix(y_true, (probs >= b_th).astype(int)).ravel()
    spec = tn / (tn + fp)
    far = fp / (fp + tn)

    pos_m = (y_true == 1)
    if pos_m.sum() > 1:
        mae = mean_absolute_error(sz_true[pos_m], pred_sz[pos_m])
        r2 = r2_score(sz_true[pos_m], pred_sz[pos_m])
    else:
        mae, r2 = 0.0, 0.0

    return {
        "roc_auc": roc * 100,
        "pr_auc": pr_auc * 100,
        "f1": f1,
        "optimal_th": b_th,
        "recall": rec * 100,
        "precision": prec * 100,
        "specificity": spec * 100,
        "far": far * 100,
        "size_mae": mae,
        "size_r2": r2
    }

def train_single_run(model_fn, train_loader, val_loader, test_loader, epochs=35, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = model_fn().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = MultiTaskLoss()

    best_val_pr = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            imgs = batch["image"].to(device)
            phys = batch["physical"].to(device)
            lbls = batch["label"].to(device)
            szs = batch["size"].to(device)

            optimizer.zero_grad()
            lg, ps = model(imgs, phys)
            loss = criterion(lg, ps, lbls, szs)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_m = evaluate_loader(model, val_loader)
        if val_m["pr_auc"] > best_val_pr:
            best_val_pr = val_m["pr_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    test_m = evaluate_loader(model, test_loader)
    return test_m

# ---------------------------------------------------------
# 4. Master SOTA Pipeline Execution
# ---------------------------------------------------------
def main():
    print("=" * 80)
    print("EXPANDED SOTA SUITE: SWIN-T, CONVNEXT-T, EFFICIENTNET-B0 ACROSS 5 SEEDS")
    print(f"Hardware: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    df, images_t, phys_t, labels_t, sizes_t = load_data()
    y_np = labels_t.numpy()

    SEEDS = [42, 100, 200, 300, 400]

    SOTA_MODELS = [
        ("Swin-T (Liu et al. ICCV 2021)", lambda: SwinThermalSOTA()),
        ("ConvNeXt-Tiny (Liu et al. CVPR 2022)", lambda: ConvNeXtThermalSOTA()),
        ("EfficientNet-B0 (Tan & Le ICML 2019)", lambda: EfficientNetThermalSOTA())
    ]

    all_records = []

    for seed in SEEDS:
        print(f"\n" + "#" * 70)
        print(f"[SOTA SUITE] RUNNING SEED: {seed}")
        print("#" * 70)

        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
        train_idx, temp_idx = next(sss1.split(images_t, y_np))

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
        val_sub, test_sub = next(sss2.split(images_t[temp_idx], y_np[temp_idx]))
        val_idx = temp_idx[val_sub]
        test_idx = temp_idx[test_sub]

        train_ds = ThermalDataset(images_t[train_idx], phys_t[train_idx], labels_t[train_idx], sizes_t[train_idx])
        val_ds = ThermalDataset(images_t[val_idx], phys_t[val_idx], labels_t[val_idx], sizes_t[val_idx])
        test_ds = ThermalDataset(images_t[test_idx], phys_t[test_idx], labels_t[test_idx], sizes_t[test_idx])

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True)

        for m_name, m_fn in SOTA_MODELS:
            t0 = time.time()
            res = train_single_run(m_fn, train_loader, val_loader, test_loader, epochs=35, seed=seed)
            dur = time.time() - t0

            rec = {
                "Seed": seed,
                "Model": m_name,
                "Test PR-AUC (%)": round(res["pr_auc"], 2),
                "Test ROC-AUC (%)": round(res["roc_auc"], 2),
                "Test F1-Score": round(res["f1"], 4),
                "Test Precision (%)": round(res["precision"], 2),
                "Test Recall (%)": round(res["recall"], 2),
                "Test Specificity (%)": round(res["specificity"], 2),
                "Test FAR (%)": round(res["far"], 2),
                "Pore Size MAE (mm)": round(res["size_mae"], 4),
                "Pore Size R2": round(res["size_r2"], 4),
                "Train Time (s)": round(dur, 1)
            }
            all_records.append(rec)
            print(f"    [{m_name:<36}] Seed {seed} | PR-AUC: {rec['Test PR-AUC (%)']}% | F1: {rec['Test F1-Score']} | MAE: {rec['Pore Size MAE (mm)']}mm ({dur:.1f}s)")

    sota_df = pd.DataFrame(all_records)
    out_csv = os.path.join(OUTPUT_DIR, "sota_expanded_5seed_runs.csv")
    sota_df.to_csv(out_csv, index=False)
    print(f"\n[SAVED] SOTA 5-seed runs saved to: {out_csv}")

if __name__ == "__main__":
    main()
