#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Seed Worker for Advanced Experimental Suite (Ti-6Al-4V LENS)
Executed in parallel across 5 independent seeds on RTX 3090.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_score, 
                             recall_score, f1_score, confusion_matrix, roc_curve, 
                             precision_recall_curve, mean_absolute_error, r2_score)

try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_datasets/2023_Ti6Al4V_Thermal_Porosity_Cropped"
FRAMES_DIR = os.path.join(DATA_DIR, "extracted_frames", "cropped_thinwall_CSV_pyrometer")
OUTPUT_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_project/advanced_experimental_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class ThermalDataset(Dataset):
    def __init__(self, images, physical, labels, sizes, coords):
        self.images = images
        self.physical = physical
        self.labels = labels
        self.sizes = sizes
        self.coords = coords

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "physical": self.physical[idx],
            "label": self.labels[idx],
            "size": self.sizes[idx],
            "coord": self.coords[idx]
        }

def load_data():
    xlsx_p = os.path.join(DATA_DIR, "thermal-porosity-table.xlsx")
    df = pd.read_excel(xlsx_p)
    n = len(df)

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
    
    rows = df["Row"].values.astype(np.float32) / 200.0
    cols = df["Column"].values.astype(np.float32) / 201.0
    coords = np.stack([rows, cols], axis=1)

    return (df, 
            torch.from_numpy(all_imgs), 
            torch.from_numpy(phys_norm), 
            torch.from_numpy(labels), 
            torch.from_numpy(sizes),
            torch.from_numpy(coords))

# Architectures
class BiMambaBlock(nn.Module):
    def __init__(self, dim, bidirectional=True):
        super().__init__()
        self.dim = dim
        self.bidirectional = bidirectional
        if MAMBA_AVAILABLE:
            self.mamba_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
            if self.bidirectional:
                self.mamba_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        else:
            self.mamba_fwd = nn.GRU(dim, dim, batch_first=True)
            if self.bidirectional:
                self.mamba_bwd = nn.GRU(dim, dim, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        if self.bidirectional:
            self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        norm_x = self.norm(x)
        if MAMBA_AVAILABLE:
            out_f = self.mamba_fwd(norm_x)
            if self.bidirectional:
                out_b = self.mamba_bwd(torch.flip(norm_x, dims=[1]))
                out_b = torch.flip(out_b, dims=[1])
                fused = self.proj(torch.cat([out_f, out_b], dim=-1))
            else:
                fused = out_f
        else:
            out_f, _ = self.mamba_fwd(norm_x)
            if self.bidirectional:
                out_b, _ = self.mamba_bwd(torch.flip(norm_x, dims=[1]))
                out_b = torch.flip(out_b, dims=[1])
                fused = self.proj(torch.cat([out_f, out_b], dim=-1))
            else:
                fused = out_f
        return x + fused

class ThermalMambaNet(nn.Module):
    def __init__(self, in_channels=1, phys_dim=7, embed_dim=96, multi_scale=True, fusion_mode="gating"):
        super().__init__()
        self.multi_scale = multi_scale
        self.fusion_mode = fusion_mode

        if self.multi_scale:
            self.stem_local = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=4, stride=4)
            self.stem_dilated = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=4, padding=1)
            self.stem_norm = nn.BatchNorm2d(embed_dim)
        else:
            self.stem_single = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4)
            self.stem_norm = nn.BatchNorm2d(embed_dim)

        self.conv_stage = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        self.mamba1 = BiMambaBlock(embed_dim, bidirectional=True)
        self.mamba2 = BiMambaBlock(embed_dim, bidirectional=True)

        if self.fusion_mode == "gating":
            self.phys_proj = nn.Sequential(
                nn.Linear(phys_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid()
            )
        elif self.fusion_mode == "concat":
            self.phys_proj = nn.Sequential(nn.Linear(phys_dim, embed_dim), nn.GELU())
            self.fusion_fc = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU())

        self.head_norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, 1)
        self.reg_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.ReLU())

    def forward(self, x, phys):
        x = F.interpolate(x, size=(200, 200), mode="bilinear", align_corners=False)
        if self.multi_scale:
            x1 = self.stem_local(x)
            x2 = self.stem_dilated(x)
            x = self.stem_norm(torch.cat([x1, x2], dim=1))
        else:
            x = self.stem_norm(self.stem_single(x))
        x = self.conv_stage(x)

        seq = x.flatten(2).transpose(1, 2)
        seq = self.mamba2(self.mamba1(seq))
        feat = seq.mean(dim=1)

        if self.fusion_mode == "gating":
            gate = self.phys_proj(phys)
            feat = feat * gate + feat
        elif self.fusion_mode == "concat":
            p_feat = self.phys_proj(phys)
            feat = self.fusion_fc(torch.cat([feat, p_feat], dim=-1))

        norm_feat = self.head_norm(feat)
        return self.cls_head(norm_feat).squeeze(-1), self.reg_head(norm_feat).squeeze(-1)

class CNNBiLSTMNet(nn.Module):
    def __init__(self, in_channels=1, phys_dim=7, embed_dim=96):
        super().__init__()
        self.stem_local = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=4, stride=4)
        self.stem_dilated = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=4, padding=1)
        self.stem_norm = nn.BatchNorm2d(embed_dim)

        self.conv_stage = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim, 
            hidden_size=embed_dim // 2, 
            num_layers=2, 
            bidirectional=True, 
            batch_first=True
        )
        self.lstm_norm = nn.LayerNorm(embed_dim)

        self.phys_proj = nn.Sequential(
            nn.Linear(phys_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid()
        )
        self.head_norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, 1)
        self.reg_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.ReLU())

    def forward(self, x, phys):
        x = F.interpolate(x, size=(200, 200), mode="bilinear", align_corners=False)
        x1 = self.stem_local(x)
        x2 = self.stem_dilated(x)
        x = self.stem_norm(torch.cat([x1, x2], dim=1))
        x = self.conv_stage(x)

        seq = x.flatten(2).transpose(1, 2)
        lstm_out, _ = self.lstm(seq)
        lstm_out = self.lstm_norm(lstm_out)
        feat = lstm_out.mean(dim=1)

        gate = self.phys_proj(phys)
        feat = feat * gate + feat

        norm_feat = self.head_norm(feat)
        return self.cls_head(norm_feat).squeeze(-1), self.reg_head(norm_feat).squeeze(-1)

class ThermalMamba3TaskNet(nn.Module):
    def __init__(self, in_channels=1, phys_dim=7, embed_dim=96):
        super().__init__()
        self.stem_local = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=4, stride=4)
        self.stem_dilated = nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=4, padding=1)
        self.stem_norm = nn.BatchNorm2d(embed_dim)

        self.conv_stage = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        self.mamba1 = BiMambaBlock(embed_dim, bidirectional=True)
        self.mamba2 = BiMambaBlock(embed_dim, bidirectional=True)

        self.phys_proj = nn.Sequential(
            nn.Linear(phys_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid()
        )

        self.head_norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, 1)
        self.reg_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.ReLU())
        self.loc_head = nn.Sequential(
            nn.Linear(embed_dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 2), 
            nn.Sigmoid()
        )

    def forward(self, x, phys):
        x = F.interpolate(x, size=(200, 200), mode="bilinear", align_corners=False)
        x1 = self.stem_local(x)
        x2 = self.stem_dilated(x)
        x = self.stem_norm(torch.cat([x1, x2], dim=1))
        x = self.conv_stage(x)

        seq = x.flatten(2).transpose(1, 2)
        seq = self.mamba2(self.mamba1(seq))
        feat = seq.mean(dim=1)

        gate = self.phys_proj(phys)
        feat = feat * gate + feat

        norm_feat = self.head_norm(feat)
        logits = self.cls_head(norm_feat).squeeze(-1)
        pred_size = self.reg_head(norm_feat).squeeze(-1)
        pred_coord = self.loc_head(norm_feat)
        return logits, pred_size, pred_coord

class ParametricMultiTaskLoss(nn.Module):
    def __init__(self, loss_type="focal", alpha=0.8, gamma=2.0, lambda_reg=2.0, lambda_loc=2.0):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_reg = lambda_reg
        self.lambda_loc = lambda_loc

    def forward(self, logits, pred_sizes, target_labels, target_sizes, pred_coords=None, target_coords=None):
        if self.loss_type == "focal":
            bce = F.binary_cross_entropy_with_logits(logits, target_labels, reduction="none")
            probs = torch.sigmoid(logits)
            p_t = probs * target_labels + (1.0 - probs) * (1.0 - target_labels)
            alpha_t = self.alpha * target_labels + (1.0 - self.alpha) * (1.0 - target_labels)
            cls_loss = (alpha_t * (1.0 - p_t) ** self.gamma * bce).mean()
        elif self.loss_type == "weighted_bce":
            pos_weight = torch.tensor([21.0], device=logits.device)
            cls_loss = F.binary_cross_entropy_with_logits(logits, target_labels, pos_weight=pos_weight)
        elif self.loss_type == "standard_bce":
            cls_loss = F.binary_cross_entropy_with_logits(logits, target_labels)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        pos_mask = (target_labels == 1)
        if pos_mask.sum() > 0:
            reg_loss = F.smooth_l1_loss(pred_sizes[pos_mask], target_sizes[pos_mask])
        else:
            reg_loss = torch.tensor(0.0, device=logits.device)

        total_loss = cls_loss + self.lambda_reg * reg_loss

        if pred_coords is not None and target_coords is not None and pos_mask.sum() > 0:
            loc_loss = F.smooth_l1_loss(pred_coords[pos_mask], target_coords[pos_mask])
            total_loss = total_loss + self.lambda_loc * loc_loss

        return total_loss

def evaluate_loader(model, loader, is_3task=False):
    model.eval()
    logits_list, sizes_list, coords_list = [], [], []
    lbls_list, szs_list, tgt_coords_list = [], [], []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            phys = batch["physical"].to(device)
            if is_3task:
                lg, ps, pc = model(imgs, phys)
                coords_list.append(pc.cpu())
                tgt_coords_list.append(batch["coord"])
            else:
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

    res = {
        "roc_auc": roc * 100,
        "pr_auc": pr_auc * 100,
        "f1": f1,
        "optimal_th": b_th,
        "recall": rec * 100,
        "precision": prec * 100,
        "specificity": spec * 100,
        "far": far * 100,
        "size_mae": mae,
        "size_r2": r2,
        "probs": probs,
        "pred_sz": pred_sz,
        "y_true": y_true,
        "sz_true": sz_true
    }

    if is_3task and pos_m.sum() > 0:
        p_coords = torch.cat(coords_list).numpy()
        t_coords = torch.cat(tgt_coords_list).numpy()
        row_diff = (p_coords[pos_m, 0] - t_coords[pos_m, 0]) * 200.0
        col_diff = (p_coords[pos_m, 1] - t_coords[pos_m, 1]) * 201.0
        pixel_dist = np.sqrt(row_diff**2 + col_diff**2)
        res["loc_pixel_mae"] = float(np.mean(pixel_dist))
        res["pred_coords"] = p_coords
        res["tgt_coords"] = t_coords

    return res

def train_single_run(model_fn, train_loader, val_loader, test_loader, loss_type="focal", is_3task=False, epochs=35, seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    model = model_fn().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = ParametricMultiTaskLoss(loss_type=loss_type)

    best_val_pr = 0.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            imgs = batch["image"].to(device)
            phys = batch["physical"].to(device)
            lbls = batch["label"].to(device)
            szs = batch["size"].to(device)
            coords = batch["coord"].to(device) if is_3task else None

            optimizer.zero_grad()
            if is_3task:
                lg, ps, pc = model(imgs, phys)
                loss = criterion(lg, ps, lbls, szs, pred_coords=pc, target_coords=coords)
            else:
                lg, ps = model(imgs, phys)
                loss = criterion(lg, ps, lbls, szs)

            loss.backward()
            optimizer.step()
        scheduler.step()

        val_m = evaluate_loader(model, val_loader, is_3task=is_3task)
        if val_m["pr_auc"] > best_val_pr:
            best_val_pr = val_m["pr_auc"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    test_m = evaluate_loader(model, test_loader, is_3task=is_3task)
    return test_m, model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, help="Random seed for partitioning")
    args = parser.parse_args()
    seed = args.seed

    print(f"[{time.strftime('%X')}] [WORKER SEED {seed}] Starting on {device} ({torch.cuda.get_device_name(0)})...")

    df, images_t, phys_t, labels_t, sizes_t, coords_t = load_data()
    y_np = labels_t.numpy()

    EXPERIMENT_CONFIGS = [
        ("CNN-BiLSTM (LRCN Temporal Baseline)", lambda: CNNBiLSTMNet(), "focal", False),
        ("Ablation: Single-Scale Stem", lambda: ThermalMambaNet(multi_scale=False, fusion_mode="gating"), "focal", False),
        ("Ablation: Naive Concat Fusion", lambda: ThermalMambaNet(multi_scale=True, fusion_mode="concat"), "focal", False),
        ("Ablation: Standard BCE Loss", lambda: ThermalMambaNet(multi_scale=True, fusion_mode="gating"), "standard_bce", False),
        ("Ablation: Weighted BCE Loss", lambda: ThermalMambaNet(multi_scale=True, fusion_mode="gating"), "weighted_bce", False),
        ("Thermal-Mamba (3-Task with Localization)", lambda: ThermalMamba3TaskNet(), "focal", True)
    ]

    seed_run_csv = os.path.join(OUTPUT_DIR, f"runs_seed_{seed}.csv")
    seed_sev_csv = os.path.join(OUTPUT_DIR, f"sev_seed_{seed}.csv")

    completed = set()
    records = []
    sev_records = []
    if os.path.exists(seed_run_csv):
        try:
            df_ex = pd.read_csv(seed_run_csv)
            for _, r in df_ex.iterrows():
                completed.add(str(r["Model"]))
            records = df_ex.to_dict("records")
            print(f"[WORKER SEED {seed}] Resuming: {len(completed)} models already finished.")
        except:
            pass

    if os.path.exists(seed_sev_csv):
        try:
            sev_records = pd.read_csv(seed_sev_csv).to_dict("records")
        except:
            pass

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, temp_idx = next(sss1.split(images_t, y_np))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_sub, test_sub = next(sss2.split(images_t[temp_idx], y_np[temp_idx]))
    val_idx = temp_idx[val_sub]
    test_idx = temp_idx[test_sub]

    train_ds = ThermalDataset(images_t[train_idx], phys_t[train_idx], labels_t[train_idx], sizes_t[train_idx], coords_t[train_idx])
    val_ds = ThermalDataset(images_t[val_idx], phys_t[val_idx], labels_t[val_idx], sizes_t[val_idx], coords_t[val_idx])
    test_ds = ThermalDataset(images_t[test_idx], phys_t[test_idx], labels_t[test_idx], sizes_t[test_idx], coords_t[test_idx])

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, pin_memory=True)

    for name, m_fn, loss_tp, is_3t in EXPERIMENT_CONFIGS:
        if name in completed:
            print(f"[WORKER SEED {seed}] Skipping {name} (already done).")
            continue

        t0 = time.time()
        res, trained_m = train_single_run(
            m_fn, train_loader, val_loader, test_loader, 
            loss_type=loss_tp, is_3task=is_3t, epochs=35, seed=seed
        )
        dur = time.time() - t0

        rec = {
            "Seed": seed,
            "Model": name,
            "Loss Type": loss_tp,
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
        if is_3t:
            rec["Centroid Error (px)"] = round(res.get("loc_pixel_mae", 0.0), 2)
        records.append(rec)
        completed.add(name)
        pd.DataFrame(records).to_csv(seed_run_csv, index=False)

        print(f"[{time.strftime('%X')}] [WORKER SEED {seed}] Done: {name:<35} | PR-AUC: {rec['Test PR-AUC (%)']}% | F1: {rec['Test F1-Score']} | FAR: {rec['Test FAR (%)']}% | MAE: {rec['Pore Size MAE (mm)']}mm ({dur:.1f}s)")

        # Severity
        probs = res["probs"]
        pred_sz = res["pred_sz"]
        y_t = res["y_true"]
        sz_t = res["sz_true"]
        th = res["optimal_th"]
        y_pred = (probs >= th).astype(int)

        pos_indices = np.where(y_t == 1)[0]
        for p_idx in pos_indices:
            p_sz = sz_t[p_idx]
            if p_sz < 0.30:
                sev = "Micro LoF (<0.30mm)"
            elif p_sz < 0.60:
                sev = "Medium LoF (0.30-0.60mm)"
            else:
                sev = "Severe LoF (>=0.60mm)"

            sev_records.append({
                "Seed": seed,
                "Model": name,
                "Severity": sev,
                "True Size (mm)": p_sz,
                "Pred Size (mm)": pred_sz[p_idx],
                "Detected": int(y_pred[p_idx] == 1),
                "Size Error (mm)": abs(pred_sz[p_idx] - p_sz)
            })
        pd.DataFrame(sev_records).to_csv(seed_sev_csv, index=False)

    print(f"[{time.strftime('%X')}] [WORKER SEED {seed}] ALL 6 MODELS SUCCESSFULLY FINISHED!")

if __name__ == "__main__":
    main()
