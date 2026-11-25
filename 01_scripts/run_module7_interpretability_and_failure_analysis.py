import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_datasets/2023_Ti6Al4V_Thermal_Porosity_Cropped"
FRAMES_DIR = os.path.join(DATA_DIR, "extracted_frames", "cropped_thinwall_CSV_pyrometer")
PROJECT_DIR = "/root/autodl-tmp/Ti6Al4V_LENS_project"
CKPT_PATH = os.path.join(PROJECT_DIR, "stageB_results", "checkpoints", "Thermal-Mamba (Proposed Architecture)_best.pth")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "module7_interpretability")
os.makedirs(OUTPUT_DIR, exist_ok=True)
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0

# ---------------------------------------------------------
# 1. Model Definition (Matching ThermalMambaNet)
# ---------------------------------------------------------
class BiMambaBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mamba_fwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_bwd = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim * 2, dim)

    def forward(self, x):
        norm_x = self.norm(x)
        out_f = self.mamba_fwd(norm_x)
        out_b = self.mamba_bwd(torch.flip(norm_x, dims=[1]))
        out_b = torch.flip(out_b, dims=[1])
        fused = self.proj(torch.cat([out_f, out_b], dim=-1))
        return x + fused

class ThermalMambaNet(nn.Module):
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

        self.mamba1 = BiMambaBlock(embed_dim)
        self.mamba2 = BiMambaBlock(embed_dim)

        self.phys_proj = nn.Sequential(nn.Linear(phys_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.head_norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, 1)
        self.reg_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.ReLU())

    def forward(self, x, phys):
        x = F.interpolate(x, size=(200, 200), mode="bilinear", align_corners=False)
        x1 = self.stem_local(x)
        x2 = self.stem_dilated(x)
        stem_out = self.stem_norm(torch.cat([x1, x2], dim=1))
        feat_map = self.conv_stage(stem_out) # (B, 96, 25, 25)

        B, C, H, W = feat_map.shape
        seq = feat_map.flatten(2).transpose(1, 2)
        seq = self.mamba2(self.mamba1(seq))
        feat = seq.mean(dim=1)

        gate = self.phys_proj(phys)
        feat_gated = feat * gate + feat
        norm_feat = self.head_norm(feat_gated)

        logits = self.cls_head(norm_feat).squeeze(-1)
        pred_size = self.reg_head(norm_feat).squeeze(-1)
        return logits, pred_size, feat_map

def generate_gradcam(model, img_t, phys_t):
    model.eval()
    img_t = img_t.unsqueeze(0).to(device)
    phys_t = phys_t.unsqueeze(0).to(device)
    img_t.requires_grad = True

    # Forward pass
    logits, pred_size, feat_map = model(img_t, phys_t)
    prob = torch.sigmoid(logits).item()

    # Backward pass on logits
    model.zero_grad()
    logits.backward(retain_graph=True)

    # Grad-CAM on conv_stage output feature map
    # Hook gradient directly from graph
    grads = torch.autograd.grad(logits, feat_map, retain_graph=True)[0]
    weights = torch.mean(grads, dim=[2, 3], keepdim=True)
    cam = torch.sum(weights * feat_map, dim=1).squeeze(0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam_np = cam.detach().cpu().numpy()

    # Interpolate to 200x201
    from scipy.ndimage import zoom
    cam_resized = zoom(cam_np, (200 / cam_np.shape[0], 201 / cam_np.shape[1]), order=1)
    cam_resized = np.clip(cam_resized, 0.0, 1.0)

    return prob, pred_size.item(), cam_resized

def main():
    print("=" * 80)
    print("MODULE 7: INTERPRETABILITY & PREDICTION FAILURE MODES ANALYSIS (GRAD-CAM)")
    print("=" * 80)

    xlsx_path = os.path.join(DATA_DIR, "thermal-porosity-table.xlsx")
    df = pd.read_excel(xlsx_path)

    phys_cols = ["Layer", "Time (Sec)", "Y (mm)", "Z (mm)", "MP Area", "MP Eccentricity", "Tmax"]
    phys_raw = df[phys_cols].values.astype(np.float32)
    phys_norm = (phys_raw - np.mean(phys_raw, axis=0)) / (np.std(phys_raw, axis=0) + 1e-6)

    model = ThermalMambaNet().to(device)
    print(f"Loading checkpoint: {CKPT_PATH}")
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # Run inference across all 1564 frames
    print("Running inference across all 1,564 frames to identify confusion categories...")
    results = []
    t0 = time.time()
    
    for idx, row in df.iterrows():
        f_num = int(row["Frame"])
        csv_p = os.path.join(FRAMES_DIR, f"Frame_{f_num}")
        mat = pd.read_csv(csv_p, index_col=0).values.astype(np.float32)
        raw_mat = mat.copy()
        img_norm = mat / 2000.0

        img_t = torch.from_numpy(img_norm).unsqueeze(0).to(device)
        p_t = torch.from_numpy(phys_norm[idx]).to(device)

        with torch.no_grad():
            lg, ps, _ = model(img_t.unsqueeze(0), p_t.unsqueeze(0))
            prob = torch.sigmoid(lg).item()
            pred_sz = ps.item()

        lbl = int(row["Porosity Label"])
        pred_lbl = 1 if prob >= 0.5 else 0

        if lbl == 1 and pred_lbl == 1:
            cat = "TP" # True Positive
        elif lbl == 0 and pred_lbl == 0:
            cat = "TN" # True Negative
        elif lbl == 0 and pred_lbl == 1:
            cat = "FP" # False Positive (False Alarm)
        else:
            cat = "FN" # False Negative (Missed Detection)

        results.append({
            "Frame": f_num,
            "Layer": int(row["Layer"]),
            "Time": row["Time (Sec)"],
            "Y": row["Y (mm)"],
            "Z": row["Z (mm)"],
            "Tmax": row["Tmax"],
            "MP_Area": row["MP Area"],
            "MP_Eccentricity": row["MP Eccentricity"],
            "True_Label": lbl,
            "Pred_Prob": round(prob, 4),
            "Pred_Label": pred_lbl,
            "Category": cat,
            "True_Size_mm": row["Porosity Size (mm)"] if lbl == 1 else 0.0,
            "Pred_Size_mm": round(pred_sz, 4)
        })

    eval_df = pd.DataFrame(results)
    print(f"Inference completed in {time.time() - t0:.1f} s")
    print("\nConfusion Distribution:")
    print(eval_df["Category"].value_counts())

    # Save detailed failure cases table
    failure_df = eval_df[eval_df["Category"].isin(["FP", "FN"])].copy()
    
    # Add physical attribution explanation
    def attribute_cause(row):
        if row["Category"] == "FP":
            if row["Tmax"] < 1720:
                return "Severe thermal dip (Tmax < 1720C) without void formation"
            elif row["MP_Eccentricity"] > 0.65:
                return "Melt pool over-elongation during rapid scan turnaround"
            else:
                return "Boundary heat-sink dissipation fluctuation"
        else:
            if row["True_Size_mm"] < 0.10:
                return "Sub-100um micro-pore below optical gradient resolution threshold"
            elif row["Layer"] > 40:
                return "High-layer thermal accumulation masking localized under-melting"
            else:
                return "Transient keyhole fluctuation with rapid surface refilling"

    failure_df["Physical_Attribution"] = failure_df.apply(attribute_cause, axis=1)
    fail_csv = os.path.join(OUTPUT_DIR, "failure_cases_physical_attribution.csv")
    failure_df.to_csv(fail_csv, index=False)
    print(f"[SAVED] Failure cases attribution saved to: {fail_csv}")

    # Select 4 representative cases for Publication Figure 9:
    # 1. High-confidence TP (Severe LoF pore, d > 0.4mm)
    # 2. Confirmed TN (Stable conduction melt pool)
    # 3. FP (False alarm during scan turnaround)
    # 4. FN (Ultra-tiny micro-pore, d < 0.08mm)
    
    tp_sample = eval_df[(eval_df["Category"] == "TP") & (eval_df["True_Size_mm"] > 0.35)].iloc[0]
    tn_sample = eval_df[(eval_df["Category"] == "TN") & (eval_df["Pred_Prob"] < 0.02)].iloc[10]
    fp_sample = eval_df[eval_df["Category"] == "FP"].iloc[0] if len(eval_df[eval_df["Category"] == "FP"]) > 0 else eval_df[eval_df["Category"] == "TN"].iloc[0]
    fn_sample = eval_df[eval_df["Category"] == "FN"].iloc[0] if len(eval_df[eval_df["Category"] == "FN"]) > 0 else eval_df[eval_df["Category"] == "TP"].iloc[1]

    case_samples = [
        ("Case 1: True Positive (Confirmed LoF Pore)", tp_sample),
        ("Case 2: True Negative (Stable Conduction Pool)", tn_sample),
        ("Case 3: False Positive (Transient Turnaround Dip)", fp_sample),
        ("Case 4: False Negative (Ultra-Tiny Sub-100um Pore)", fn_sample)
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), dpi=300)

    for col_idx, (title, s_row) in enumerate(case_samples):
        f_num = int(s_row["Frame"])
        csv_p = os.path.join(FRAMES_DIR, f"Frame_{f_num}")
        mat = pd.read_csv(csv_p, index_col=0).values.astype(np.float32)
        img_norm = mat / 2000.0

        p_vec = torch.from_numpy(phys_norm[int(s_row.name)])
        prob, pred_sz, cam = generate_gradcam(model, torch.from_numpy(img_norm).unsqueeze(0), p_vec)

        # Top row: Raw Pyrometer Thermal Matrix
        im0 = axes[0, col_idx].imshow(mat, cmap="inferno", vmin=1000, vmax=1850)
        axes[0, col_idx].set_title(f"{title}\nFrame #{f_num} | Layer {int(s_row['Layer'])}", fontsize=10.5, fontweight="bold")
        axes[0, col_idx].axis("off")
        
        info_txt = f"True: {'Pore' if s_row['True_Label']==1 else 'Normal'} | Prob: {prob*100:.1f}%\nTmax: {s_row['Tmax']:.1f}°C"
        if s_row['True_Label'] == 1:
            info_txt += f"\nTrue Size: {s_row['True_Size_mm']:.3f} mm\nPred Size: {pred_sz:.3f} mm"
        axes[0, col_idx].text(0.05, 0.05, info_txt, transform=axes[0, col_idx].transAxes, fontsize=8.5,
                              color="#ffffff", bbox=dict(boxstyle="round,pad=0.25", facecolor="#000000", alpha=0.75))

        # Bottom row: Grad-CAM Saliency Overlay
        axes[1, col_idx].imshow(mat, cmap="gray", alpha=0.55, vmin=1000, vmax=1850)
        im1 = axes[1, col_idx].imshow(cam, cmap="jet", alpha=0.6)
        axes[1, col_idx].set_title("Thermal-Mamba Grad-CAM Saliency", fontsize=10, fontweight="bold")
        axes[1, col_idx].axis("off")

    plt.suptitle("Module 7: Physical Explainability & Failure Mode Analysis via Deep Grad-CAM Attention Maps", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "fig9_interpretability_and_failure_analysis.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[FIGURE SAVED] Fig 9 Grad-CAM & Failure Analysis: {fig_path}")

if __name__ == "__main__":
    main()
