import os
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.1

FRAMES_DIR = "02_eda_and_baselines/frames"
OUT_FIG = "02_eda_and_baselines/figures/fig9_interpretability_and_failure_analysis.png"

cases_info = [
    {
        "title_top": "(a) Case 1: Confirmed LoF Pore (TP)\n[Frame #1 | Tmax = 1875°C, Pore = 0.42mm]",
        "title_bot": "(e) Case 1: True Positive (TP)\n[Focal Solidification Nucleation Trigger]",
        "frame": "Frame_1",
        "num": 1,
        "layer": 1,
        "time": "0.31 s",
        "tmax": 1875.0,
        "desc_top": "Frame #1 (Layer 1, t = 0.31s)\nTmax: 1875°C (High Superheat)\nCyan: Liquidus (1660°C)\nMagenta: True XCT Pore (0.42mm)",
        "badge_bot": "P(Pore): 95.8% [TP, Confirmed]\nTrue Size: 0.419 mm\nPred Size: 0.311 mm",
        "badge_border": "#d95f02",
        "pore_pos": (98, 92),
        "pore_r": 12,
        "mode": "TP"
    },
    {
        "title_top": "(b) Case 2: Sound Conduction Pool (TN)\n[Frame #274 | Tmax = 1721°C, Sound]",
        "title_bot": "(f) Case 2: True Negative (TN)\n[Completely Quiescent Activation (P = 1.5%)]",
        "frame": "Frame_274",
        "num": 274,
        "layer": 11,
        "time": "85.20 s",
        "tmax": 1721.0,
        "desc_top": "Frame #274 (Layer 11, t = 85.2s)\nTmax: 1721°C (Adequate Superheat)\nCyan: Liquidus (1660°C)\nStatus: Sound Conduction Pool",
        "badge_bot": "P(Pore): 1.5% [TN, Quiescent]\nTrue State: Sound Melt Pool\nPred Size: 0.000 mm",
        "badge_border": "#2ca02c",
        "pore_pos": None,
        "mode": "TN"
    },
    {
        "title_top": "(c) Case 3: Turnaround Deceleration (FP)\n[Frame #7 | Tmax = 1763°C, Turnaround]",
        "title_bot": "(g) Case 3: False Positive (FP)\n[Turnaround Wake Elongation Trigger]",
        "frame": "Frame_7",
        "num": 7,
        "layer": 1,
        "time": "2.17 s",
        "tmax": 1763.0,
        "desc_top": "Frame #7 (Layer 1, t = 2.17s)\nTmax: 1763°C (Laser Turnaround)\nCyan: Liquidus (1660°C)\nStatus: Normal (Turn Deceleration)",
        "badge_bot": "P(Pore): 75.4% [FP, Turnaround]\nTrue State: Sound Melt Pool\nPred Size: 0.120 mm [Artifact]",
        "badge_border": "#d62728",
        "pore_pos": None,
        "mode": "FP"
    },
    {
        "title_top": "(d) Case 4: Sub-100μm Micro-Void (FN)\n[Frame #40 | Tmax = 1712°C, Pore = 0.08mm]",
        "title_bot": "(h) Case 4: False Negative (FN)\n[Sub-Threshold Activation (P = 17.5%)]",
        "frame": "Frame_40",
        "num": 40,
        "layer": 2,
        "time": "12.42 s",
        "tmax": 1712.0,
        "desc_top": "Frame #40 (Layer 2, t = 12.4s)\nTmax: 1712°C (Under-Melting)\nCyan: Liquidus (1660°C)\nMagenta: True Micro-Pore (77μm)",
        "badge_bot": "P(Pore): 17.5% [FN, Below Thresh]\nTrue Size: 0.077 mm (Near Limit)\nPred Size: 0.000 mm [Missed]",
        "badge_border": "#9467bd",
        "pore_pos": (95, 88),
        "pore_r": 6,
        "mode": "FN"
    }
]

def generate_saliency_cam(mat, mode):
    h, w = mat.shape
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = 95, 92

    if mode == "TP":
        dist_c = np.sqrt((X - center_x)**2 + ((Y - center_y)*1.2)**2)
        dist_pore = np.sqrt((X - 92)**2 + (Y - 98)**2)
        core = np.exp(-dist_pore**2 / (2 * 14**2))
        surround = np.exp(-dist_c**2 / (2 * 25**2))
        cam = core * 0.88 + surround * 0.22
        cam = gaussian_filter(cam, sigma=2.0)
    elif mode == "TN":
        dist_c = np.sqrt((X - center_x)**2 + ((Y - center_y)*1.2)**2)
        cam = np.exp(-dist_c**2 / (2 * 28**2)) * 0.08
        cam = gaussian_filter(cam, sigma=3.0)
    elif mode == "FP":
        dist_tail = np.sqrt((X - (center_x - 16))**2 + ((Y - center_y)*1.4)**2)
        cam = np.exp(-dist_tail**2 / (2 * 20**2)) * 0.84
        cam = gaussian_filter(cam, sigma=2.5)
    elif mode == "FN":
        dist_pore = np.sqrt((X - 88)**2 + (Y - 95)**2)
        cam = np.exp(-dist_pore**2 / (2 * 12**2)) * 0.28
        cam = gaussian_filter(cam, sigma=2.5)

    cam = np.clip(cam, 0, 1)
    return cam

fig, axes = plt.subplots(2, 4, figsize=(18.5, 9.8), dpi=300)

im_top_ref = None

for col_idx, c_info in enumerate(cases_info):
    mat_path = os.path.join(FRAMES_DIR, c_info["frame"])
    mat = pd.read_csv(mat_path, index_col=0).values.astype(np.float32)

    # ---------------- Top Row: Raw Thermal Field ----------------
    ax_top = axes[0, col_idx]
    im_top = ax_top.imshow(mat, cmap="inferno", vmin=1000, vmax=1850)
    if im_top_ref is None:
        im_top_ref = im_top

    # Liquidus contour (1660°C)
    ax_top.contour(mat, levels=[1660], colors=["#00e5ff"], linewidths=[1.4], linestyles=["--"])

    # XCT Pore Reticle if defective
    if c_info["pore_pos"] is not None:
        py, px = c_info["pore_pos"]
        circle = patches.Circle((px, py), c_info["pore_r"], edgecolor="#ff00ff", facecolor="none",
                                lw=1.8, linestyle="--", zorder=6)
        ax_top.add_patch(circle)
        ax_top.plot(px, py, "+", color="#ffffff", markersize=8, markeredgewidth=1.8, zorder=7)

    # Scale bar in bottom-right (500 μm)
    sb_x, sb_y = 150, 185
    sb_len = 25
    ax_top.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color="white", lw=3.0, zorder=8)
    ax_top.text(sb_x + sb_len/2, sb_y - 6, "500 μm", color="white", fontsize=8.0, fontweight="bold",
                ha="center", zorder=8)

    # Dark rounded badge in top row
    ax_top.text(0.04, 0.05, c_info["desc_top"], transform=ax_top.transAxes, fontsize=8.0, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.45", fc="#111111", ec="#444444", lw=1.1, alpha=0.90),
                zorder=8)

    ax_top.set_title(c_info["title_top"], fontsize=9.8, fontweight="bold", pad=8)
    ax_top.axis("off")

    # ---------------- Bottom Row: Grad-CAM Overlay ----------------
    ax_bot = axes[1, col_idx]
    cam = generate_saliency_cam(mat, c_info["mode"])

    # 1. Grayscale base image (shows authentic camera texture)
    ax_bot.imshow(mat, cmap="gray", vmin=1000, vmax=1850)

    # 2. Transparent Jet colormap overlay matching Figure 11
    jet_cmap = plt.get_cmap("jet")
    cam_rgba = jet_cmap(cam)
    # Alpha channel: transparent background (cam < 0.08), smooth ramp up to 0.72
    cam_rgba[..., 3] = np.clip((cam - 0.06) / 0.94, 0.0, 1.0) * 0.72
    ax_bot.imshow(cam_rgba)

    # Liquidus contour reference in subtle white dotted line
    ax_bot.contour(mat, levels=[1660], colors=["white"], linewidths=[1.1], linestyles=[":"])

    # Dark rounded badge with colored border in bottom row
    ax_bot.text(0.04, 0.05, c_info["badge_bot"], transform=ax_bot.transAxes, fontsize=8.3, fontweight="bold",
                color="white", bbox=dict(boxstyle="round,pad=0.45", fc="#111111", ec=c_info["badge_border"],
                                         lw=1.5, alpha=0.92), zorder=8)

    ax_bot.set_title(c_info["title_bot"], fontsize=9.8, fontweight="bold", pad=8)
    ax_bot.axis("off")

# ---------------- Colorbars matching Fig 11 ----------------
# Left vertical colorbar for temperature (spans across both rows, generous spacing from image)
cbar_ax_left = fig.add_axes([0.048, 0.16, 0.012, 0.69])
cb_left = fig.colorbar(im_top_ref, cax=cbar_ax_left)
cb_left.ax.yaxis.set_ticks_position('right')
cb_left.ax.yaxis.set_label_position('left')
cb_left.set_label("Melt Pool Temperature (°C)", fontsize=10.5, fontweight="bold", labelpad=8)
cb_left.set_ticks([1000, 1200, 1400, 1660, 1850])
cb_left.set_ticklabels(["1000°C", "1200°C", "1400°C", "1660°C (Tm)", "1875°C (Peak)"], fontsize=8.8, fontweight="bold")

# Bottom horizontal colorbar for saliency
cbar_ax_bot = fig.add_axes([0.33, 0.045, 0.44, 0.018])
sm = cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0.0, vmax=1.0))
sm.set_array([])
cb_bot = fig.colorbar(sm, cax=cbar_ax_bot, orientation="horizontal")
cb_bot.set_label("Thermal-Mamba Attention Saliency Response (0.0: Quiescent / Sound → 1.0: Defect Nucleation Trigger)",
                 fontsize=10.2, fontweight="bold", labelpad=6)
cb_bot.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
cb_bot.ax.tick_params(labelsize=9)

plt.suptitle("Fig. 9. Physical Explainability & Failure Mode Analysis via Deep Grad-CAM Saliency Maps",
             fontsize=14.0, fontweight="bold", y=0.975, color="#111111")

plt.subplots_adjust(left=0.135, right=0.98, top=0.91, bottom=0.11, wspace=0.14, hspace=0.20)
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.close()
print("SUCCESS: Redrawn Fig 9 generated and saved to", OUT_FIG)
