# Thermal-field complexity descriptors for porosity prediction in LENS Ti–6Al–4V

Experiment code and results for porosity prediction from melt-pool pyrometry in laser engineered
net shaping (LENS) of Ti–6Al–4V thin walls, using a public dataset.

The study asks a simple question: **which conclusions survive when a porosity model is evaluated
along the build direction instead of on a random split?** Using the public LENS thin-wall dataset
(1564 pyrometer frames, 71 pore frames, 60 layers), we show that evaluation protocol moves results
more than model choice does, and that three physically interpretable descriptors of the thermal
field carry most of the achievable gain.

---

## Headline results

Forward-chaining evaluation trains on layers `1…k` and tests on layers `k+1…k+10`, mimicking
deployment on material deposited later in the build.

| Model | Protocol | Baseline features (18-d) | + entropy + gradient (21-d) |
|---|---|---|---|
| CatBoost | grouped CV | 0.938 | 0.955 |
| CatBoost | **forward chaining** | 0.903 ± 0.083 | **0.970 ± 0.029** |
| XGBoost | forward chaining | 0.903 ± 0.094 | 0.951 ± 0.044 |
| Random forest | forward chaining | 0.811 ± 0.162 | 0.903 ± 0.126 |

Ten seeds × seven split boundaries; paired ΔPR-AUC for CatBoost = +0.068 (95% CI [0.050, 0.086]).
The three descriptors recover 96% of the gain available from a 21-dimensional spatial feature set,
and the split-to-split standard deviation falls by 65%.

**The gain is not a dimensionality artefact.** Permuting the three descriptors across frames —
same marginals, no information — yields only +0.017, and three random projections of the field reach
+0.024. Generic distribution moments (skewness, kurtosis, RMS) do capture part of the signal
(+0.086), but the complexity descriptors remain significantly better: 0.974 vs 0.947 over 21 splits
(paired Δ = +0.028, 95% CI [0.008, 0.051], p < 0.0001), and adding the moments on top changes
nothing (p = 0.30).

**The gain is protocol-conditional.** Under the random 70/30 + SMOTE protocol used in earlier work
on this dataset, the same descriptors are *worse* than conventional summary statistics
(F1 0.872 vs 0.895). They help only when the model has to extrapolate.

---

## Repository layout

```
.
├── data/
│   ├── download.sh              # fetch the dataset from Harvard Dataverse
│   └── README.md
├── src/
│   ├── features.py              # 18 baseline features
│   ├── features_spatial.py      # 21 physics-inspired spatial descriptors
│   └── features_temporal.py     # 11 temporal features (supplementary ablation only)
├── experiments/
│   ├── 01_build_cache.py        # CSV frames -> thermal.npy + labels.parquet
│   ├── 02_baselines.py          # six classifiers x two protocols
│   ├── 03_protocol_comparison.py
│   ├── 04_ablation_groups.py    # which descriptor families help
│   ├── 05_ablation_combinations.py
│   ├── 06_stability.py          # 10 seeds x 7 boundaries
│   ├── 07_robustness_first_layers.py
│   ├── 08_layer_level.py
│   ├── 09_icinco_and_cost.py    # reproduce the published protocol; cost-sensitive points
│   ├── 10_abstention.py
│   ├── 11_uncertainty.py        # calibration, conformal prediction, bootstrap
│   ├── 12_oneclass_svm.py
│   ├── 13a_shap.py
│   ├── 13b_shap_excluding_first_layers.py
│   ├── 14_cnn.py
│   ├── 15_figures.py
│   ├── 16_controls.py           # permutation / projection / moment controls
│   ├── 17_controls_paired.py    # complexity vs moments, 21 splits
│   └── 18_protocol_position_holdout.py  # protocol, positional ablation, held-out layers
├── tests/test_claims.py         # regression tests: every reported number
├── results/                     # generated tables (tracked, small)
├── figures/                     # generated figures
├── Makefile
└── requirements.txt
```

---

## Quick start

```bash
git clone https://github.com/WangAnChuan/Thermal-field-complexity-descriptors-for-porosity-prediction-in-LENS-Ti-6Al-4V.git
cd thermal-porosity

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

bash data/download.sh          # ~52 MB from Harvard Dataverse
make all                       # ~35 min on 4 CPU cores, no GPU required
```

`make all` ends with `tests/test_claims.py`, which asserts that every headline number still matches
the generated result files. If a refactor changes any reported value, the test fails.

Run a single stage:

```bash
make data        # parse the 1564 CSV frames
make ablations   # feature ablation tables
make figures     # regenerate all six figures
make test        # verify the reported numbers
```

---

## Experimental protocols

| Protocol | Definition | Purpose |
|---|---|---|
| **Grouped CV** | 5 folds over the 60 layers; all frames of a layer stay together | prevents leakage between adjacent frames |
| **Forward chaining** | train layers `1…k`, test `k+1…k+10`, `k ∈ {20,30,40,50}` | extrapolation along the build direction |
| **Random split** | stratified 70/30 + SMOTE, 10 repeats | reproduces the protocol of prior work |

Metrics: PR-AUC is primary (4.5% positive rate makes accuracy uninformative), with ROC-AUC,
precision/recall at a 5% false-positive rate, and Brier/ECE for calibration.

Two bootstrap procedures are reported and they answer different questions:

- **split-level** (resample the test splits): is the gain consistently positive across splits?
- **layer-level** (resample whole layers): does the gain transfer to unseen layers?

The split-level test is the one reported in the headline table; the layer-level test gives
p ≈ 0.05 and is the more conservative statement.

---

## The three descriptors

Computed per frame from the temperature field `T(r, c)`:

1. **Histogram entropy** over 32 bins spanning 0–2000 °C.
2. **Mean gradient magnitude**: sum of the mean absolute row-wise and column-wise forward differences.
3. **99th percentile of gradient magnitude**.

They describe how heterogeneous the field is, not how hot it is. Peak temperature ranks 11th by
mean |SHAP| (0.269) behind entropy (2.593).

---

## Reproducibility

- All random seeds are fixed; no stage of the pipeline is stochastic beyond the declared seeds.
- Independent re-runs reproduce every table bit-for-bit (differences < 1e-6; the only varying
  column is wall-clock runtime).
- Versions used: Python 3.12.3, numpy 2.5.3, scikit-learn 1.9.1, XGBoost 3.4.1,
  LightGBM 4.7.0, CatBoost 1.2.10, TabPFN 9.1.0, PyTorch 2.13.0+cpu.
- The CNN baseline stops at the epoch that maximises PR-AUC on validation layers held out from the
  training portion. The test fold is never used for model selection. An earlier version of this
  code selected the epoch on the test fold, which inflated the CNN by about 0.2 PR-AUC; that
  variant has been removed.

---

## Known limitations

- Single geometry, single build. "Extrapolation" means along the build direction, not across part
  geometries.
- Only 71 pore frames; 29 remain after removing layers 1–2, and each forward-chaining test segment
  holds 4–7 positives. Confidence intervals are wide by necessity.
- 59% of pore frames occur in layers 1–2, adjacent to the substrate. We report results with those
  layers removed; the gain grows rather than shrinks.
- The CNN baseline is single-seed and sensitive to the stopping criterion.

---

## Citation

If you use this code, please cite the original dataset:

```bibtex
@dataset{zamiela2023thermal,
  title  = {Thermal-porosity characterization data of additively manufactured Ti-6Al-4V
            thin-walled structure via laser engineered net shaping},
  author = {Zamiela, Christian and Tian, Wenmeng and Guo, Shenghan and Bian, Linkan},
  year   = {2023},
  doi    = {10.7910/DVN/BWHYEH}
}
```

## License

Code released under the MIT License (see `LICENSE`). The dataset is distributed separately under
the terms of its Dataverse record and is not included in this repository.
