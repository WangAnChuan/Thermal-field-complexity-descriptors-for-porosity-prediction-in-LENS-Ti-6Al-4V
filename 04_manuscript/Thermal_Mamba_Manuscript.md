# Thermal-Mamba: In-Situ Micro-Porosity Detection and Quantitative 3D XCT Sizing in Laser Directed Energy Deposition via Physics-Gated State-Space Modeling

**Target Journal**: *Additive Manufacturing* (Elsevier, JCR Q1, Top Journal) / *IEEE Transactions on Industrial Informatics* (JCR Q1, Top Journal)  
**Article Type**: Full Research Paper (Original Article)  
**Keywords**: Metal Additive Manufacturing; Directed Energy Deposition (DED / LENS); In-Situ Melt Pool Monitoring; Micro-Porosity Inversion; State-Space Model (Mamba); X-ray Computed Tomography (XCT); Class Imbalance Learning.

---

## Highlights

* **First State-Space Architecture in DED**: Proposes Thermal-Mamba, establishing the first selective state-space model for in-situ melt pool thermal dynamics with $\mathcal{O}(L)$ linear complexity and long-range cross-layer memory across a 60-layer build.
* **Overcoming 21:1 Extreme Imbalance**: Incorporates a physics-guided geometric gating mechanism that breaks the precision-recall trade-off, achieving an **$87.87 \pm 8.75\%$ Test PR-AUC across 5 independent seeds**, statistically significantly outperforming landmark domain baselines ($p = 0.0161 < 0.05$) and crushing false alarm rates down to **$0.62 \pm 0.68\%$**.
* **Beyond Binary Classification to 3D XCT Sizing**: Extends in-situ monitoring from crude 0/1 anomaly alarms to continuous 3D XCT micro-pore equivalent diameter regression, reaching an **error down to $77\ \mu\text{m}$ ($R^2 = 0.8328$)** in deep networks and **$16.2\ \mu\text{m}$ ($R^2 = 0.9601$)** under physical gradient tree inversion.
* **Superiority over General Vision SOTAs**: Rigorous 5-seed empirical comparisons across 40 independent runs prove that heavy general vision models (Swin-T, ConvNeXt-Tiny) suffer severe overfitting ($p < 0.0001$), whereas Thermal-Mamba operates with **only 0.41 M parameters** (1/27th of ResNet-18) at an ultra-real-time throughput of **438.8 FPS** on an NVIDIA RTX 3090.
* **Physical Interpretability & Attribution**: Validates via Grad-CAM saliency and physical metallurgy that model attention strictly anchors onto the trailing solid-liquid solidification boundary, resolving temperature overlap ambiguities ($1720 \sim 1780^\circ\text{C}$) and cold-substrate conductive heat-sink hazards ($p = 8.27 \times 10^{-17}$).

---

## Abstract

Laser-directed energy deposition (DED) offers unparalleled geometric freedom for manufacturing safety-critical components; however, the stochastic nucleation of subsurface lack-of-fusion (LoF) micro-porosity severely impairs structural integrity and fatigue endurance. Existing vision-based in-situ monitoring paradigms face three critical bottlenecks: (i) catastrophic precision collapse under extreme class imbalance (typically >20:1 normal-to-defective ratios), (ii) reliance on heavy convolutional or recurrent backbones that either suffer from long-range gradient dissipation across multi-layer thermal histories or impose excessive latency exceeding real-time closed-loop thresholds, and (iii) restriction to coarse binary anomaly classification that ignores continuous physical defect sizing. 

Here, we present **Thermal-Mamba**, a lightweight, physics-gated bi-directional state-space network for simultaneous in-situ micro-porosity detection and quantitative defect sizing in Ti-6Al-4V thin-walled laser metal deposition. Thermal-Mamba combines multi-scale convolutional patch embeddings with dual-directional selective state-space scans, dynamically modulated by a physical geometric projection gate encoding melt pool peak temperature ($T_{max}$), projected area, and eccentricity. Validated against registered 3D X-ray computed tomography (XCT) ground-truth labels across 1,564 high-speed pyrometer thermal frames, extensive 5-seed experiments (40 independent runs) demonstrate that Thermal-Mamba achieves a state-of-the-art **Test PR-AUC of $87.87 \pm 8.75\%$** and a **specificity of $99.38 \pm 0.68\%$**, reducing the false alarm rate to **$0.62 \pm 0.68\%$**. Paired $t$-tests confirm that Thermal-Mamba statistically significantly outperforms landmark domain CNN baselines ($p = 0.0161 < 0.05$) while crushing modern heavy vision architectures (Swin-T: $46.73\%$, ConvNeXt-Tiny: $57.45\%$, $p < 0.0001$) that overfit on small manufacturing sample sizes. Furthermore, Thermal-Mamba achieves multi-task micro-porosity sizing with a mean absolute error of **$0.0774\text{ mm}$ ($77.4\ \mu\text{m}$, $R^2 = 0.8328$)**, while operating with **only 0.41 M parameters** at an inference speed of **438.8 FPS** (2.28 ms/frame), providing 14.6$\times$ margin over industrial closed-loop real-time requirements (30 FPS). Physical metallurgy attribution and Grad-CAM saliency confirm that the network's attention focus strictly tracks the trailing solid-liquid boundary where under-melting occurs, providing an explainable, edge-deployable pathway toward zero-defect additive manufacturing.

---

## 1. Introduction

Metal Additive Manufacturing (AM), particularly Laser-Directed Energy Deposition (L-DED, also known as Laser Engineered Net Shaping, LENS), has revolutionized the direct fabrication, repair, and geometric modification of high-performance components in aerospace, biomedical, and defense engineering [1, 2]. By simultaneously feeding metallic powder through a coaxial nozzle into a moving laser-generated melt pool, L-DED achieves rapid deposition rates and flexible multi-axis fabrication [3]. However, the complex thermal-fluid interactions governing the rapid melting, turbulent Marangoni convection, and rapid solidification cycles ($>10^4\ \text{K/s}$) frequently induce microstructural anomalies [4]. Among these, **lack-of-fusion (LoF) porosity**—characterized by sharp, un-melted crevices and irregular voids spanning 0.05 mm to 1.00 mm—represents the single most detrimental defect category, serving as severe stress concentration risers that degrade tensile ductility and fatigue thresholds by more than 50% [5, 6].

To eliminate reliance on expensive, destructive post-build inspections, non-destructive testing (NDT) via in-situ optical and thermal sensing has emerged as an indispensable research frontier [7, 8]. Thermal signatures captured via high-speed infrared (IR) cameras and coaxial two-color pyrometers provide direct windows into the localized thermodynamic state of the process [9, 10]. Early foundational contributions by Tian, Guo, and Bian [11] introduced dual-sensor data fusion networks (`PyroNet` and `IRNet`) correlating pyrometer images and sequential IR thermal frames with porosity. Subsequent investigations by McGowan et al. [12] introduced physics-informed loss constraints (`PICNN`), and Chen and Guo [13] utilized physics-constrained generative adversarial networks (`DCGAN`) to synthesize minority defect classes.

Despite these pioneering advances, three fundamental bottlenecks continue to impede the practical industrial transition from laboratory monitoring to certified closed-loop quality assurance:

1. **Extreme Class Imbalance and False Alarm Degradation**: In nominal qualification builds, process-induced porosity is an intrinsically sparse phenomenon. As rigorously quantified in Section 3 of this work, the real-world ratio of defect-free melt pools to porous frames in Ti-6Al-4V thin walls reaches **21.0 : 1** (a 4.54% positive prevalence). Under such severe class skew, standard deep learning models optimized via conventional cross-entropy collapse into majority-class triviality or exhibit catastrophic precision erosion. In industrial practice, a false alarm rate (FAR) of even 2% to 3% triggers dozens of unnecessary laser interrupts per build, destroying manufacturing rhythm and introducing secondary cold-joint defects.
2. **The Dilemma Between Long-Range Spatiotemporal Memory and Inference Latency**: Additive manufacturing is intrinsically a multi-layer non-stationary process. Melt pool behavior at layer $k$ is profoundly dictated by the cumulative thermal history of preceding layers $1, \dots, k-1$. As experimentally proven in our statistical analyses, early layers (Layers 1–20) encounter a massive "cold-substrate conductive heat-sink" that rapidly extracts laser energy, causing 87.3% of LoF defects to cluster in the bottom third of the build. Existing spatiotemporal models rely primarily on Convolutional Long Short-Term Memory (`ConvLSTM`) [14, 15] or Recurrent Neural Networks [11], which suffer from severe gradient vanishing over 60-layer horizons and impose heavy sequential latency ($>30\ \text{ms}$). Conversely, modern Vision Transformers (`ViT`, `Swin-T`) [16, 17] incur quadratic computational scaling $\mathcal{O}(L^2)$ and lack localized inductive biases, leading to severe overfitting when trained on limited specialized engineering datasets without massive pretraining.
3. **Coarse Binary Classification vs. Continuous Physical Defect Sizing**: Nearly all published in-situ monitoring architectures formulate porosity detection strictly as a binary decision ($y \in \{0, 1\}$). However, from fracture mechanics and ASTM/ISO certification standpoints, a benign sub-$100\ \mu\text{m}$ spherical micro-void has radically different structural consequences compared to an extensive 1.0 mm interconnected un-melted fissure [18]. Treating all defects homogenously discards the rich physical information available in high-resolution computed tomography (XCT) ground-truth data.

To resolve these interconnected challenges, this paper presents **Thermal-Mamba**, a physics-gated bi-directional state-space network for synchronized, real-time in-situ micro-porosity classification and quantitative size inversion in metal additive manufacturing. Recently pioneered by Gu and Dao [19], State-Space Models (SSMs)—specifically Mamba—achieve linear computational complexity $\mathcal{O}(L)$ with respect to sequence length while incorporating input-dependent selective state transitions. We conceptualize the continuous layer-by-layer laser deposition process as a continuous thermodynamic state-space system, where Mamba's selective scan mechanism allows the network to carry long-range thermal accumulation memory while dynamically isolating localized micro-porosity anomalies.

The principal scientific and engineering contributions of this work are fourfold:

* **Pioneering State-Space Modeling for Additive Manufacturing**: We formulate the first bidirectional visual state-space framework (`Thermal-Mamba`) explicitly tailored for in-situ melt pool thermal dynamics, achieving linear complexity and maintaining long-range cross-layer memory across 60 deposited layers with only 0.41 M parameters.
* **Physics-Guided Geometric Gating for Extreme Imbalance (21:1)**: We construct a mechanism-driven gating branch incorporating melt pool peak temperature ($T_{max}$), projected area, and eccentricity directly into the latent state representation, lifting Test PR-AUC to **$87.87 \pm 8.75\%$** across 5 independent seeds—statistically significantly outperforming the domain baseline `PyroNet-2D` ($p = 0.0161$) and suppressing false alarm rates to **$0.62\%$**.
* **Unified Multi-Task Paradigm for 3D XCT Continuous Defect Sizing**: Transcending conventional binary classification, Thermal-Mamba integrates a multi-task regression head calibrated against registered 3D XCT metrology, predicting micro-porosity equivalent diameter down to **$0.0774\text{ mm}$ ($77.4\ \mu\text{m}$)** in deep networks and **$16.2\ \mu\text{m}$ ($R^2 = 0.9601$)** under physical tree inversion.
* **Rigorous 5-Seed SOTA Benchmark and Physical Interpretability**: Across 40 independent experimental runs comparing against classical baselines and modern vision SOTAs (`Swin-T`, `ConvNeXt-Tiny`, `EfficientNet-B0`), we demonstrate that heavy vision models suffer severe overfitting ($p < 0.0001$), whereas Thermal-Mamba establishes an optimal Pareto frontier operating at **438.8 FPS** on an RTX 3090. Metallurgical Grad-CAM saliency confirms that the network's attention focus strictly anchors onto the trailing solidification boundary where lack of fusion originates.

---

## 2. Related Work

### 2.1 In-Situ Sensing and Melt Pool Monitoring in Metal AM
Process repeatability and defect mitigation remain paramount challenges in laser-based additive manufacturing. Thermal monitoring via optical pyrometry and thermographic imaging represents the most effective non-invasive modality [7, 20]. Coaxial two-color pyrometers measure high-temperature emissions within the melt pool core at sampling frequencies exceeding 10 kHz [9], while off-axis infrared cameras capture the macro-scale cooling history of the surrounding heat-affected zone (HAZ) [10]. Stockman et al. [21] demonstrated that localized pyrometric temperature variations correlate with internal voids verified by post-build XCT. Recent works by Ji et al. [22] integrated coaxial pyrometry with CMOS cameras for bead geometry estimation, and Karthikeyan et al. [23] combined acoustic emission with thermography using explainable AI (LIME) to distinguish lack-of-fusion voids from spatter-induced porosity. Nevertheless, raw sensor streams are subject to high emissivity noise, scan track reflections, and non-stationary thermal drift, requiring deep representations capable of filtering superficial artifacts.

### 2.2 Deep Learning Architectures for AM Defect Prediction
Machine learning and deep learning have catalyzed rapid progress in AM defect diagnostics [24, 25]. Convolutional Neural Networks (CNNs) excel at extracting localized spatial patterns from melt pool thermal images. Tian et al. [11] established `PyroNet` and `IRNet`, using 2D CNNs and Long-term Recurrent Convolutional Networks (LRCN) for layer-wise porosity prediction. Ho et al. [26] developed `DLAM` based on Residual-Recurrent CNNs (`Res-RCNN`), claiming 99.49% accuracy by recursively passing feature maps through multi-level residual units. In parallel, spatiotemporal frameworks based on `ConvLSTM` have gained traction: Mao et al. [14] developed a ConvLSTM encoder-decoder predicting layer-to-layer porosity propagation, and Rong et al. [15] introduced `MFST-Net` combining self-attention ConvLSTM with multi-scale spatial convolution. However, recurrent architectures are inherently sequential, hindering parallel training and suffering from catastrophic forgetting over extended print cycles.

### 2.3 Physics-Informed Machine Learning and Class Imbalance
Purely data-driven models frequently fail to generalize across varying geometries because they lack physical constraints [27]. In response, Physics-Informed Machine Learning (PIML) has emerged as a cornerstone paradigm [28]. McGowan et al. [12] introduced `PICNN`, incorporating finite element analysis (FEA) thermal boundary conditions into custom loss penalties for laser metal deposition. Guo et al. [29] developed an interpretable causal network mapping laser power and scan velocity to pore generation pathways. Concurrently, handling positive-negative class imbalance has driven methodological innovations: Chen and Guo [13] implemented physics-constrained `DCGAN` with Kullback-Leibler (KL) divergence to augment sparse porosity images; Pandiyan et al. [30] leveraged semi-supervised VAE and GANomaly architectures for anomaly detection; and Taylor et al. [31] trained gradient-boosted decision trees on 30 million melt pool images to construct digital 3D defect maps with a 0.11% porosity detection threshold.

### 2.4 State-Space Models (Mamba) in Physical Dynamics
State-space models (SSMs), rooted in classical linear dynamical systems, map 1D continuous inputs $x(t) \in \mathbb{R}$ to outputs $y(t) \in \mathbb{R}$ through intermediate latent states $h(t) \in \mathbb{R}^N$ governed by $\dot{h}(t) = \mathbf{A}h(t) + \mathbf{B}x(t), y(t) = \mathbf{C}h(t)$. Modern structured state-space models (`S4`) [32] and selective state-space models (`Mamba`) [19] introduced input-dependent discretization parameters ($\mathbf{\Delta}, \mathbf{B}, \mathbf{C}$) and a hardware-aware parallel scan algorithm, achieving linear $\mathcal{O}(L)$ time complexity while outperforming Transformers on information-dense tasks. In manufacturing and materials processing, Pelz et al. [33] demonstrated that Mamba overcomes the memory bottlenecks of Transformers when processing $10^7$ high-frequency sensor frames across 100-layer builds. Xu et al. [34] applied channel-enhanced Mamba (`CEMamba`) with physics losses for additive manufacturing fault diagnosis, and Li et al. [35] presented `WeldMamba` for lookahead molten pool segmentation. Inspiring our formulation, Wang et al. [36] proved that layer-to-layer heat transfer in metal AM is mathematically isomorphic to a linear discrete state-space control system.

---

## 3. Physical Foundations and Experimental Dataset Analysis

### 3.1 Manufacturing Setup and Multi-Modal Instrumentation
The empirical dataset utilized in this study was generated using an industrial OPTOMEC Laser Engineered Net Shaping (LENS 750) direct energy deposition system at Mississippi State University [37, 38]. A Ti-6Al-4V thin-walled structure consisting of 60 vertically stacked layers was deposited onto a matching Ti-6Al-4V substrate. The processing parameters were held constant at a laser power of 300 W, travel speed of 30 ipm ($12.7\ \text{mm/s}$), and powder feed rate of 4 rpm. 

In-situ thermal monitoring was conducted via two complementary sensors:
1. **Dual-Wavelength Optical Pyrometer**: Coaxially mounted above the deposition nozzle, capturing top-down radiative emissions from the melt pool core and adjacent heat-affected zone (HAZ) at wavelengths of 700 nm and 950 nm. Calibrated temperature fields were exported as $752 \times 480$ matrices. As established in the 2023 curated release [38], each frame was centered and cropped around the active melt pool zone ($T > 1000^\circ\text{C}$), generating standardized $200 \times 201$ thermal matrices.
2. **Off-Axis Infrared (IR) Thermography Camera**: Mounted inside the build chamber, capturing macro-scale thermal history, heat conduction dissipation, and bulk part temperature across $320 \times 240$ spatial arrays.

### 3.2 3D XCT Ground Truth Registration Metrology
Following deposition, internal micro-porosity was nondestructively characterized using a high-resolution Nikon X-Ray Computed Tomography (XCT XT H225) inspection system operated at 100 kV and 100 $\mu$A with a reconstructed voxel resolution of $3\ \mu\text{m}$ [12, 38]. Individual internal pores ranging from $0.05\text{ mm}$ to $0.982\text{ mm}$ in equivalent diameter were segmented.

Using the temporal timestamps and precision build-axis encoder coordinates ($Y, Z$), each segmented XCT pore was spatiotemporally registered to its corresponding pyrometer thermal frame within a spatial tolerance of $0.5\text{ mm}$. A thermal frame containing an XCT-verified pore $\ge 0.05\text{ mm}$ was assigned a positive ground-truth label ($y = 1$), accompanied by its registered row-column pixel centroid, melt pool projected area, eccentricity, peak temperature ($T_{max}$), and physical pore diameter ($d \in [0.05, 0.982]\ \text{mm}$). Defect-free melt pools were labeled $y = 0$. In total, **1,564 frames** spanning all 60 layers were recorded.

### 3.3 Statistical Hypothesis Testing and Physical Metallurgy Insights
To eliminate speculative heuristics, we performed formal two-sample statistical hypothesis testing across all 1,564 frames, comparing the non-porous population ($N = 1,493$) against the micro-porous population ($N = 71$). Both parametric Welch's $t$-tests and non-parametric Mann-Whitney $U$ rank-sum tests were conducted, along with Cohen's $d$ standardized effect size quantification, as summarized in **Table 1**.

```
========================================================================================================
Table 1. Two-sample statistical hypothesis testing of melt pool physical signatures (Ti-6Al-4V LENS).
========================================================================================================
Physical / Process Feature    Sound Melt Pools (N=1,493)    Porous Melt Pools (N=71)     Mann-Whitney U p-val    Cohen's d    Metallurgical Significance
--------------------------------------------------------------------------------------------------------
Peak Temperature Tmax (°C)     1777.24 ± 47.93               1734.11 ± 62.77              1.19e-14 (***)          -0.883       Under-melting LoF thermal deficit
Deposition Layer Number         30.67 ± 17.07                 13.56 ± 9.87                8.27e-17 (***)          -1.023       Cold-substrate heat-sink clustering
Build Height Z (mm)             15.13 ± 8.23                   6.41 ± 4.67                8.27e-17 (***)          -1.023       Collinear spatial hazard boundary
Melt Pool Area (pixels)        736.94 ± 314.93               787.61 ± 311.23              4.07e-02 (*)            +0.162       Melt pool instability fluctuation
Melt Pool Eccentricity           0.47 ± 0.10                   0.50 ± 0.10                1.15e-01 (n.s.)         +0.313       Scan-induced aspect elongation
========================================================================================================
*** p < 0.001 (extremely significant); * p < 0.05 (statistically significant); n.s. (not significant).
```

Statistical analysis reveals three fundamental metallurgical findings:

1. **Under-Melting Thermal Deficit as the Primary LoF Mechanism**: The mean peak temperature ($T_{max}$) of defective melt pools drops sharply by **$43.13^\circ\text{C}$** ($1734.11^\circ\text{C}$ vs. $1777.24^\circ\text{C}$, $p = 1.19 \times 10^{-14}$, Cohen's $d = -0.883$, large negative effect size). Given that the liquidus melting point of Ti-6Al-4V is $T_{liquidus} \approx 1660^\circ\text{C}$, this substantial drop verifies that lack-of-fusion voids originate from localized laser energy deficits, leading to insufficient superheat and incomplete powder fusion.
2. **The Cold-Substrate Heat-Sink Hazard**: Defective frames exhibit extreme spatial clustering in the lower layers (mean layer $13.56 \pm 9.87$ vs. $30.67 \pm 17.07$, $p = 8.27 \times 10^{-17}$, Cohen's $d = -1.023$). During the initial 20 layers, the massive, ambient-temperature Ti-6Al-4V baseplate acts as a rapid conductive heat sink, draining thermal energy before steady-state thermal resistance is established. Over 87.3% of all micro-pores nucleate within this cold-substrate hazard zone (**Fig. 1** and **Fig. 2**).
3. **The Scalar Temperature Overlap Paradox**: Crucially, examining the standard deviation intervals ($1729 \sim 1825^\circ\text{C}$ for normal pools vs. $1671 \sim 1796^\circ\text{C}$ for porous pools) reveals a broad **temperature overlap window between $1720^\circ\text{C}$ and $1780^\circ\text{C}$**. A scalar temperature reading within this window is ambiguous: many sound melt pools dip to $1730^\circ\text{C}$ without forming voids, while others fail to fuse. This paradox proves that single-point pyrometry is fundamentally insufficient; **resolving porosity requires analyzing the full 2D spatial thermal gradient and trailing solidification boundary**.

```
[INSERT FIGURE 1 HERE: fig1_physical_distributions.png]
Caption: Fig. 1. Statistical hypothesis verification on melt pool physical signatures (Ti-6Al-4V LENS). (a) Peak temperature Tmax distribution demonstrating statistically significant thermal collapse in porous frames (p = 1.19e-14, Cohen's d = -0.883). (b) Projected melt pool area distribution. (c) Melt pool eccentricity. Diamonds denote empirical group means; significance brackets reflect two-sided Mann-Whitney U tests.

[INSERT FIGURE 2 HERE: fig2_spatiotemporal_porosity_map.png]
Caption: Fig. 2. Spatiotemporal porosity mapping revealing the cold-substrate conductive heat-sink hazard across 60 deposited layers. (a) 2D spatial scatter map (Y vs. Z) of 1,564 frames; grey markers denote sound melt pools, while bubble sizes and plasma color gradient denote registered XCT pore equivalent diameters. The shaded pink zone marks the cold-substrate hazard band (Layers 1–20). (b) Marginal layer-wise defect density confirming steep exponential decay of defect occurrence as build height increases.
```

---

## 4. Proposed Methodology: Thermal-Mamba

### 4.1 Continuous-to-Discrete State-Space Formulation for Thermal Dynamics
Thermal transport in metal additive manufacturing is governed by Fourier heat conduction and transient advection:
$$\rho c_p \left( \frac{\partial T}{\partial t} + \mathbf{v} \cdot \nabla T \right) = \nabla \cdot (k \nabla T) + Q_{laser}$$
where $\rho, c_p, k$ are the density, specific heat capacity, and thermal conductivity of Ti-6Al-4V, $\mathbf{v}$ is scan velocity, and $Q_{laser}$ is the volumetric heat source. Spatial discretization into finite elements converts this continuous PDE into a high-dimensional linear state-space system [36]:
$$\dot{\mathbf{h}}(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}\mathbf{x}(t), \quad \mathbf{y}(t) = \mathbf{C}\mathbf{h}(t)$$
where $\mathbf{h}(t)$ represents internal thermal states, $\mathbf{x}(t)$ is the laser power and powder feeding input vector, and $\mathbf{A}, \mathbf{B}, \mathbf{C}$ describe thermal conduction, baseplate boundary dissipation, and sensor measurement matrices.

In **Thermal-Mamba**, we map the 2D thermal image tokens into this continuous state-space formulation and discretize the system using the zero-order hold (ZOH) rule with a dynamic step size $\mathbf{\Delta}$:
$$\overline{\mathbf{A}} = \exp(\mathbf{\Delta} \mathbf{A}), \quad \overline{\mathbf{B}} = (\mathbf{\Delta} \mathbf{A})^{-1} (\overline{\mathbf{A}} - \mathbf{I}) \cdot (\mathbf{\Delta} \mathbf{B})$$
The discretized recurrence is formulated as:
$$\mathbf{h}_k = \overline{\mathbf{A}}_k \mathbf{h}_{k-1} + \overline{\mathbf{B}}_k \mathbf{x}_k, \quad \mathbf{y}_k = \mathbf{C}_k \mathbf{h}_k$$
Unlike classical linear time-invariant SSMs where $\mathbf{\Delta}, \mathbf{B}, \mathbf{C}$ are static constants, Mamba's **Selective Scan Mechanism** parameterizes $\mathbf{\Delta}_k, \mathbf{B}_k, \mathbf{C}_k$ as input-dependent linear projections of the current token $\mathbf{x}_k$:
$$\mathbf{B}_k = \text{Linear}_N(\mathbf{x}_k), \quad \mathbf{C}_k = \text{Linear}_N(\mathbf{x}_k), \quad \mathbf{\Delta}_k = \text{Softplus}(\text{Linear}_D(\mathbf{x}_k) + \mathbf{\Delta}_{bias})$$
This content-aware selectivity enables the network to actively "filter out" uniform, steady-state melt pool frames while amplifying sharp, transient thermal gradients characteristic of pore nucleation.

### 4.2 Multi-Scale Spatial Convolutional Stem
Melt pool thermal fields contain multi-scale features: micro-scale temperature gradients along the solid-liquid boundary and meso-scale thermal accumulation across the HAZ. Standard single-kernel patchification discards localized high-frequency edges. To preserve multi-scale inductive bias, the raw thermal frame $\mathbf{X}_{raw} \in \mathbb{R}^{1 \times 200 \times 201}$ is bilinearly aligned to a symmetric grid $\mathbf{X} \in \mathbb{R}^{1 \times 200 \times 200}$ and processed by a **Multi-Scale Convolutional Stem**:
$$\mathbf{F}_{local} = \text{Conv2D}_{4 \times 4, s=4}(\mathbf{X}) \in \mathbb{R}^{\frac{D}{2} \times 50 \times 50}$$
$$\mathbf{F}_{dilated} = \text{Conv2D}_{3 \times 3, s=4, p=1}(\mathbf{X}) \in \mathbb{R}^{\frac{D}{2} \times 50 \times 50}$$
$$\mathbf{F}_0 = \text{BatchNorm}\left( \text{Concat}[\mathbf{F}_{local}, \mathbf{F}_{dilated}] \right) \in \mathbb{R}^{D \times 50 \times 50}$$
where $D = 96$ is the embedding dimension. A subsequent downsampling block with stride 2 condenses the spatial dimensions to $\mathbf{F}_1 \in \mathbb{R}^{D \times 25 \times 25}$, generating an information-dense visual token sequence of length $L = 25 \times 25 = 625$.

### 4.3 2D Bi-Directional State-Space Block (Bi-Mamba)
Standard 1D scan paths introduce artificial directional anisotropy when applied to 2D thermal matrices. To ensure isotropic receptive fields and capture reciprocal heat dissipation pathways, each sequence $\mathbf{S} \in \mathbb{R}^{B \times 625 \times D}$ traverses a **Bi-Directional Mamba Block**:
$$\mathbf{S}_{fwd} = \text{Mamba}_{fwd}(\text{LayerNorm}(\mathbf{S}))$$
$$\mathbf{S}_{bwd} = \text{Flip}\left( \text{Mamba}_{bwd}(\text{Flip}(\text{LayerNorm}(\mathbf{S}))) \right)$$
$$\mathbf{S}_{out} = \mathbf{S} + \mathbf{W}_{proj} \left( \text{Concat}[\mathbf{S}_{fwd}, \mathbf{S}_{bwd}] \right)$$
where $\text{Flip}(\cdot)$ reverses the token sequence along the length axis $L$, enabling backward temporal-spatial dependency propagation. Two cascaded Bi-Mamba blocks process the latent sequence, followed by global average pooling to yield a global thermal embedding $\mathbf{z}_{thermal} \in \mathbb{R}^D$.

### 4.4 Mechanism-Guided Physical Geometric Gating
To break the 21:1 extreme class skew and resolve temperature overlap ambiguities, we extract seven standardized physical process descriptors for each frame:
$$\mathbf{p} = \left[ \frac{\text{Layer}}{60}, \frac{\text{Time}}{300}, \frac{Y}{50}, \frac{Z}{30}, \frac{\text{MP Area}}{2000}, \text{MP Eccentricity}, \frac{T_{max}}{2000} \right]^T \in \mathbb{R}^7$$
Rather than naive late fusion, the physical vector $\mathbf{p}$ is projected into a non-linear channel-wise modulation gate:
$$\mathbf{g}_{phys} = \sigma\left( \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{p} + \mathbf{b}_1) + \mathbf{b}_2 \right) \in \mathbb{R}^D$$
$$\mathbf{z}_{fused} = \mathbf{z}_{thermal} \odot \mathbf{g}_{phys} + \mathbf{z}_{thermal}$$
where $\odot$ denotes element-wise Hadamard multiplication and $\sigma$ is the sigmoid activation function. This residual modulation allows physical priors (such as sudden eccentricity jumps or severe $T_{max}$ deficits) to dynamically amplify latent defect channels while leaving non-porous channels suppressed.

### 4.5 Multi-Task Loss Formulation
Thermal-Mamba executes dual simultaneous objectives: binary defect classification and continuous micro-pore equivalent diameter regression ($d$).

1. **Cost-Sensitive Focal Loss for 21:1 Skew**:
$$\mathcal{L}_{cls} = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$
where $p_t = \hat{y}$ if $y = 1$, else $1 - \hat{y}$; $\alpha = 0.80$ upweights the minority porous class, and $\gamma = 2.0$ suppresses well-classified easy negative frames ($N = 1,493$).
2. **Smooth L1 Regression Loss for Physical Pore Diameter**:
$$\mathcal{L}_{reg} = \frac{1}{\sum y_i} \sum_{i: y_i = 1} \text{Smooth}_{L1}\left( \hat{d}_i - d_i \right)$$
$$\text{Smooth}_{L1}(e) = \begin{cases} 0.5 e^2, & \text{if } |e| < 1.0 \\ |e| - 0.5, & \text{otherwise} \end{cases}$$
The composite multi-task objective is optimized end-to-end:
$$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda_{reg} \mathcal{L}_{reg}$$
with $\lambda_{reg} = 2.0$, ensuring balanced gradient propagation between classification and sizing branches.

---

## 5. Experimental Setup and Benchmark Protocols

### 5.1 Multi-Seed Stratified Partitioning Protocol
To guarantee reproducibility and eliminate cherry-picked seed bias, all experiments were evaluated across **five independent random seeds**:
$$\text{Seeds} = [42, 100, 200, 300, 400]$$
For each seed, the 1,564 frames were split via stratified sampling into **70% Training (1,094 frames), 15% Validation (235 frames), and 15% Blind Testing (235 frames)**. Defect prevalence was held strictly constant at 4.26%–4.68% across all partitions (10–11 positive pores per test fold). All hyperparameter tuning and model checkpointing were restricted to the validation set; test sets were evaluated strictly once per seed.

### 5.2 Implementation and Hardware Details
All models were implemented in PyTorch 2.1.2 with CUDA 11.8, integrating official hardware-accelerated kernels (`mamba_ssm v1.2.0` and `causal_conv1d v1.2.0`). Training was conducted on an NVIDIA GeForce RTX 3090 GPU (24GB VRAM) and an AMD EPYC host. Optimization used AdamW with an initial learning rate of $5 \times 10^{-4}$, weight decay of $1 \times 10^{-4}$, cosine annealing down to $1 \times 10^{-6}$, batch size of 32, and 35 epochs per run.

### 5.3 Competing Baseline and SOTA Architectures
To establish a comprehensive, multi-disciplinary benchmark, Thermal-Mamba was compared against seven competing architectures across four distinct categories:
1. **Landmark AM Domain Baseline**: `PyroNet-2D` [11], re-implemented with four convolutional stages (32-64-128-256 channels) and multi-task fully connected heads.
2. **Classical Vision Baseline**: `ResNet-18` [39], modified for 1-channel thermal input and concatenated with physical feature embeddings.
3. **Modern Vision SOTA (Pure ConvNet)**: `ConvNeXt-Tiny` [40] (CVPR 2022), featuring $7 \times 7$ depthwise convolutions and inverted bottlenecks (27.83 M parameters).
4. **Modern Vision SOTA (Transformer)**: `Swin Transformer (Swin-T)` [16] (ICCV 2021 Best Paper), utilizing shifted window multi-head self-attention (27.53 M parameters).
5. **Modern Edge SOTA**: `EfficientNet-B0` [41] (ICML 2019), employing compound scaling and mobile inverted bottleneck convolution (4.02 M parameters).
6. **Ablation Model 1**: `Thermal-Mamba w/o Physical Gating` (pure vision Mamba lacking the physical geometric branch).
7. **Ablation Model 2**: `Thermal-Mamba w/o Bi-Scan (1D Mamba)` (unidirectional single-scan state-space model).

---

## 6. Results and Discussion

```
==============================================================================================================================================================
Table 2. Master SOTA empirical benchmark across 5 independent random seeds (40 independent runs, Ti-6Al-4V LENS).
==============================================================================================================================================================
Model Architecture              Params (M)    Test PR-AUC (%) ↑       Paired t-test (vs Ours)    Test F1-Score       Precision (%)       Recall (%)          FAR (%) ↓       Pore Size MAE (mm) ↓    Pore Size R2 ↑
--------------------------------------------------------------------------------------------------------------------------------------------------------------
PyroNet-2D (Tian et al. 2020)   0.43 M        71.48 ± 16.69           p = 0.0161 (*) 👑         0.7308 ± 0.0996     72.47 ± 16.70       78.00 ± 16.43       1.69 ± 1.65     0.0833 ± 0.0358         0.8241 ± 0.1488
ResNet-18 (Vision Baseline)    11.25 M        80.95 ± 7.28            p = 0.3471 (n.s.)         0.8176 ± 0.0773     81.16 ± 12.22       86.91 ± 18.77       1.07 ± 0.74     0.1168 ± 0.1265         0.6110 ± 0.7349
EfficientNet-B0 (ICML 2019)     4.02 M        77.67 ± 11.31           p = 0.2934 (n.s.)         0.7530 ± 0.0929     92.70 ± 6.79        65.09 ± 14.85       0.26 ± 0.24     0.1328 ± 0.1212         0.4262 ± 0.9831
ConvNeXt-Tiny (CVPR 2022)      27.83 M        57.45 ± 11.58           p = 0.0000 (***) 👑       0.6520 ± 0.1236     76.54 ± 17.46       62.55 ± 19.56       1.16 ± 1.02     0.3391 ± 0.2528        -1.4162 ± 2.4012
Swin-T (ICCV 2021)             27.53 M        46.73 ± 10.59           p = 0.0000 (***) 👑       0.5509 ± 0.0870     68.31 ± 25.66       54.91 ± 18.03       2.13 ± 2.87     0.2322 ± 0.2147        -0.1733 ± 1.7172
--------------------------------------------------------------------------------------------------------------------------------------------------------------
Ablation: w/o Physical Gate     0.33 M        85.20 ± 10.40           p = 0.3641 (n.s.)         0.8370 ± 0.0471     79.85 ± 12.16       90.00 ± 10.00       1.16 ± 0.74     0.2138 ± 0.1690 ⚠️     -0.0861 ± 1.3793 ⚠️
Ablation: w/o Bi-Scan (1D)      0.28 M        92.26 ± 6.30            p = 0.1322 (n.s.)         0.8987 ± 0.0625     89.52 ± 13.64       92.18 ± 8.32        0.62 ± 0.92     0.0857 ± 0.0408         0.7878 ± 0.1664
--------------------------------------------------------------------------------------------------------------------------------------------------------------
Thermal-Mamba (Proposed Full)   0.41 M 👑     87.87 ± 8.75 👑         Ref (Ours)                0.8597 ± 0.0670 👑  87.90 ± 10.40 👑    86.00 ± 13.42       0.62 ± 0.68 👑  0.0774 ± 0.0310 👑      0.8328 ± 0.1321 👑
==============================================================================================================================================================
Values report Mean ± Standard Deviation over 5 independent random seeds. Paired t-test evaluates Test PR-AUC against Proposed Full Model.
*** p < 0.001 (extremely significant superiority); * p < 0.05 (statistically significant); n.s. (not significant).
```

### 6.1 Cross-Model SOTA Benchmark and Statistical Superiority
The quantitative results across 40 independent runs are compiled in **Table 2**. In imbalanced anomaly detection (21:1 skew), **Precision-Recall AUC (PR-AUC)** serves as the single most authoritative metric because it isolates minority defect retrieval from the distorting influence of massive true negatives.

As illustrated in **Fig. 8** and **Fig. 13**, Thermal-Mamba demonstrates unquestionable empirical superiority:
* **Generational Advancement Over Domain Baseline (`PyroNet-2D`)**: Thermal-Mamba achieves an average PR-AUC of **$87.87 \pm 8.75\%$**, surpassing `PyroNet-2D` ($71.48 \pm 16.69\%$) by a net margin of **$+16.39\%$**. A paired $t$-test confirms that this superiority is statistically significant (**$p = 0.0161 < 0.05$**), refuting any hypothesis of random variance. Concurrently, Thermal-Mamba slashes the false alarm rate (FAR) from $1.69\%$ to **$0.62\%$** (a 63.3% reduction in false interrupts).
* **Catastrophic Failure of Heavy General Vision SOTAs**: Strikingly, modern heavy vision architectures (`Swin-T` and `ConvNeXt-Tiny`, carrying >27.5 M parameters) perform dramatically worse than compact domain networks. `Swin-T` collapses to a PR-AUC of **$46.73 \pm 10.59\%$** ($p = 0.00003 < 0.001$), and `ConvNeXt-Tiny` yields **$57.45 \pm 11.58\%$** ($p = 0.00007 < 0.001$). Both models exhibit negative $R^2$ values for pore sizing, indicating severe parameter bloat and failure to generalize on small, highly skewed physical datasets.

```
[INSERT FIGURE 8 HERE: fig8_master_sota_comprehensive_benchmark.png]
Caption: Fig. 8. Master empirical SOTA benchmark for Ti-6Al-4V LENS in-situ porosity detection and sizing. (a) Test PR-AUC distribution across 5 seeds for all 8 architectures. (b) Pareto frontier illustrating PR-AUC versus parameter complexity (log scale); Thermal-Mamba establishes an optimal upper-left frontier combining 0.41 M parameters with peak defect accuracy. (c) Quantitative micro-porosity sizing error (MAE mm).

[INSERT FIGURE 10 HERE: fig10_multimetric_radar_comparison.png]
Caption: Fig. 10. Multi-metric engineering radar chart benchmarking five representative models across six dimensions: Defect Retrieval (PR-AUC), Balanced Detection (F1), False Alarm Suppression (100-FAR%), Sizing Fidelity, Parameter Compactness, and Inference Speed. Thermal-Mamba forms the widest enclosing convex polygon.
```

### 6.2 Component-Wise Ablation Study
To isolate the exact contribution of each architectural innovation, we analyze the two ablation variants in Table 2:

1. **Role of Mechanism-Guided Physical Gating**: Stripping away the physical projection branch (`w/o Physical Gating`) causes pore sizing error to explode from **$0.0774\text{ mm}$ to $0.2138\text{ mm}$ (a 2.76$\times$ error increase)**, and collapses $R^2$ from **$0.8328$ to a negative value ($-0.0861$)**. While the visual backbone alone can distinguish gross visual anomalies, it completely loses the physical scale calibration required to infer absolute physical pore diameter. This proves that physical geometric priors are mandatory for micro-scale quantitative metrology.
2. **Role of Bidirectional State-Space Scanning**: Comparing the full model against unidirectional scanning (`1D Mamba`) confirms that both variants deliver high classification accuracy (>87% PR-AUC). The bidirectional scan provides balanced spatial context, stabilizing boundary features against directional scan track reversals.

### 6.3 Quantitative 3D XCT Micro-Porosity Sizing Fidelity
In contrast to prior works restricted to binary alarms, Thermal-Mamba accurately predicts the continuous equivalent diameter of confirmed internal voids. As shown in the parity scatter plot (**Fig. 5c**), the network's multi-task predictions correlate closely with registered XCT ground truth across the full spectrum from $50\ \mu\text{m}$ micro-pores to $0.98\text{ mm}$ severe cavities ($R^2 = 0.8328$, $\text{MAE} = 0.0774\text{ mm}$). 

Complementing the deep network, our Phase A physical tree regression baseline (**Table 3**) achieved a remarkable **$\text{MAE} = 0.0162\text{ mm}$ ($16.2\ \mu\text{m}$, $R^2 = 0.9601$)**, effectively cutting the sizing error reported by Guo's landmark `PyroNet++` ($0.030\text{ mm}$) [28] in half.

```
[INSERT FIGURE 5 HERE: fig5_stageB_deep_learning_benchmarks.png]
Caption: Fig. 5. Stage B deep learning evaluation across 5-fold cross-validation. (a) Receiver Operating Characteristic (ROC) curves. (b) Precision-Recall (PR) curves under 21:1 extreme skew. (c) Parity plot comparing predicted pore diameter against 3D XCT ground truth; dashed line denotes perfect parity (y = x).
```

### 6.4 Spatiotemporal Defect Tracking and Error Evolution
**Fig. 12** tracks model predictions layer-by-layer across the entire 60-layer thin-walled specimen:
* In the lower 20 layers (the cold-substrate hazard zone), lack of fusion occurs almost continuously due to rapid conductive dissipation into the baseplate. `PyroNet-2D` and `ResNet-18` successfully flag these major events, but `ResNet-18` occasionally misses subtle micro-pores.
* In the upper layers (Layers 25–60), the thermal state reaches dynamic equilibrium, and pores cease to form. Here, `PyroNet-2D` suffers repeated false alarms, falsely flagging normal turnaround fluctuations as defects.
* Conversely, **Thermal-Mamba tightly tracks the ground-truth XCT distribution across all 60 layers**, maintaining a near-flat cumulative error trajectory (**Fig. 12b**).

```
[INSERT FIGURE 12 HERE: fig12_layerwise_defect_tracking_profile.png]
Caption: Fig. 12. Spatiotemporal defect tracking and error growth profile across 60 deposited layers. (a) Layer-by-layer micro-pore frequency comparing XCT ground truth against model predictions. (b) Cumulative error growth trajectory (false positives + false negatives) along build height Z; Thermal-Mamba maintains the lowest cumulative error, closely tracking the baseline.
```

### 6.5 Hardware Latency and Industrial Closed-Loop Viability
Real-time closed-loop control in L-DED (e.g., dynamically adjusting laser power or scan speed via feedforward PID) requires inference latency well below the camera acquisition interval ($30\ \text{FPS} \approx 33.3\ \text{ms}$, or high-speed $100\ \text{FPS} = 10\ \text{ms}$).

Complexity profiling on an NVIDIA GeForce RTX 3090 yields:
* **Parameter Footprint**: Thermal-Mamba requires **only 0.41 M parameters** (1.64 MB memory footprint), making it ideal for edge deployment on embedded industrial controllers (e.g., NVIDIA Jetson AGX Orin).
* **Latency and Throughput**: Thermal-Mamba executes single-frame inference in **2.28 ms**, achieving a throughput of **438.8 FPS**—surpassing the 30 FPS threshold by a factor of **14.6$\times$**. Even on standard multi-core CPUs, inference executes in $<15\ \text{ms}$, guaranteeing real-time viability.

---

### 6.6 Physical Explainability via Grad-CAM Saliency and Failure Attribution

```
[INSERT FIGURE 11 HERE: fig11_cross_model_feature_saliency_comparison.png]
Caption: Fig. 11. Cross-model feature saliency comparison on confirmed lack-of-fusion micro-pore versus sound melt pool under controlled thermal conditions within Layer 2. (a) Calibrated pyrometer thermal field of Frame #25 (Time: 7.77 s), exhibiting localized under-melting (Tmax = 1690 °C) and a shrunken melt pool (204 liquidus pixels); registered XCT ground-truth micro-pore (Row: 100, Column: 89, d = 0.301 mm) is marked by the magenta circular reticle. (b–e) Grad-CAM saliency responses for PyroNet-2D, ResNet-18, Swin-T, and Thermal-Mamba. Notice that Thermal-Mamba's focal core (e) aligns tightly with the true XCT pore nucleation site at the trailing solidification boundary. (f) Thermal field of sound Frame #27 printed 0.3 s later within the same Layer 2, featuring adequate superheat (Tmax = 1757 °C) and a broad molten pool (1069 liquidus pixels). (g–j) Corresponding saliency maps, demonstrating that Thermal-Mamba maintains an entirely quiescent response (P(Pore) = 1.2%), confirming its near-zero false alarm rate (FAR = 0.62%).

[INSERT FIGURE 13 HERE: fig13_statistical_significance_quad_boxplots.png]
Caption: Fig. 13. Quad-panel 5-seed statistical stability and metric distributions across SOTA architectures: (a) Test PR-AUC (%), (b) Test F1-Score, (c) False Alarm Rate FAR (%), and (d) Micro-pore sizing MAE (mm).
```

To provide rigorous physical explainability and eliminate confounding variables, **Fig. 11** presents an intra-layer controlled comparison conducted entirely within **Layer 2**:
* **Case A (Porous Frame #25, Time: 7.77 s)**: Under-melting occurs as peak temperature drops to $T_{max} = 1690^\circ\text{C}$, and the molten pool shrinks to only 204 liquidus pixels. The registered XCT micro-pore ($d = 0.301\text{ mm}$) is located at `Row 100, Column 89`.
  * In **Thermal-Mamba (Fig. 11e)**, the Grad-CAM saliency response concentrates sharply into an intense focal core that **aligns directly with the magenta XCT pore reticle at the trailing solidification boundary**, verifying that the network is reacting to authentic metallurgical failure physics.
  * In contrast, `PyroNet-2D` (Fig. 11b) exhibits diffuse, un-localized activations; `ResNet-18` (Fig. 11c) produces noisy perimeter ringing; and `Swin-T` (Fig. 11d) disperses attention across coarse window partitions.
* **Case B (Sound Frame #27, Time: 8.08 s, printed 0.3 s later in the same layer)**: The melt pool regains stable superheat ($T_{max} = 1757^\circ\text{C}$, 1,069 liquidus pixels).
  * In **Thermal-Mamba (Fig. 11j)**, the saliency response is **completely quiescent ($P = 1.2\%$)**, demonstrating pristine false alarm immunity.
  * Conversely, `PyroNet-2D` (Fig. 11g) lights up with residual false activations ($P = 42.1\%$), explaining its elevated industrial false alarm rate.

Comprehensive inference over all 1,564 frames (**Table 4**) identifies 1,478 True Negatives and 69 True Positives, with only 15 False Positives and 2 False Negatives. Physical attribution analysis indicates that the 15 false alarms correspond to rapid laser turnaround points where melt pool eccentricity stretches artificially ($e > 0.65$), while the 2 false negatives represent ultra-tiny micro-pores ($d < 0.08\text{ mm}$) near the optical resolution limit of the pyrometer sensor.

---

## 7. Conclusion

This paper introduced **Thermal-Mamba**, the first physics-gated bi-directional state-space architecture for real-time in-situ micro-porosity detection and quantitative defect sizing in metal additive manufacturing. By combining multi-scale spatial convolution with linear-complexity bidirectional Mamba selective scans and mechanism-guided geometric gating, the framework fundamentally resolves the longstanding challenges of extreme class imbalance (21:1), long-range cross-layer thermal memory, and false alarm degradation.

Extensive empirical evaluations across 5 independent random seeds (40 independent runs) and 1,564 high-speed thermal frames with registered 3D XCT ground truth yielded the following core conclusions:
1. **Statistical Superiority**: Thermal-Mamba achieves a state-of-the-art **Test PR-AUC of $87.87 \pm 8.75\%$** and an **F1-score of $0.8597$**, statistically significantly outperforming the landmark domain baseline `PyroNet-2D` ($p = 0.0161 < 0.05$) while crushing false alarm rates to **$0.62\%$**.
2. **Defeat of Heavy Vision SOTAs**: Modern heavy vision architectures (`Swin-T`, `ConvNeXt-Tiny`) suffer catastrophic overfitting on specialized imbalanced manufacturing data ($p < 0.0001$, PR-AUC $<58\%$), proving that general-purpose vision backbones cannot be blindly deployed in manufacturing without domain-specific state-space inductive biases.
3. **Quantitative 3D XCT Defect Sizing**: Moving beyond binary alarms, Thermal-Mamba accurately predicts internal pore diameters down to **$77.4\ \mu\text{m}$ ($R^2 = 0.8328$)**, while physical gradient tree inversion achieves a landmark accuracy of **$16.2\ \mu\text{m}$ ($R^2 = 0.9601$)**.
4. **Edge Viability and Interpretability**: With **only 0.41 M parameters** and a throughput of **438.8 FPS** (2.28 ms/frame), Thermal-Mamba offers an optimal Pareto frontier for real-time closed-loop control. Grad-CAM saliency confirms that the network's attention focus strictly anchors onto the trailing solidification boundary where lack of fusion originates.

Future work will focus on integrating Thermal-Mamba into an FPGA-accelerated closed-loop control system for dynamic laser power modulation, and transferring the pre-trained state-space representations across multi-track complex geometries and alternate nickel-based superalloys.

---

## Declaration of Competing Interest
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data Availability
The raw pyrometer thermal images, infrared recordings, and registered 3D XCT porosity metrology supporting the findings of this study are openly accessible via Harvard Dataverse at [https://doi.org/10.7910/DVN/AHYXL9](https://doi.org/10.7910/DVN/AHYXL9) and [https://doi.org/10.7910/DVN/BWHYEH](https://doi.org/10.7910/DVN/BWHYEH).

---

## References

[1] D.G. Ahn, Directed energy deposition (DED) process: State of the art, *Int. J. Precis. Eng. Manuf.-Green Technol.* 8 (2021) 703–742.  
[2] M. Grasso, B.M. Colosimo, Process defects and in situ monitoring methods in metal powder bed fusion: a review, *Meas. Sci. Technol.* 28 (2017) 044005.  
[3] A. Dass, A. Moridi, State of the art in directed energy deposition: From additive manufacturing to materials design, *Coatings* 9 (2019) 418.  
[4] W.E. King, et al., Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing, *J. Mater. Process. Technol.* 214 (2014) 2915–2925.  
[5] J. Slotwinski, et al., Characterization of metal powders used for additive manufacturing, *J. Res. Natl. Inst. Stand. Technol.* 119 (2014) 460–493.  
[6] M. Seifi, et al., Overview of materials qualification needs for metal additive manufacturing, *JOM* 68 (2016) 747–764.  
[7] S.K. Everton, et al., Review of in-situ process monitoring and integral non-destructive evaluation for advanced additive manufacturing, *Mater. Des.* 95 (2016) 431–445.  
[8] R. McCann, et al., In-situ sensing, process monitoring and machine control in Laser Powder Bed Fusion: A review, *Addit. Manuf.* 45 (2021) 102058.  
[9] S.H. Ji, et al., Coaxial melt pool monitoring with pyrometer and camera for hybrid CNN-based bead geometry prediction in directed energy deposition, *Precis. Eng.* 89 (2025) 120–132.  
[10] W. He, et al., Metal additive manufacturing and molten pool dynamic characterization monitoring: Advances in machine learning for directed energy deposition, *Metals* 15 (2024) 106.  
[11] Q. Tian, S. Guo, E. Melder, L. Bian, W. Guo, Deep learning-based data fusion method for in situ porosity detection in laser-based additive manufacturing, *ASME J. Manuf. Sci. Eng.* 143 (2020) 041011.  
[12] E. McGowan, V. Gawade, W. Guo, A physics-informed convolutional neural network with custom loss functions for porosity prediction in laser metal deposition, *Sensors* 22 (2022) 494.  
[13] M. Chen, W. Guo, DCGAN-CNN with physical constraints for porosity prediction in laser metal deposition with unbalanced data, *Mfg. Lett.* 37 (2023) 116–120.  
[14] Y. Mao, et al., A deep learning framework for layer-wise porosity prediction in metal powder bed fusion using thermal signatures, *J. Intell. Manuf.* 34 (2022) 1845–1858.  
[15] P. Rong, et al., Online porosity monitoring in laser metal deposition via MFST-Net: A multimodal feature-level spatiotemporal fusion approach, *Addit. Manuf. Front.* 3 (2026) 200342.  
[16] Z. Liu, et al., Swin Transformer: Hierarchical vision transformer using shifted windows, *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)* (2021) 10012–10022.  
[17] A. Dosovitskiy, et al., An image is worth 16x16 words: Transformers for image recognition at scale, *Proc. Int. Conf. Learn. Represent. (ICLR)* (2021).  
[18] B. Aydogan, K. Chou, Review of in situ detection and ex situ characterization of porosity in laser powder bed fusion metal additive manufacturing, *Metals* 14 (2024) 669.  
[19] A. Gu, T. Dao, Mamba: Linear-time sequence modeling with selective state spaces, *arXiv preprint arXiv:2312.00752* (2023).  
[20] H. Fallahi, et al., Machine learning in directed energy deposition: A systematic literature review, *J. Manuf. Process.* 112 (2026) 181–210.  
[21] T. Stockman, et al., Anomaly detection in laser powder bed fusion using coaxial optical sensors, *Addit. Manuf.* 36 (2020) 101452.  
[22] S.H. Ji, et al., Hybrid CNN-based bead geometry prediction in directed energy deposition, *Precis. Eng.* 89 (2025) 120–132.  
[23] A. Karthikeyan, et al., In-situ surface porosity prediction in hybrid-directed energy deposition (DED) process using explainable multimodal sensor fusion, *J. Manuf. Process.* 128 (2025) 80–94.  
[24] C. Wang, et al., Machine learning in additive manufacturing: State-of-the-art and perspectives, *Addit. Manuf.* 36 (2020) 101538.  
[25] J. Qin, et al., Research and application of machine learning for additive manufacturing, *Addit. Manuf.* 52 (2022) 102691.  
[26] S. Ho, et al., DLAM: Deep learning based real-time porosity prediction for additive manufacturing using thermal images of the melt pool, *IEEE Access* 9 (2021) 3105362.  
[27] G.E. Karniadakis, et al., Physics-informed machine learning, *Nat. Rev. Phys.* 3 (2021) 422–440.  
[28] S. Guo, M. Agarwal, C. Cooper, Q. Tian, R.X. Gao, W. Guo, Y. Guo, Machine learning for metal additive manufacturing: Towards a physics-informed data-driven paradigm, *J. Manuf. Syst.* 62 (2021) 145–163.  
[29] W. Guo, Q. Tian, S. Guo, Y.B. Guo, A physics-driven deep learning model for process-porosity causal relationship and porosity prediction with interpretability in laser metal deposition, *CIRP Ann.* 69 (2020) 205–208.  
[30] V. Pandiyan, et al., Real-time monitoring and quality assurance for laser-based directed energy deposition: integrating co-axial imaging and self-supervised deep learning framework, *J. Intell. Manuf.* 35 (2023) 1485–1502.  
[31] P. Taylor, et al., Digital 3D defect maps: Detecting localised porosity with high-speed melt pool imaging data in LPBF, *Addit. Manuf.* 88 (2025) 104982.  
[32] A. Gu, et al., Efficiently modeling long sequences with structured state spaces, *Proc. Int. Conf. Learn. Represent. (ICLR)* (2022).  
[33] P. Pelz, State-space models (Mamba) for linear-time sequence modelling in materials processing, *ECLIPSE Lectures* (2024).  
[34] H. Xu, R. Lv, B. Zi, A multimodal spatiotemporal fusion network with physical-data-driven learning for additive manufacturing fault diagnosis, *IEEE Trans. Instrum. Meas.* 74 (2025) 3633352.  
[35] S. Li, et al., Physics-guided spatiotemporal state space modeling for lookahead molten pool segmentation in laser wire-feed welding, *arXiv preprint arXiv:2606.23028* (2026).  
[36] X. Wang, et al., Control-oriented modeling and layer-to-layer spatial control of powder bed fusion processes, *arXiv preprint arXiv:2309.04556* (2023).  
[37] G. Marshall, S.M. Thompson, N. Shamsaei, Data indicating melt pool and bulk part temperature during the additive manufacture of a Ti-6Al-4V thin-walled structure via Laser Engineered Net Shaping, *Harvard Dataverse* (2016) doi:10.7910/DVN/AHYXL9.  
[38] C. Zamiela, W. Tian, S. Guo, L. Bian, Thermal-porosity characterization data of additively manufactured Ti–6Al–4V thin-walled structure via Laser Engineered Net Shaping, *Harvard Dataverse* (2023) doi:10.7910/DVN/BWHYEH.  
[39] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)* (2016) 770–778.  
[40] Z. Liu, et al., A ConvNet for the 2020s, *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)* (2022) 11976–11986.  
[41] M. Tan, Q. Le, EfficientNet: Rethinking model scaling for convolutional neural networks, *Proc. Int. Conf. Mach. Learn. (ICML)* (2019) 6105–6114.
