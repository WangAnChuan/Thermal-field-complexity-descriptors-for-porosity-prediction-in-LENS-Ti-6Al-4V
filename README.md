# Ti-6Al-4V LENS 增材制造原位热监测与 XCT 微孔隙反演研究全景指南

本目录为专门针对 **Ti-6Al-4V 钛合金激光定向能量沉积（LENS / DLD）原位熔池监测与三维微观孔隙缺陷预测** 设立的独立过程性文件库。

---

## 一、 目录组织架构

```text
Ti6Al4V_LENS_Porosity_Study/
├── README.md                                       # 本指南主说明文档
│
├── 01_scripts/                                     # 关键研发与自动化脚本
│   ├── download_harvard_datasets.py                # Harvard Dataverse 自动化下载与哈希校验脚本
│   ├── fast_download_and_scp.py                    # 本地多线程加速下载与跨洋传输流水线
│   ├── run_stepA_eda_and_tabular_baselines.py      # 阶段 A：纯 CPU 物理特征挖掘与机器学习基线评测脚本
│   └── scp_to_path.sh                              # 免密安全 SCP 自动化传输工具
│
├── 02_eda_and_baselines/                           # 阶段 A：数据探索性分析、统计假设检验与机器学习基线
│   ├── reports/                                    # 统计与评测 CSV 数据表格
│   │   ├── physical_features_statistical_tests.csv # 熔池物理量双样本统计检验（曼-惠特尼 U、Cohen's d）
│   │   ├── tabular_machine_learning_benchmarks.csv # 6 种分类器在 21:1 样本失衡下的 5 折交叉验证指标
│   │   ├── pore_size_regression_benchmarks.csv     # 71 处真实微孔隙尺寸连续定量回归反演指标
│   │   └── feature_importance_ranking.csv          # 随机森林与 XGBoost 特征重要度量化排序
│   │
│   ├── figures/                                    # 300 DPI 顶刊级出版质量矢量位图
│   │   ├── fig1_physical_distributions.png         # 正常 vs 孔隙熔池特征箱线对比图（含 p 显著性标注）
│   │   ├── fig2_spatiotemporal_porosity_map.png    # 60 层打印空间 (Y, Z) 孔隙时空分布气泡图
│   │   ├── fig3_baseline_roc_pr_curves.png         # 6 种机器学习算法 ROC 与 PR 曲线对比
│   │   └── fig4_feature_importance_and_correlation.png # 特征相关性热力矩阵与特征贡献条形图
│   └── stepA_pack.tar.gz                           # 阶段 A 全量成果远程打包归档压缩包
│
└── 03_research_notes/                              # 学术理论推演与论文撰写核心笔记
    ├── 01_academic_lineage_and_innovations.md      # 郭维红老师（Prof. Weihong Guo）学术脉络与代际创新规划
    └── 02_phaseA_physical_findings_report.md       # 阶段 A 关键物理机理归因与实测数据分析报告
```

---

## 二、 核心数据集元数据一览

所有原始数据已在服务器端 `/root/autodl-tmp/Ti6Al4V_LENS_datasets/` 完整备份并通过 100% MD5 校验：
* **2016 原生监测集**（DOI: `10.7910/DVN/AHYXL9`）：包含双波长比色高温计（752×480）与红外热像仪（320×240）全过程温度场与视频，共 1.4 GB。
* **2023 处理与标签集**（DOI: `10.7910/DVN/BWHYEH`）：包含 1,564 帧聚焦 >1000℃ 的裁剪熔池切片（$201\times202$）与工业级高精度 XCT（0.05~1.00 mm 未熔合孔隙）时空对齐真值表，共 53 MB。
