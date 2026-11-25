#!/bin/bash
PROJECT_DIR="/root/autodl-tmp/Ti6Al4V_LENS_project"
cd $PROJECT_DIR
export PATH=/root/miniconda3/bin:$PATH

echo "[$(date)] Queue watcher started. Waiting for run_rigorous_5seed_suite.py to finish..."

while ps aux | grep run_rigorous_5seed_suite.py | grep -v grep > /dev/null; do
    sleep 10
done

echo "[$(date)] run_rigorous_5seed_suite.py finished! Starting expanded SOTA suite (Swin-T, ConvNeXt-Tiny, EfficientNet-B0)..."
python -u run_sota_models_5seed.py > sota_models.log 2>&1
echo "[$(date)] SOTA models suite finished with exit code $?!"

echo "[$(date)] Starting master merge and figure generation..."
python -u merge_all_sota_and_mamba_benchmarks.py > merge.log 2>&1
echo "[$(date)] Master benchmark pipeline complete!"

echo "[$(date)] Starting Module 7 Grad-CAM and failure mode analysis..."
python -u run_module7_interpretability_and_failure_analysis.py > module7.log 2>&1
echo "[$(date)] ALL 7 RESEARCH MODULES 100% EXECUTED AND COMPLETED!"
