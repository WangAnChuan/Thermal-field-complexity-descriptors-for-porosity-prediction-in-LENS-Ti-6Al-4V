# Full pipeline. Tested with Python 3.12 on a 4-core CPU (no GPU required).
# Total runtime is roughly 35 minutes, dominated by the multi-seed study and the CNN.

PY ?= python

.PHONY: all data baselines protocols ablations stability robustness reproduction uncertainty interpretability controls protocol_position entropy_robustness pore_size cnn figures test clean

all: data baselines protocols ablations stability robustness reproduction uncertainty interpretability controls protocol_position entropy_robustness pore_size cnn figures test

data:
	$(PY) experiments/01_build_cache.py

baselines:
	$(PY) experiments/02_baselines.py

protocols:
	$(PY) experiments/03_protocol_comparison.py

ablations:
	$(PY) experiments/04_ablation_groups.py
	$(PY) experiments/05_ablation_combinations.py

stability:
	$(PY) experiments/06_stability.py

robustness:
	$(PY) experiments/07_robustness_first_layers.py
	$(PY) experiments/08_layer_level.py

reproduction:
	$(PY) experiments/09_icinco_and_cost.py
	$(PY) experiments/10_abstention.py

uncertainty:
	$(PY) experiments/11_uncertainty.py
	$(PY) experiments/12_oneclass_svm.py

interpretability:
	$(PY) experiments/13a_shap.py
	$(PY) experiments/13b_shap_excluding_first_layers.py

entropy_robustness:
	$(PY) experiments/19_entropy_robustness.py

pore_size:
	$(PY) experiments/20_pore_size.py

protocol_position:
	$(PY) experiments/18_protocol_position_holdout.py

controls:
	$(PY) experiments/16_controls.py
	$(PY) experiments/17_controls_paired.py

cnn:
	$(PY) experiments/14_cnn.py

figures:
	$(PY) experiments/21_figures.py

test:
	$(PY) tests/test_claims.py

clean:
	rm -rf results/*.csv results/*.json figures/*.png
