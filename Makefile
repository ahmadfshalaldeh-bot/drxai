PY ?= python

all: reproduce tables decision calibration qc audit

reproduce:
	$(PY) src/reproduce_metrics.py --bootstrap-replicates 1000 --seed 2026

tables:
	$(PY) src/recompute_table_uncertainty.py --replicates 1000 --seed 2026
	$(PY) src/complete_extended_bootstrap_intervals.py

decision:
	$(PY) src/ordinal_risk_decision_analysis.py --root . --replicates 1000 --seed 2026

calibration:
	$(PY) src/nested_ordinal_calibration.py --root . --seed 2026
	$(PY) src/paired_group_bootstrap_calibration.py --root . --replicates 1000 --seed 2026
	$(PY) src/seed_calibration_sensitivity.py --seeds 2026 2027 2028 2029 2030

qc:
	$(PY) src/qc_sensitivity_bootstrap.py --root . --replicates 1000 --seed 2026

audit:
	$(PY) audit/independent_audit.py . audit/outputs

figure6:
	$(PY) src/make_confusion_matrix_figure.py results/raw/oof_predictions.csv submission_figures/Figure_6_Confusion_Matrices.png

verify:
	sha256sum -c checksums.sha256

check:
	$(PY) -m py_compile src/*.py audit/independent_audit.py
