# Rebuild every table, note and figure from data/raw/ (after `make data`).
.PHONY: data panel stack all test external replication
# single-threaded BLAS, low priority: keeps a laptop cool (see README)
# data cutoff: outputs are reproduced as of this RTFP release date, even after newer releases appear
export RTFP_AS_OF ?= 2026-10-05

PY := VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 5 .venv/bin/python -I

data:
	$(PY) -m rtfp.download --root data/raw

external:
	$(PY) -m rtfp.bbs download && $(PY) -m rtfp.bbs build
	$(PY) -m rtfp.tcb --out data/external/tcb

panel:
	$(PY) -m rtfp.panel

stack:
	$(PY) -m rtfp.vintages

all: panel stack
	$(PY) analysis/a1_observed_flag_audit.py
	$(PY) analysis/a2_holdout_counts.py
	$(PY) -c "import runpy; runpy.run_path('analysis/a2_holdout_counts.py', run_name='lib')['write_note']()"
	$(PY) analysis/t1_noise_gate.py
	$(PY) analysis/t2_levels.py
	$(PY) analysis/t3_local_changes.py
	$(PY) analysis/t4_error_covariates.py
	$(PY) analysis/t7_uncertainty.py
	$(PY) analysis/f7_bbs_national.py
	$(PY) analysis/a5_tcb_dhaka.py
	$(PY) analysis/t5_switchers.py --boot 1000
	$(PY) analysis/a4_simulation.py
	$(PY) analysis/t6_alerts.py
	$(PY) analysis/notes.py
	$(PY) analysis/t8_safe_use.py
	$(PY) analysis/figures.py

# five-country replication (downloads ~2.3 GB; run separately)
replication:
	$(PY) analysis/t9_replication.py all

test:
	.venv/bin/python -m pytest -q
