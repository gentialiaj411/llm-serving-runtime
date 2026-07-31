PYTHON ?= python

.PHONY: benchmark-final dashboard-snapshot enforce-slos

benchmark-final:
	$(PYTHON) scripts/run_final_benchmark.py

dashboard-snapshot:
	$(PYTHON) bench/scripts/dashboard_snapshot.py --output-name local-dashboard-snapshot

enforce-slos:
	$(PYTHON) bench/scripts/enforce_slos.py --csv bench/results/runtime-final-local.csv

