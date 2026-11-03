include ../../../common/Makefile.problem

.PHONY: baseline-test baseline-all student-baseline-test baseline-plan competition-baseline competition-evaluate competition-score compact-reference-test
baseline-test: all
	$(PYTHON) runner.py --case ../test/test1.in --baseline
baseline-plan: all
	$(PYTHON) inspect_plan.py
baseline-all: all
	@failed=0; for case in ../test/test*.in; do $(PYTHON) runner.py --case "$$case" --baseline --warmups 0 --repeats 1 || failed=1; done; exit $$failed
student-baseline-test: all
	@failed=0; for case in ../test/test*.in; do $(PYTHON) runner.py --case "$$case" --student-baseline --warmups 0 --repeats 1 || failed=1; done; exit $$failed
competition-baseline: all
	$(PYTHON) evaluate.py --baseline
competition-evaluate: all
	$(PYTHON) evaluate.py
competition-score: all
	$(PYTHON) score_compact.py --result suite/run/compact-candidate/report.json --baseline suite/run/compact-baseline/report.json
compact-reference-test: all
	@failed=0; for case in ../test/student*.in; do $(PYTHON) runner.py --case "$$case" --reference || failed=1; done; exit $$failed
