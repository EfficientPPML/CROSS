include ../../../common/Makefile.problem

.PHONY: baseline-test baseline-plan
baseline-test: all
	$(PYTHON) runner.py --case ../test/test1.in --baseline
baseline-plan: all
	$(PYTHON) inspect_plan.py
