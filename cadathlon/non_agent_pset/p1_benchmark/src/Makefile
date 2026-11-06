include ../../../common/Makefile.problem

.PHONY: baseline-test baseline-plan
baseline-test: all
	$(PYTHON) runner.py --case ../test/test1.in --baseline
baseline-plan: all
	$(PYTHON) inspect_plan.py

.PHONY: suite-check suite-install suite-baseline suite-test suite-score \
        suite-gates suite-stop

# The gate self-test is pure arithmetic over synthetic summaries -- no
# accelerator and no suite checkout -- so it is safe to run with `all`.
all: suite-gates
SUITE_DIR := $(abspath suite)
MLINF_ROOT ?= $(HOME)/ml-inference
export MLINF_ROOT
export P3_SUITE_DIR := $(SUITE_DIR)

suite-check:
	$(PYTHON) suite/install_suite.py --check
suite-install:
	$(PYTHON) suite/install_suite.py --install
# One ranked case: SIZE selects the instance (0 single, 1 small, 2 medium),
# P3_DEVICES the chip count. P3_BASELINE=1 runs baseline.py instead of bench_opt.py.
SIZE ?= 0
P3_DEVICES ?= 1
# The competition path uses C3/C4. Select legacy explicitly for six cases.
P1_BASELINE_PROFILE ?= student
export P1_BASELINE_PROFILE
EVALUATION_PROFILE ?= $(P1_BASELINE_PROFILE)
EVALUATION_BUDGET_S ?= 300
BASELINE_LABEL ?= $(if $(filter student,$(P1_BASELINE_PROFILE)),student-baseline,baseline)
CANDIDATE_LABEL ?= $(if $(filter student,$(P1_BASELINE_PROFILE)),student-candidate,candidate)
BASELINE_MANIFEST ?= $(SUITE_DIR)/baseline_manifest$(if $(filter student,$(P1_BASELINE_PROFILE)),_student).json
# The suite keys measurements by instance only, so two cases that share an
# instance overwrite each other; archive each run under its own case id.
suite-baseline: suite-install
	cd $(MLINF_ROOT) && P3_BASELINE=1 P3_DEVICES=$(P3_DEVICES) \
	  $(PYTHON) harness/run_submission.py $(SIZE) --seed 3 --num_runs 3
	$(PYTHON) suite/archive.py --instance $(SIZE) --devices $(P3_DEVICES) \
	  --label $(BASELINE_LABEL) --suite-root $(MLINF_ROOT)
suite-test: suite-install
	cd $(MLINF_ROOT) && P3_DEVICES=$(P3_DEVICES) \
	  $(PYTHON) harness/run_submission.py $(SIZE) --seed 3 --num_runs 3
	$(PYTHON) suite/archive.py --instance $(SIZE) --devices $(P3_DEVICES) \
	  --label $(CANDIDATE_LABEL) --suite-root $(MLINF_ROOT)
suite-gates:
	$(PYTHON) suite/test_gates.py
suite-score:
	$(PYTHON) suite/score.py --suite-root $(MLINF_ROOT) --label $(CANDIDATE_LABEL) --baseline $(BASELINE_MANIFEST) --evaluation-profile $(EVALUATION_PROFILE)
suite-stop:
	cd $(MLINF_ROOT) && $(PYTHON) submissions/mnist/server_stop.py || true

.PHONY: suite-evaluate suite-evaluate-baseline
# Full selected schedule with wall-time budget, per-case archives and cleanup.
suite-evaluate:
	$(PYTHON) suite/evaluate.py --profile $(P1_BASELINE_PROFILE) --budget-s $(EVALUATION_BUDGET_S)
suite-evaluate-baseline:
	$(PYTHON) suite/evaluate.py --profile $(P1_BASELINE_PROFILE) --baseline --budget-s $(EVALUATION_BUDGET_S)
