include ../../../common/Makefile.problem

.PHONY: benchmark-check benchmark-stage
benchmark-check:
	$(PYTHON) stage_benchmark.py --upstream "$(BERT_ROOT)" --check
benchmark-stage:
	$(PYTHON) stage_benchmark.py --upstream "$(BERT_ROOT)" --destination "$(BERT_RUN)"

.PHONY: preflight
preflight:
	$(PYTHON) runner.py --preflight
