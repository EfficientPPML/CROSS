include ../../../common/Makefile.problem

WISE_ROOT ?= $(abspath ../../../../../WISE)
export WISE_ROOT
.PHONY: install-wise cpu-test tpu-test cpu-evaluate tpu-evaluate
install-wise:
	$(PYTHON) wise_source.py "$(WISE_ROOT)"
cpu-test: all
	$(PYTHON) runner.py --case ../test/test1.in --backend cpu
tpu-test: all
	$(PYTHON) runner.py --case ../test/test1.in --backend tpu
cpu-evaluate: all
	$(PYTHON) evaluate.py --backend cpu --output run/cpu-bsgs
tpu-evaluate: all
	$(PYTHON) evaluate.py --backend tpu --output run/tpu-bsgs
