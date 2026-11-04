PYTHON ?= python3
.PHONY: all reference-test docs release clean
all:
	$(MAKE) -C agent_pset/p1_benchmark/src all
	$(MAKE) -C agent_pset/p2_bert/src all
	$(MAKE) -C agent_pset/p3_cnn/src all
	$(MAKE) -C agent_pset/p4_qwen/src all
reference-test:
	$(MAKE) -C agent_pset/p1_benchmark/src reference-test
	$(MAKE) -C agent_pset/p2_bert/src reference-test
	$(MAKE) -C agent_pset/p3_cnn/src reference-test
	$(MAKE) -C agent_pset/p4_qwen/src reference-test
docs:
	$(PYTHON) common/render_docs.py
release:
	$(PYTHON) common/package.py release .
clean:
	$(MAKE) -C agent_pset/p1_benchmark/src clean
	$(MAKE) -C agent_pset/p2_bert/src clean
	$(MAKE) -C agent_pset/p3_cnn/src clean
	$(MAKE) -C agent_pset/p4_qwen/src clean
