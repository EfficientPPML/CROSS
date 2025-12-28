# Artifacts Evaluation Step-by-step guidance.

Use this checklist to review and reproduce the reported artifacts.

## How to run it?
```bash
conda activate jaxite
cd <CROSS>/jaxite_word
python3 <script>.py
```
where `<script>` could take from `tabV`, `tabVI`, `tabVII`, `tabVIII`, `tabIX`.
- `tabV.py` — measures latency of processing high-precision modular integer multiplication
as low-precision matrix multiplication to use MXU available in TPU. We compare proposed
Basis Aligned Transformation (BAT) and compare it against the SoTA GPU librarby across
parameter sets used in Tab. IV.
- `tabVI.py` — benchmarks the latency of basis conversion under various test cases. We
again compare the proposed BAT against the SoTA vectorized arithmetic used in SoTA
GPU's libraries under various configurations (Tab. V).
- `tabVII.py` — evaluates NTT throughput under different degrees.
- `tabVIII.py` — profiles latency of homomorphic encryption operators under
various security parameters of the prior works.
- `tabIX.py` — profiles latency of all built-in kernels used in packed bootstrapping,
and estimate its overall latency.

## How to reproduce figure?
- Run each individual script to obtain the results table, and then
- run `<CROSS>/figure_drawer/figure_drawer.ipynb` to redraw all these figures.

## Expected runtimes on TPUv6e (per test case)
- `tabV.py` — ~5 minutes.
- `tabVI.py` + `fig11.py` — ~30 minutes total for NTT throughput and plotting.
- `tabVII.py` — ~10 minutes.
- `source tabVIII.sħ` — ~2 hours.
- `tabIX.py` — ~15 minutes for packed bootstrapping estimates.
