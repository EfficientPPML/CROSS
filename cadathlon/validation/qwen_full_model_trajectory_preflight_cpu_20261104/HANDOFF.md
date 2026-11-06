# Fixed full28 Qwen ideal trajectory prelaunch

This is a new harness around the exact87-source focused runtime, candidate3 manifest `7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333`. The runtime files and two reference implementations are unchanged. No learned whole-model trajectory has been launched by this preflight.

The fixed cases are checkpoint embedding row0 at S1 and the first file-order S8 prefix in the authenticated original calibration token inventory. Original positions are 0 and 0..7. The successful original416 report/archive, fixed checkpoint/config hashes, public export descriptors, and exact input hash gate every body read. Neither fresh nor held-out inputs/outcomes select fixtures. No tokenizer/model forward is needed: inputs are exact rows of the authenticated tied embedding/vocabulary export.

`trajectory_assets.py` owns regular-file NPY descriptors/mappings, validates file/header/raw hashes and performs final identity/digest readback. The small gamma NPZ is authenticated and checked against all113 public descriptors. Bodies remain outside the source archive. Public matrices are mapped and reused; the immutable runtime additionally copies authenticated arrays, including the full vocabulary. No peak-memory or latency budget pass is claimed.

`trajectory_worker.py` first runs independent dense NumPy and unchanged frozen approximate references from the original embeddings. Their all421 operators,28 blocks and151936 logits must agree at fixed absolute1e-8, with all declared nonlinear domains enforced. Reference failure stops that fixture before the packed owner. Neither reference supplies intermediate operands to the packed owner.

The packed run uses actual chained matrix loops and unchanged attention, RMS, softmax, PV, SiLU, tail and full-model owners, with explicit NumPy primitive doubles and identity refresh. The double checks every primitive for finite payloads and checks each real refresh input against its declared bound. Returned refresh scales alternate adjacent binary64 values; all runtime metadata is replayed from those actual scales. The complete paths require638 identity refreshes at S1 and3270 at S8.

`trajectory_observer.py` only reads callback arrays. It checks421 operator outputs,28 full-block outputs and38 vocabulary blocks against independent dense references on every physical65536-slot array, including declared zero padding, at the unchanged absolute1e-8 gate. This is31,916,032 metric slots per completed fixture. Actual RMS variance anchors (including inactive dummy anchors), centered/exp arguments, all softmax reciprocal denominators, SiLU inputs/base arguments and rational denominators are checked against frozen domains. Every callback also rejects nonfinite values. Domain failures are retained, not clipped or bypassed. The final logits are independently rechecked over38 full arrays;151936 valid logits are reported.

`collect_trajectory.py` acquires a fresh output before entering failure-republication scope. It snapshots every source, verifies the tar, builds a new verified execution tree, and runs children there with `-B`, explicit `PYTHONDONTWRITEBYTECODE=1`, and fixed child import path. Both original and execution-tree before/after hashes must equal the tar. Direct-child wait and owned-group absence are separate required receipts. Two fixtures are independently predeclared and each gets its own dedicated process; a failed fixture is retained and cannot be replaced. No healthy timeout is configured. Output-directory collisions cannot rewrite previous evidence.

Preflight runs only synthetic fixtures/metadata/lifecycle checks under the ordinary Python no-body/no-backend audit. It is not OS isolation. The original focused runtime proof remains87/150. Development-v3's13 failures were incomplete mock source inventory after execution-tree hardening; they and post-run source copies are retained without a startup source assertion. No numerical gate was changed.

Scope limits: identity refresh does not prove CKKS/noise accuracy or domain preservation;868 broad score-domain implications remain unproved; zero exp-margin sites remain conditional. No actual encryption, model forward, native benchmark accuracy, file handlers, throughput or wholegraph HE readiness follows from this harness. A separate explicit parent review/grant is required before the actual ideal trajectory command.

Planned command after review only:

```
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JAX_PLATFORMS=cpu taskset -c 172-175 /tmp/cross-pretrained-validation-env/bin/python -B collect_trajectory.py --execute --output /tmp/qwen-full-model-trajectory-20261104/actual-v1
```
