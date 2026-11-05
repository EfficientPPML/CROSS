# Run P5 on another TPU host

This branch is self-contained for CROSS execution. WISE remains a separate,
pinned dependency. Use an exclusively allocated TPU host; the commands below
do not stop other jobs or modify device locks. No P5 TPU speedup has yet been
verified. The historical CPU results in `VALIDATION.md` are development
measurements, not measurements of this new PR checkout.

## 1. Checkout and environment

```bash
git clone --branch feat/p5-wise-tpu-20261105 https://github.com/EfficientPPML/CROSS.git CROSS-p5
cd CROSS-p5
export CROSS_ROOT="$PWD"
export P5="$CROSS_ROOT/cadathlon/agent_pset/p5_wise"
export WISE_ROOT="$(dirname "$CROSS_ROOT")/WISE-p5"
```

Use the TPU Python environment from the CROSS root README. In a fresh Python
3.13 virtual environment, its installation plus the planner/test dependencies is:

```bash
python -m pip install 'jax[tpu]' absl-py gmpy2 pandas numpy scipy torch einops matplotlib tqdm pytest sympy
make -C "$P5/src" install-wise WISE_ROOT="$WISE_ROOT" PYTHON=python
```

`install-wise` requires a new destination. If WISE is already present, set
`WISE_ROOT` to it and check that its HEAD is
`485655782dc331e6ebb48b159acac8ba924a9933`. The runner rejects modified planner
sources. The unchanged original checkpoint and MNIST test assets are checked
by hash for pretrained runs. OpenFHE is not needed for CROSS CPU/TPU execution.

The development CPU environment used Python 3.13.18, JAX/jaxlib 0.12.0.1,
libtpu 0.0.50.1, NumPy 2.5.4 and CPU PyTorch 2.15.0. Record the actual environment
on the new host rather than assuming those versions or its performance.

```bash
mkdir -p "$CROSS_ROOT/p5-results"
python -m pip freeze > "$CROSS_ROOT/p5-results/environment.txt"
git rev-parse HEAD > "$CROSS_ROOT/p5-results/commit.txt"
git diff --exit-code
JAX_PLATFORMS=tpu python -c 'import jax; d=jax.devices(); assert d and all(x.platform == "tpu" for x in d); print(d)'
```

That device check initializes TPU; run it only after obtaining the allocation.
The collectors explicitly request TPU and reject CPU fallback. P5 currently
executes on one selected device and reports both available and executed devices.

## 2. Verify the checkout and collect matched measurements

Run these commands from the CROSS root. Choose CPU cores within the host's
allowed affinity mask (`taskset -pc $$`) and keep the allocation recorded.
Each output directory must be new; use a different name for each attempt.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CPU_CORES=0-7  # adjust to the allocated host's allowed cores
JAX_PLATFORMS=cpu make -C cadathlon all reference-test PYTHON=python
JAX_PLATFORMS=cpu CROSS_TEST_WISE_HE=1 python -m pytest -q cadathlon/tests/test_wise*.py

taskset -c "$CPU_CORES" python "$P5/src/evaluate.py" \
  --backend cpu --wise-root "$WISE_ROOT" --convolution tiled-toeplitz \
  --rotation-cache-mib 256 --warmups 1 --repeats 3 \
  --output "$CROSS_ROOT/p5-results/cpu-tiled"
taskset -c "$CPU_CORES" python "$P5/src/evaluate.py" \
  --backend tpu --wise-root "$WISE_ROOT" --convolution tiled-toeplitz \
  --rotation-cache-mib 256 --warmups 1 --repeats 3 \
  --output "$CROSS_ROOT/p5-results/tpu-tiled"

python "$P5/src/score.py" \
  --cpu "$CROSS_ROOT"/p5-results/cpu-tiled/test{1..6}.json \
  --tpu "$CROSS_ROOT"/p5-results/tpu-tiled/test{1..6}.json
```

The braces above require Bash. The scorer requires all six passing cases,
one warmup and at least three samples, matching workloads/schedules/security,
and unchanged source evidence. It recomputes medians from individual requests.
It reports ratios below one when TPU is slower. Keep the entire result
directories: they contain frozen runtime sources, logs, failures, process
receipts and raw timing/quality reports. Never replace a failed attempt.

For tuning, repeat into new directories with `--rotation-cache-mib 512`,
`--linear-chunk 8`, or `--convolution winograd --cases 4 5 6`. Compare layouts
only after their numerical gates pass. Use matching CROSS CPU schedules for
the contest scorer; use the fastest passing native CPU configurations for
broader CPU-system claims.

## 3. Complete pretrained model and native CPU comparison

Start with one image before increasing the batch. These execute all nine
layers of the original WISE checkpoint, including learned HerPN activations.

```bash
taskset -c "$CPU_CORES" python "$P5/src/evaluate.py" \
  --backend tpu --wise-root "$WISE_ROOT" --pretrained-batch 1 \
  --convolution tiled-toeplitz --rotation-cache-mib 512 --cache-mib 1024 \
  --warmups 1 --repeats 3 --output "$CROSS_ROOT/p5-results/tpu-mnist1"
taskset -c "$CPU_CORES" python "$P5/src/evaluate.py" \
  --backend tpu --wise-root "$WISE_ROOT" --pretrained-batch 8 \
  --convolution hybrid --linear-chunk 8 --rotation-cache-mib 512 --cache-mib 1024 \
  --warmups 1 --repeats 3 --output "$CROSS_ROOT/p5-results/tpu-mnist8"
```

Build the unchanged native WISE extension using the pinned OpenFHE instructions
in [README.md](README.md). Run `native_evaluate.py` with a fresh output directory,
`--module` pointing to that extension, `--degrees 32768 --threads 1 4 8
--scaling flexible --pretrained-batch 8 --convolution hybrid`. Compare the
fastest passing warm native report to the same TPU batch:

```bash
python "$P5/src/score.py" --native-only --pretrained \
  --native-cpu /absolute/path/to/selected-native-report.json \
  --tpu "$CROSS_ROOT/p5-results/tpu-mnist8/testmnist8.json"
```

Report the selected CPU model, affinity, thread count, native library hashes,
TPU type and device count, exact source versions, cache/fusion/layout controls,
quality metrics, setup/warmup time and raw server/client timings. An eight-image
check is not accuracy over the complete 10,000-image encrypted test set. CPU
and TPU results on different hosts must disclose both host configurations.
