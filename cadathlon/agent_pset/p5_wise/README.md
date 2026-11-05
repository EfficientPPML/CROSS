# P5 · WISE scheduling for CROSS

Generate packed neural-network schedules using the pinned
[JianmingTONG/WISE](https://github.com/JianmingTONG/WISE) planner and execute
them using CROSS CKKS on CPU and TPU. Optimize the TPU schedule while preserving
the network, inputs, security and numerical gates.

Read [the problem](doc/problem.md). The starting implementation includes actual
WISE fully-connected, convolution and pooling packers, a BSGS executor, a direct
diagonal control, independent PyTorch references and a matched benchmark scorer.
All six public networks include a nonlinear encrypted square.

For a fresh machine, follow [TPU_TESTING.md](TPU_TESTING.md), including the
explicit CPU/TPU checks and native CPU comparison. The branch contains all
CROSS runtime dependencies; the separate WISE checkout stays pinned.

```sh
export WISE_ROOT=/path/to/WISE
# If no checkout exists, make -C src install-wise WISE_ROOT=/fresh/path/WISE
# In your own environment (Python 3.13 was used for development):
python -m pip install numpy scipy torch einops matplotlib tqdm
make -C src reference-test
make -C src cpu-test
make -C src tpu-test
# Whole six-case collection, with exact source snapshots and process timing:
make -C src cpu-evaluate
make -C src tpu-evaluate
```

For CNNs, also compare `--convolution tiled-toeplitz` and `--convolution winograd`.
The full pretrained network is available through `evaluate.py --pretrained-batch
1 --convolution tiled-toeplitz`; it retains every learned HerPN activation and
all nine layers. Its `hybrid` option uses a Toeplitz first convolution followed
by native Winograd. `pretrained_reference.py --full-cleartext-test-set` checks
the original checkpoint on all 10,000 images and verifies selected WISE layouts.
These plaintext checks do not measure encrypted accuracy.

`runner.py` and `evaluate.py` accept `--linear-chunk 8` to fuse up to eight
public diagonal products per executable, using the existing CROSS modular
kernels. The default `0` retains separate primitives. `--cache-mib` bounds the
public plaintext cache (default256MiB). Record both controls when comparing
timings; neither changes the WISE schedule or encryption profile.
`--rotation-cache-mib 256` optionally retains public rotation keys and transform
controls across calls. The default0 uses the original low-memory helper. The
budget covers retained cache entries; it is not a whole-process memory limit.

## Native CPU reference

Build the unchanged WISE C++ backend against OpenFHE v1.4.2.6, commit
`94289854bb024bcef7e6b5c9a59eda5e9e204770`. The following uses separate build
and installation directories; Python must have `pybind11` installed.

```sh
git clone --recursive --branch v1.4.2.6 https://github.com/openfheorg/openfhe-development.git openfhe-wise
git -C openfhe-wise rev-parse HEAD  # must equal the pinned commit above
cmake -S openfhe-wise -B openfhe-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD/openfhe-install" -DWITH_OPENMP=ON -DBUILD_UNITTESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF
cmake --build openfhe-build -j 8
cmake --install openfhe-build
cmake -S "$WISE_ROOT/backend/openfhe/bindings/python" -B wise-bindings -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE="$(command -v python)" -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" -DOpenFHE_DIR="$PWD/openfhe-install/lib/OpenFHE"
cmake --build wise-bindings -j 8
export LD_LIBRARY_PATH="$PWD/openfhe-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# From P5 src/, supplying the absolute resulting extension path:
python native_evaluate.py --module /path/to/wise-bindings/fhecore.cpython-313-x86_64-linux-gnu.so --degrees 16384 --threads 1 4 --scaling flexible --convolution tiled-toeplitz --output run/native-cpu
```

Use the extension filename produced for your Python version. The collector
preserves source snapshots, every attempted configuration, native binary
hashes and failures. Its `--pretrained-batch 1 --degrees 32768` path measures
the full checkpoint with a 60-bit first modulus. `--diagnose --warmups 0
--repeats 1` decrypts between layers on the client for troubleshooting; these
records are explicitly excluded from performance selection.
The native collector accepts `--first-bits` for explicit modulus-headroom
experiments. The default remains60 for pretrained runs; security and accuracy
gates apply to every precision/headroom attempt.

After collecting matched CROSS CPU/TPU and native CPU reports, use
`score.py --pretrained --cpu CPU.json --tpu TPU.json --native-cpu NATIVE.json`
for a separate trained-model comparison. Multiple distinct batches may be
listed in matching order. This checks the complete-checkpoint/sample evidence
and trained-model error gates, and awards no synthetic-suite points.
To compare tuned native CPU directly with TPU, use
`score.py --native-only --pretrained --native-cpu NATIVE.json --tpu TPU.json`.
This mode requires complete warm repeated results and the same workload,
pinned WISE assets, security and accuracy gates. It needs no CROSS CPU report
and awards no contest points. Omit `--pretrained` to compare all six fixed
synthetic networks; supply six native/TPU reports in matching order.

Measured status lives in [VALIDATION.md](VALIDATION.md). Native Winograd full-CNN
execution and plaintext pretrained layouts have passed. Complete pretrained
encrypted repeated measurements have passed for one image in CROSS and for
one/eight/64 images in native OpenFHE. The complete eight-image CROSS hybrid
network also passed one cold encrypted request. All TPU measurements remain
in progress. No TPU speedup is claimed before the matched measurements complete.
