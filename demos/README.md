# CROSS HE demos — LeNet / LoLA / AlexNetTiny / AlexNetFull

End-to-end CKKS HE inference demos for four registered networks (AlexNetFull
has no TPU inference entrypoint yet) running on TPUv6e-8.
Each demo follows the canonical 4-file structure:

| file | purpose |
|---|---|
| `<model>_he.py` | HE class + cleartext reference + unified Mapping execution |
| `<model>_train.py` | PyTorch training |
| `<model>_he_test.py` | Functional correctness gate |
| `<model>_he_perf_test.py` | Latency / throughput gate |

Shared infrastructure:

| file | purpose |
|---|---|
| `cleartext_ops.py` | Packing-independent NumPy convolution, pooling, dense, bias, and quadratic reference operations |
| `canonical_demo.py` | `CanonicalDemo`: the shared torch-model -> plan -> Mapping path every demo runs |
| `encrypted_demos.py` | The four demos on that path, plus their weight provenance |
| `../jaxite_word/nn.py` | `vectorize`: traces a `torch.nn.Module` into a semantic `VectorizedProgram`. No HE concept appears here |
| `../jaxite_word/packing.py` | `pack`: global layout reconciliation, the PP-op DAG and its constants, and the secure `RingConfig` derived from them |
| `../jaxite_word/mapping.py` | `Mapping`: schedules that DAG, instantiates its `CKKSContext`, and owns device placement, preparation, compilation and execution |
| `../jaxite_word/ckks_ctx.py` | `CKKSContext`: the codec, HE primitives and operators the schedule binds against |

All demo entrypoints encrypt and decrypt through the sole model-serving
facade: `Mapping.encrypt_input`, `Mapping.execute`, and
`Mapping.decrypt_output` (or their `Mapping.infer` composition). Mapping's
private codec implementation delegates to its owned `CKKSContext`; demo files
never call `ctx.encrypt_slots*` or `ctx.decrypt_slots*` directly. The codec
uses the **kernel CSPRNG (`os.urandom` + Box-Muller)** for noise sampling—
production-safe and ~250× faster than the historical
`secrets.SystemRandom().normalvariate` Python loop. Raw ciphertext arrays do
not cross the Mapping boundary; ciphertexts there are rank-5 `Polynomial`s.

Each demo's source of model truth is an ordinary `torch.nn.Module`. No demo
lowers weights by hand: `nn.vectorize` traces the module, `packing.pack`
reconciles the layouts and derives the ring, and `Mapping` schedules the
result. Constructing one `Mapping` instantiates its `CKKSContext`, propagates
levels/scales/noise metadata, creates the compute and virtual-memory
trajectories, chooses a BSGS plan per matvec, aggregates exact rotations,
generates keys and controls, encodes constants, and performs JAX compilation
offline. Serving calls only `mapping.execute(ciphertext)` or `mapping.infer`.

The common model lifecycle is:

```python
from jaxite_word import Mapping, nn, packing

model = QuadLeNet().double().eval()      # a real torch.nn.Module

program = nn.vectorize(model, (1, 28, 28))
packed = packing.pack(program)           # PP-ops + the ring they need
mapping = Mapping(packed, global_batch=1)

scores = mapping.infer(example)
```

The historical `LeNetHE`, `LoLAHE`, `AlexNetTinyHE`, and `AlexNetHE`
constructors are thin adapters over that lifecycle. Their shared
`CanonicalDemoAdapter` implementation owns preparation, logical
encrypt/decrypt, batching, and inference. Model files retain only their weight
argument order, optional client preprocessing, ring assertions, and reporting.
`demo_api_contract_test.py` prevents a model from adding a second pipeline or
calling the context codec directly.

The ring is not supplied: `packing.pack` derives it from this model's widest
live layout and the depth its operations actually emit, and `Mapping` runs on
exactly that, so normal callers pass neither a parameters dict nor a `dnum`.

Both are still accepted in narrow forms. A parameters dict must be wrapped in
`packing.test_only_parameters(...)`, which is how a test says out loud that it
is running on a small insecure ring; a plain dict is refused. A `dnum` is
accepted when it restates the ring's own value and refused when it
contradicts it. The rule behind both: the parameters that execute must be the
ones the security analysis validated.

Planning is separate from materialization on purpose. `demo.build_plan()` is
cheap and is what inspection uses; `demo.materialize_mapping()` is the
expensive half. The cost is not key generation, which is well under a second
even at degree 16384 -- it is constant materialization and per-level operator
controls, which reach tens of gigabytes at demo width. Demos therefore plan
with `packing.PackingPolicy(lazy_constants=True)`; the packer's own default is
eager.
`compute_trajectory` and `memory_trajectory` describe dependencies, levels,
virtual buffers, and live intervals. JAX remains free to optimize physical
instruction and device-buffer scheduling inside each fused region.

`global_batch` is always the public ciphertext batch. Device placement is an
optional Mapping construction choice. If `B` inputs are assigned to `B`
devices, each device receives a local batch of one while the public input and
output remain rank-5 `Polynomial` values with batch `B`.

The demo model constructors retain their existing model-level `batch=B`
argument because it describes how many inputs `model.infer(...)` accepts. When
a model constructs its Mapping, it forwards that value as
`global_batch=self._batch`; `Mapping` itself does not use a `batch=` keyword.

The concrete `B`-inputs-on-`B`-devices Mapping is:

```python
import jax

B = 8
devices = tuple(jax.devices("tpu")[:B])
mapping = Mapping(packed, global_batch=B, devices=devices)

input_ct = mapping.encrypt_input(examples[:B])
output_ct = mapping.execute(input_ct)  # Polynomial(B) -> Polynomial(B)
outputs = mapping.decrypt_output(output_ct)
```

Mapping validates that the global batch is divisible by the device count. It
privately reshapes the payload from global batch `B` to
`(device_count, local_batch, ...)`, invokes its one prepared executable, then
flattens and wraps the result back into a global-batch `Polynomial`. Raw
ciphertext arrays never become a demo or serving API.


### Rings today

The chains quoted in the sections above are historical: they describe the
hand-built graphs, at ring degree 2048, which provided no security margin.
`packing.pack` now derives a 128-bit-secure ring per model from its slot
demand and emitted depth:

| model | depth | num_q | dnum | degree | num_slots |
| --- | --- | --- | --- | --- | --- |
| LeNet | 7 | 8 | 4 | 32768 | 16384 |
| LoLA | 5 | 6 | 3 | 32768 | 16384 |
| AlexNetTiny | 7 | 8 | 4 | 32768 | 16384 |
| AlexNetFull | 15 | 16 | 8 | 65536 | 32768 |

Depth, not slot demand, is what forces each degree: LeNet needs 784 slots of
the 16384 it gets.


---

## Validated single-device performance — TPUv6e-8 host

> **Historical.** The tables in this section and the AlexNet and CPU latency
> tables that follow were recorded on the retired pre-unification path and are
> kept for context. The dated validation sections at the end of this file
> (2026-08-10 and 2026-08-19) and the 3.0.0 release validation supersede them:
> on the current canonical Mapping path at 128-bit security, expect roughly
> 0.4–0.7 s per image (LoLA 566 ms at B=1 and 397 ms/image at B=32 over eight
> chips; LeNet 714 ms at B=1 and 475 ms/image at B=64; AlexNetTiny 662 ms at
> B=1 and 463 ms/image at B=32).

Recorded warm wall median of 5 runs on the pre-unification single-device path.
**All numbers measured under the canonical CSPRNG encrypt path** (the historical
README values quoted slower numbers from the legacy `secrets`-based path
— those are now obsolete).

| network | B=1 | B=8 | B=32 | B=64 |
|---|---|---|---|---|
| LoLA | 263–286 ms | n/a | **43.0 ms / img** | n/a |
| LeNet | 39.7 ms | 38.5 ms | 30.5 ms | **27.55 ms / img** |
| AlexNetTiny | 53.1 ms | 42.4 ms | **40.6 ms / img** | ~38.5 ms |

These are retained single-device baselines, not claims for the newly unified
multi-device Mapping path. Multi-device throughput must be remeasured with a
Mapping built for the exact global batch and device tuple. Measurements from
the retired standalone execution utility are not current performance results.

### Setup costs (one-time)

| network | full build (cold) |
|---|---|
| LoLA (7Q+3P) | ~6 min |
| LeNet (depth-7) | ~16 min |
| AlexNetTiny (depth-7) | **12 min** |

Cold build is the only setup cost today: `materialize_mapping()` generates keys,
encodes constants, and compiles, and every subsequent request reuses that one
Mapping. **Historical only:** earlier revisions also published warm cache-reload
times (~4–6 min) and cache file sizes (~140–200 MB) for a pickled network cache.
That cache is gone — `save_cache`/`from_cache` no longer exist — because the
canonical pipeline re-derives the plan from the torch model cheaply.

### AlexNet variants — performance (CIFAR-10)

Latency columns are the HE numbers from the tables above; depth and params are
the architecture. **CIFAR-10 / MNIST test accuracy for all four models lives in
the single accuracy matrix below** — this table does not duplicate it.

| network | depth | params | validated latency | cold build |
|---|---|---|---|---|
| **AlexNetTiny** | 7 | 2,282 | 40.6 ms/image (single TPU, B=32) | 12 min |
| **AlexNet (full)** | 15 | 35,626 | CPU 5.82 s¹ | ~27 min (CPU build) |

The accuracy matrix below is the consistent 30-epoch figure (all four models ×
both datasets). Neither the datasets nor the weights are committed — fetch and
train them first, else the accuracy gates skip and the encrypted demos raise
`FileNotFoundError`:

```bash
python3 demos/prepare_demo_assets.py            # CIFAR-10 + MNIST, then train all four
python3 demos/prepare_demo_assets.py --data-only        # download only
python3 demos/prepare_demo_assets.py --models lola      # one model
python3 demos/prepare_demo_assets.py --epochs 3         # fast smoke run
```

That script downloads CIFAR-10 to `demos/cifar10_data/` and MNIST IDX files to
`mnist/data/`, then writes AlexNet and LeNet binaries to `demos/maple_data/`
and LoLA's to `$CROSS_DATA_ROOT/pretrained_weights/lola/`. The individual
trainers still work standalone — `python3 alexnet_train.py --epochs 30`,
`--full` for depth-15, `python3 lenet_train.py`, `python3 lola_train.py
--train` — but they assume their dataset is already on disk.

Notes:
- **Both variants classify well above chance**, and on this fixed dataset the
  deeper "full" net beats Tiny (see the matrix) — capacity helps; there is no
  "bigger-is-worse" effect.
- LeNet / LoLA are **MNIST**-first networks but were also trained on CIFAR-10
  here (see the matrix below); they land in the same capacity-limited band as
  the AlexNet variants — confirming the CIFAR/MNIST gap is the dataset, not the
  architecture.
- For **Tiny**, the encrypted forward reproduces its cleartext accuracy within
  the CKKS noise floor (max\|HE − cleartext\| ≈ 6e-3 at depth-7, argmax 96/96).
- The **depth-15 net is LR-sensitive** (7 stacked `x²`): it must be trained in a
  fresh process at `lr=1e-3` or it can collapse toward chance. A learnable
  per-channel polynomial activation `a·x² + b·x + c` (same HE multiplicative
  depth) lifts it to **~59%** and trains robustly — not yet wired into the HE
  path.
- Both variants run on a **client-side 16×16 downsample** of the 32×32 image,
  not full resolution. True full-res detection would need a multi-ciphertext
  tiling engine + ring degree ≥ 16384 — a separate, unimplemented effort.
- Accuracy is gated by `alexnet_he_test.py`
  (`test_{tiny,full}_cleartext_accuracy_beats_chance`); the other gates only
  check HE ≡ cleartext, which a chance-level model would pass.

¹ The depth-15 `AlexNetHE` encrypted path runs end-to-end on CPU: the decrypted
argmax matches the cleartext reference with max\|HE − cleartext\| ≈ 1.6e-2. It
is gated by `alexnet_he_test.py::test_full_cleartext_accuracy_beats_chance` and
`::test_full_cleartext_random_inputs_runs`; it is not benchmarked on TPU. The
17Q+4P chain quoted for that CPU run was the historical hand-built one; the
canonical path now derives its ring from the model (see "Rings today").

### Cleartext accuracy — all four models × both datasets

Every model trained on **both** CIFAR-10 and MNIST under one consistent recipe
(30 epochs, Adam `lr=1e-3`/`wd=1e-4`, `batch=128`, `seed=42`, per-channel
standardization, full 10k test sets). Each dataset is adapted to the model's
native input shape, architecture unchanged: LeNet/LoLA take **1×28×28** (CIFAR
→ grayscale + resize 32→28), AlexNet takes **3×32×32** (MNIST → pad 28→32 +
1→3-channel replicate). Random-guess = 10%.

| model | CIFAR-10 | MNIST |
|---|---|---|
| **LoLA** | 49.6% | 97.8% |
| **LeNet** | 49.2% | 98.5% |
| **AlexNetTiny** | 50.2% | 96.9% |
| **AlexNetFull** | 52.4% | 98.6% |

Same story for every model: **~97–98.6% on MNIST vs ~49–52% on CIFAR-10.** The
~48-point gap is the **dataset difficulty** (MNIST is near-saturated for any of
these small nets), not the architecture — the *same* network scores in both
bands depending only on the task. On the harder CIFAR-10 all four sit in a
tight, capacity-limited ~49–52% band, with the deeper AlexNetFull best.

### Reaching CIFAR-10 SotA (>90%): ResNet-20

That ~49–52% is the accuracy **cost of the HE shortcuts** these demos take, **not**
an architecture ceiling. The FHE-DL state of the art, **ORION**
([arXiv:2311.03470](https://arxiv.org/abs/2311.03470),
[github.com/baahl-nyu/orion](https://github.com/baahl-nyu/orion)), reaches >90%
by running a *real* network. Reproducing its model + recipe here
(`resnet20_cifar.py`) gives **92.77%** (ReLU) / **92.26%** (SiLU) — a **~+40
point** jump — with a normally-trained **ResNet-20**.

**Where the gap comes from.** Factors 1–3 are structural (the demos shrink
channels 8–12× and downsample so every activation fits `NUM_SLOTS=1024` on a
degree-2048 chain); 4–5 are free wins the demos simply weren't using.

| # | factor | demo nets (LoLA/LeNet/AlexNet) | ORION ResNet-20 | impact |
|---|---|---|---|---|
| 1 | **architecture** | x²-nets, 2.3K–36K params, no residuals | ResNet-20, **272K params**, residual + BN | ★★★ |
| 2 | **input resolution** | client-side avg-pool **32→16** (¾ of pixels dropped) | full **32×32** | ★★★ |
| 3 | **activation** | raw **`x²`** (even, unbounded, low expressivity) | **ReLU** | ★★★ |
| 4 | **data augmentation** | none | RandomCrop(32,pad=4) + HorizontalFlip | ★★ |
| 5 | **training** | Adam, 20–30 epochs | **SGD 0.1**/mom 0.9/wd 5e-4, cosine, **200 epochs** | ★★ |

| model | activation | params | CIFAR-10 test acc |
|---|---|---|---|
| **ResNet-20** (`resnet20_cifar.py`) | ReLU | 272K | **92.77%** |
| ResNet-20 | SiLU | 272K | 92.26% |
| best demo net (AlexNetFull) | x² | 36K | see accuracy matrix above |

**Key insight — ORION's activation.** `on.ReLU()` *trains as exact ReLU* (its
cleartext composite-sign returns the true step, so `x·step(x) = ReLU(x)`); the
polynomial (a composite minimax-sign of degrees `[15,15,27]`, precision 128)
appears only at HE inference. So the accuracy is just a normally-trained
ResNet-20's — the HE machinery preserves it, it does not create it. SiLU is
included because it is smooth with a cheap low-degree polynomial approximation
(the more HE-friendly choice, and nearly as accurate).

**Reproduce:** `python3 resnet20_cifar.py --act relu` (or `--act silu`) — ORION's
exact recipe (SGD lr=0.1/mom 0.9/wd 5e-4, CosineAnnealingLR, 200 epochs,
RandomCrop(32,pad=4)+HorizontalFlip). ~6 min on a modern GPU; a 200-epoch CPU
run takes days.

**Running it *encrypted* (the honest caveat).** Higher accuracy ≠ runnable on
CROSS's current HE stack. ORION runs ResNet-20 encrypted with ReLU as a
high-degree composite minimax-sign polynomial (depth ≈ 13+ multiplicative levels
*per activation* vs 1 for `x²`), **bootstrapping** between blocks, a **degree-2^16**
ring (`LogN=16`, `LogScale=40`), and multi-ciphertext packing for the
full-resolution feature maps. CROSS's demos run on degree-2048 / `max_level≤16`
with **no bootstrapping in the inference path**, so they cannot execute
ResNet-20 as-is. The deepest net CROSS runs encrypted is the depth-15
`AlexNetHE`. Closing the accuracy gap *under encryption* is thus an
infrastructure problem (bigger secure ring + bootstrapping + tiled conv +
poly-ReLU), separate from the cleartext accuracy result above.

### HE inference latency on CPU (no TPU, jax cpu backend)

Encrypted end-to-end inference, single image, warm median (JIT compile excluded),
measured on this host with `JAX_PLATFORMS=cpu` (32-core CPU, no TPU). Build =
one-time key-gen + BSGS precompute. Correctness = decrypted HE argmax vs the
cleartext reference.

| network | depth | build (keygen + precompute) | warm latency / img | HE vs cleartext |
|---|---|---|---|---|
| **LoLA** | 5 | 195 s + 83 s | **0.80 s** | ✅ argmax, max err 9.7e-3 |
| **AlexNetTiny** | 7 | 113 s + 229 s | **1.51 s** | ✅ argmax, max err 2.6e-3 |
| **LeNet** | 7 | 114 s + 334 s | **1.83 s** | ✅ argmax, max err 5.6e-2 |
| **AlexNetFull**¹ | 15 | 265 s + 1373 s | **5.82 s** | ✅ argmax, max err 1.6e-2 |

CPU HE latency is ~0.8–5.8 s/image for these recorded single-image runs. Both
depth and width push latency up (depth-5 LoLA 0.8 s; depth-7 nets ~1.5–1.8 s;
depth-15 Full 5.8 s). These CPU results are independent of the unified
multi-device Mapping path and should not be used to infer its throughput. The
depth-15 Full path runs correctly after the fixes noted in ¹.

---

## File map

```
demos/
├── lenet_he.py              # LeNetHE class + cleartext + Mapping execution
├── lenet_he_test.py         # Functional gate
├── lenet_he_perf_test.py    # Perf gate, CLI, and mocked setup contracts
├── lenet_train.py           # PyTorch MNIST training
│
├── alexnet_he.py            # AlexNetTinyHE/AlexNetHE + cleartext
├── alexnet_he_test.py       # Cleartext + HE correctness gates
├── alexnet_he_perf_test.py  # Perf gates and mocked setup contracts
├── alexnet_train.py         # PyTorch CIFAR-10 training (BN-folded export)
│
├── lola_he.py               # LoLAHE class + cleartext + Mapping execution
├── lola_he_test.py          # Functional gate
├── lola_he_perf_test.py     # Perf gate
├── lola_train.py            # PyTorch MNIST training + test-slice export
│
├── resnet20_cifar.py        # Standard CIFAR-10 accuracy reference model
├── prepare_demo_assets.py   # Download data + train/export every demo model
├── model_tools_test.py      # Standalone trainer and asset integrity gates
│
├── canonical_demo.py        # Shared plan, Mapping, and serving lifecycle
├── encrypted_demos.py       # Registered torch models and weight provenance
├── demo_api_contract_test.py # Cross-demo API ownership gates
├── demo_test_utils.py       # Shared lightweight demo test doubles
├── log/                     # Perf CSVs and demo outputs (gitignored)
└── maple_data/              # Trained weight binaries + dataset slices
```

---

## How to run the demos

All commands are run from `demos/`.

### Common quick check (any model)

To confirm the models and canonical deployment classes import:

```bash
cd demos
python3 -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, '../jaxite_word')
import lenet_he, lola_he, alexnet_he
from jaxite_word import Mapping, nn, packing
import jax
print(f'JAX devices: {jax.device_count()}')
print(f'Mapping.execute available: {hasattr(Mapping, \"execute\")}')
print(f'packing.pack available: {hasattr(packing, \"pack\")}')
"
```

The device count depends on the host. TPUv6e-8 should report eight visible
devices; `Mapping.execute available` should be `True` on every backend.

### LeNet on MNIST

```bash
# 1. Train (CPU ~5 min). Outputs to demos/maple_data/lenet_*.bin .
python3 lenet_train.py

# 2. Validated latency. `build` materializes a Mapping for exactly --batch
#    images and benchmarks it end to end (~16 min each on TPUv6e):
python3 lenet_he_perf_test.py build --batch 1     # single-image latency
python3 lenet_he_perf_test.py build --batch 64    # throughput mode

# 3. Functional gate — argmax matches cleartext, decryption noise floor.
python3 lenet_he_test.py
LENET_REGRESSION=1 python3 lenet_he_perf_test.py LeNetRegressionTest

# 4. Predictions, one image per call. `forward` takes no --batch: a Mapping
#    serves one exact global batch, and this command's is 1.
python3 lenet_he_perf_test.py forward --n 5
```

> **On the commands above.** No HE cache is involved: each subcommand
> materializes the Mapping from the torch model and then measures it. The
> cache existed to avoid rebuilding a hand-lowered graph and its keys; the
> canonical pipeline re-derives the plan cheaply, so `save_cache`/`from_cache`
> are gone. `dnum` is likewise not a caller concern -- `packing.pack` derives
> it, and Mapping refuses a value that contradicts the plan. Layout is chosen
> by the packer's templates, not by a `--mux` flag. `LeNetRegressionTest`
> builds one Mapping per batch (B=1, B=32, B=64) and keeps only one resident
> at a time, releasing the previous one through `model.release()` before the
> next build, because two resident Mappings plus a build exceed one chip's
> 32 GiB HBM. It skips unless `LENET_REGRESSION=1` is set.

### AlexNetTiny on CIFAR-10

```bash
# 1. Train AlexNetTiny variant (CPU ~20-40 min, GPU ~5 min):
python3 alexnet_train.py --epochs 30

# 2. Bind weights and build the Mapping:
python3 -c "
import sys
sys.path.insert(0, '.'); sys.path.insert(0, '../jaxite_word')
from alexnet_he import (AlexNetTinyHE, load_trained_alexnet_tiny_weights,
                          prepare_tiny_args)
m = AlexNetTinyHE(batch=1)
m.precompute_plaintexts(*prepare_tiny_args(load_trained_alexnet_tiny_weights()))
"

# 3. Functional gate:
python3 alexnet_he_test.py AlexNetTinyCleartextTest AlexNetTinyAccuracyTest

# 4. Current validated performance gates. Each builds its own Mapping
#    (minutes, gigabytes), so they skip unless ALEXNET_PERF is set:
ALEXNET_PERF=1 python3 alexnet_he_perf_test.py AlexNetTinyHEPerfTest
```

### LoLA on MNIST

```bash
# 1. Provide trained weights under $CROSS_DATA_ROOT/pretrained_weights/lola/.
#    If they are absent, lola_he.py creates deterministic fallback data.

# 2. Bind weights and build the Mapping:
python3 -c "
import sys
sys.path.insert(0, '.'); sys.path.insert(0, '../jaxite_word')
from lola_he import LoLAHE, prepare_weights, load_or_generate_data
m = LoLAHE(batch=1)
data, _ = load_or_generate_data()
m.precompute_plaintexts(*prepare_weights(data))
"

# 3. Functional + single-chip perf gates. The perf gate builds a B=1 and a
#    B=32 Mapping, so it skips unless LOLA_PERF is set:
python3 lola_he_test.py
LOLA_PERF=1 python3 lola_he_perf_test.py
```

For device-parallel execution, build the model's Mapping with the desired
global batch and exact device tuple as shown in the lifecycle example above.
The packed plan is placement-independent, but a Mapping is not: it compiles
for one exact device tuple and one global batch. Pass `batch=B,
devices=devices` to the model constructor and call
`precompute_plaintexts(...)` to build one for that placement.

---

## Architecture details

### LeNet (depth-7, MNIST)

```
Input:  1×28×28 = 784 slots, packed into NUM_SLOTS=1024
Conv1(1→4, 5×5/s=2/p=2)        →  4×14×14 = 784  →  L8 → L7
Quad                                              →  L7 → L6
Conv2(4→8, 5×5/s=2/p=2)        →  8×7×7 = 392    →  L6 → L5
Quad                                              →  L5 → L4
FC1(392→32, BSGS)              →  32              →  L4 → L3
Quad                                              →  L3 → L2
FC2(32→10, BSGS)               →  10              →  L2 → L1
```

* CKKS (historical, hand-built chain): 9Q+4P modulus pool, max_level=8,
  dnum=4, ring degree 2048
* Optimizations (historical, same hand-built chain): state cache,
  JIT-fused bias-add, multiplexed conv1 lowering, full-pipeline JIT
  fusion, batched B=8/B=32/B=64

### LoLA (depth-5, MNIST)

```
Input:  multiplexed-stride-2 packing of (1, 28, 28)  →  784 slots
Conv1(1→5, 2×2/s=2)            →  5×14×14 = 980  →  L6 → L5
Quad                                              →  L5 → L4
FC1(980→100, BSGS Toeplitz)    →  100             →  L4 → L3
Quad                                              →  L3 → L2
FC2(100→10, BSGS)              →  10              →  L2 → L1
```

* CKKS (historical, hand-built chain): 7Q+3P modulus pool, max_level=6,
  dnum=3, ring degree 2048

### AlexNetTiny (depth-7, CIFAR-10)

```
Input:  CIFAR-10 (3, 32, 32) → client-side avg-pool 2×2 → (3, 16, 16) = 768 slots
Conv1(3→4, 3×3/s=1/p=1) + AvgPool 2×2 →  4× 8× 8 = 256  →  L8 → L7
Quad                                                    →  L7 → L6
Conv2(4→8, 3×3/s=1/p=1) + AvgPool 2×2 →  8× 4× 4 = 128  →  L6 → L5
Quad                                                    →  L5 → L4
Conv3(8→16, 3×3/s=1/p=1) + AdaptivePool 2×2 → 16× 2× 2 = 64 →  L4 → L3
Quad                                                    →  L3 → L2
FC(64→10)                                  → 10         →  L2 → L1
```

* CKKS: same 9Q+4P pool as LeNet
* **Multiplexed conv1**: stride-multiplexed `(rs, ci, h_stride, w_stride)`
  packing reduces conv1 active-diagonal count from 897 → 81 (11.1×)

### AlexNet (full, depth-15, CIFAR-10)

```
Input:  CIFAR-10 (3, 32, 32) → client-side avg-pool 2×2 → (3, 16, 16) = 768 slots
Conv1(3→8,   3×3/s=1/p=1) + AvgPool 2×2     →  8× 8× 8 = 512  →  L16 → L15
Quad                                                          →  L15 → L14
Conv2(8→16,  3×3/s=1/p=1) + AvgPool 2×2     → 16× 4× 4 = 256  →  L14 → L13
Quad                                                          →  L13 → L12
Conv3(16→32, 3×3/s=1/p=1)                   → 32× 4× 4 = 512  →  L12 → L11
Quad                                                          →  L11 → L10
Conv4(32→32, 3×3/s=1/p=1)                   → 32× 4× 4 = 512  →  L10 → L9
Quad                                                          →  L9  → L8
Conv5(32→32, 3×3/s=1/p=1) + AdaptivePool 2×2 → 32× 2× 2 = 128 →  L8  → L7
Quad                                                          →  L7  → L6
FC1(128→64)                                  → 64             →  L6  → L5
Quad                                                          →  L5  → L4
FC2(64→32)                                   → 32             →  L4  → L3
Quad                                                          →  L3  → L2
FC3(32→10)                                   → 10             →  L2  → L1
```

* CKKS: **17Q+4P** modulus pool (`DEEP_Q_TOWERS_POOL` /
  `DEEP_P_TOWERS_POOL`), max_level=16, ring degree=2048, **dnum=5**. The
  depth-15 schedule consumes 15 logical levels, leaving two Q towers so the
  logits land at level 1 rather than level 0. dnum=5 (not 4) keeps each
  key-switch group ≤ the 4 P-towers. The pool extends LoLA's 9 NTT-friendly
  primes with 8 more consecutive 30-bit primes ≡ 1 mod 4096.
* Mirrors the OpenFHE reference architecture
  (`openfhe_ref_code/models/alexnet.py` — channels [64, 192, 384, 256, 256])
  with channels SCALED DOWN ([8, 16, 32, 32, 32]) so all activations
  fit in NUM_SLOTS=1024 throughout.
* **Security note**: at degree=2048 with 17Q+4P (~634 bits log Q·P),
  this is a research-grade demo only — for IND-CPA-secure depth-15
  inference, ring degree ≥ 16384 is required. Same regime as
  LeNet/AlexNetTiny/LoLA at degree 2048.
* Trained via `python3 alexnet_train.py --full`. HE class: **`AlexNetHE`**.
  Cleartext: `alexnet_full_cleartext`. Device placement follows the same
  Mapping construction contract as every other network.
* Cache build cost: ~30–45 min projected (vs ~12 min for AlexNetTiny) —
  deeper modulus chain plus 8 BSGS layers vs 4 scale build wall ~3×.

---

## CSPRNG noise (production-safe)

The private encryption kernel behind `ctx.encrypt` and `ctx.encrypt_slots`
samples encryption noise via the kernel CSPRNG (`os.urandom`):

```python
def _csprng_uniform_open(n):
    """n float64 uniform on (0, 1] from one os.urandom syscall."""
    raw = os.urandom(8 * n)
    u64 = np.frombuffer(raw, dtype=np.uint64)
    return (u64 >> 11).astype(np.float64) * 2**-53 + 2**-53

def _csprng_gaussian(n, sigma):
    """n samples ~ N(0, sigma²) via vectorized Box-Muller."""
    u = _csprng_uniform_open(2 * n_pairs).reshape(n_pairs, 2)
    radius = np.sqrt(-2.0 * np.log(u[:, 0]))
    angle = 2.0 * np.pi * u[:, 1]
    z = np.empty(2 * n_pairs)
    z[0::2] = radius * np.cos(angle)
    z[1::2] = radius * np.sin(angle)
    return z[:n] * sigma
```

The ephemeral encryption mask is sampled separately by `_csprng_ternary`
using 2-bit rejection sampling, uniformly mapping accepted values onto
`{-1, 0, 1}`.

Distribution sanity (validated against 100k samples, σ=3.19):

| gate | observed | tolerance |
|---|---|---|
| Gaussian mean | +0.016 | \|Δ\| < 0.05 ✅ |
| Gaussian std | 3.188 | \|Δ\| < 0.05 ✅ |

End-to-end correctness across 96 distinct CIFAR-10 images on AlexNetTiny:

| gate | result |
|---|---|
| argmax HE vs cleartext | 96/96 ✅ |
| max\|HE − cleartext\| logits (depth-7) | 1.77e-09 (CKKS noise floor) |
| decryptability max\|slots − decrypt(encrypt(slots))\| | 4.04e-13 |

---

## Unified device-parallel execution

There is no model-specific or standalone multi-device execution layer. A
single Mapping owns both the low-level HE trajectory and its device placement.
The same public call is used for one device and many devices:

```python
output_ct = mapping.execute(input_ct)
```

For global batch `B` on `D` devices, Mapping requires `B % D == 0` and assigns
`B / D` ciphertexts to each device. When callers omit `devices`,
whole-program bootstrap-free mappings choose the widest visible device prefix
that divides `B`; B=32 and B=64 therefore use all eight TPUv6e devices with
local batches four and eight. The leading device axis and any payload
reshaping are private implementation details. Encryption produces one global-batch
`Polynomial`; execution consumes and returns that `Polynomial`; decryption
consumes the returned `Polynomial`.

Device topology and static payload shapes are compilation inputs. Construct a
new Mapping when either changes, and keep construction out of the request path.
Bootstrap mappings retain their explicit temporal barriers and currently
require one device; all network demos in this directory are non-bootstrap.

---

## Key takeaways

1. **Packing and Mapping are the only deployment phases.** Device count is a
   Mapping construction parameter, not a reason to create another inference
   implementation.
2. **The ciphertext boundary remains `Polynomial`.** Global batches are never
   exposed as raw per-device payloads.
3. **The binding stage migrates with each optimization.** Profile each
   step rather than guessing — the actual bottleneck (Python `secrets`
   loops in encrypt) wasn't where the obvious suspect (NTT) lived.
4. **Multi-device performance needs a fresh baseline.** Results from the
   retired standalone pipeline are not supported by the unified Mapping path.

The retained tuning conclusions are intentionally small: decomposition is
chain-specific (`dnum=4` for LeNet's 9Q+4P chain and `dnum=3` for LoLA's
7Q+3P chain), while batching, fused Mapping execution, and multiplexed input
packing are the supported throughput and setup-time optimizations. Raw
experiment chronology and measurements for removed APIs are omitted.

---

## Latest TPUv6e-8 validation and comparison with CROSS v2.0.0 (2026-08-19)

This is the current result set for the canonical Mapping path. It supersedes
the 2026-08-10 model status below. Kernel tables from the older run remain
useful historical context, but the numbers in this section are the latest
measurements on this host.

| item | value |
|---|---|
| accelerator | 8 × `TPU v6 lite` (TPUv6e-8), 128 MiB VMEM/device |
| topology | 2 × 4 local devices |
| Python / JAX | 3.11.13 / 0.10.2 |
| current tree | this repository (the 3.0.0 release tree), 2026-08-19 |
| comparison tree | CROSS v2.0.0, public `main` commit `69c46d2` |
| model timing | warm end-to-end wall: host pack/encrypt + Mapping + decrypt/unpack |
| kernel timing | warm Xprof device time; B=8 sharded values are total batch time |

### Memory-safe automatic placement

An omitted `devices` argument no longer means device 0 for every global
batch. For whole-program bootstrap-free mappings, Mapping chooses the widest
prefix of visible devices that divides the batch. Thus B=32 becomes 8 × B=4
and B=64 becomes 8 × B=8 on this host. Explicit device tuples are unchanged;
prime batches with no wider equal partition fall back to one device.

After AOT compilation, Mapping also releases the source bindings and evaluator
caches captured by the executable. Without that release, LeNet B=64 compiled
an 11.20 GiB constant graph but failed on its first request while trying to
load an 11.34 GiB program beside the duplicate source buffers. The final run
reported `BINDINGS 0` before first execution and completed normally.

The device's 128 MiB figure is on-chip VMEM, not its HBM capacity. The observed
`RESOURCE_EXHAUSTED ... HBM0` failures were HBM residency failures. At maximum
input level, automatic sharding makes one local ciphertext payload 6 MiB for
LoLA B=32, 16 MiB for LeNet B=64, and 8 MiB for AlexNetTiny B=32; XLA tiles the
operators through VMEM. Releasing the duplicate captured source graph is what
fixes the separate executable/HBM collision.

The focused placement/model suites pass: 439 tests, 1 environment-gated skip,
and 39 parameterized subtests. Real encrypted acceptance runs completed for
all three models with performance entrypoints.

### Secure-model performance

| model / mode | placement | cold setup | warm batch wall | wall/image | numeric gate |
|---|---:|---:|---:|---:|---|
| LoLA B=1 | 1 × B=1 | 661.8 s | 540.8 ms median (5) | 540.8 ms | 5/5 argmax match |
| **LoLA B=32** | **8 × B=4** | **769.3 s** | **13,026.5 ms** | **407.1 ms** | 32/32 match; max error 2.83e-11 |
| LeNet B=1 | 1 × B=1 | 930.2 s | 692.7 ms median (5) | 692.7 ms | trained weights; deterministic zero input because MNIST binaries are absent |
| **LeNet B=64** | **8 × B=8** | **1,084.8 s** | **30,854.7 ms median (3)** | **482.1 ms** | 64/64 cleartext argmax match; max error 4.15e-12 |
| AlexNetTiny B=1 | 1 × B=1 | 693.6 s | 640.8 ms median (5) | 640.8 ms | deterministic weights/input |
| AlexNetTiny B=8 | 8 × B=1 | 707.2 s | 3,670.3 ms median (5) | 458.8 ms | 8/8 argmax match |
| **AlexNetTiny B=32** | **8 × B=4** | **820.8 s** | **15,216.7 ms median (3)** | **475.5 ms** | 32/32 match; max error 7.32e-13 |

The B=32/B=64 cases previously failed during device-0 allocation. They now
materialize, load, and execute across all eight devices without OOM. LeNet's
B=64 run is the strongest local-batch memory check: each device owns eight
ciphertexts, versus four for the B=32 modes.

`AlexNetFull` has no performance entrypoint or shipped checkpoint. Its
degree-65536, depth-15 B=1 plan still exceeds one device during pre-AOT
materialization; data-parallel batch placement cannot divide a single sample.
It therefore has no valid latency row. Supporting that plan requires
model-parallel or multi-device streaming MatVec, not another batch placement
heuristic.

### HE operators and arithmetic kernels

Each HE-operator cell is Barrett / Montgomery in microseconds. Montgomery is
faster for every common operator here. B=8 is the total latency of eight
ciphertexts sharded over all eight devices.

| operator (degree 65536, 51 limbs) | B=1 | B=8 sharded |
|---|---:|---:|
| HEAdd | 69.36 / **37.57** | — |
| HESub | 58.61 / **41.86** | — |
| PtCt VPU | 83.44 / **56.79** | — |
| PtCt BAT | 134151.30 / 134104.60 | — |
| Rescale | 604.53 / **438.78** | 709.97 / **529.13** |
| HEMul, no relinearization | 162.11 / **90.78** | — |
| full HEMul | 3875.11 / **3182.72** | 3960.73 / **3322.09** |
| HERot | 2331.05 / **2257.83** | 3050.91 / **2522.05** |

Other current kernel results:

| kernel | current result |
|---|---|
| modular multiply, 51 moduli, B=1/B=2 | Montgomery **74.65/144.91 µs**; Barrett 86.80/168.22; Shoup 193.09/720.71; BAT-lazy 493.47/1435.36 |
| BSGS matvec, degree 4096 | n=64: 1.412/1.238/1.218 ms sync/async/device; n=256: 3.327/3.164/3.143 ms |
| bootstrap PtCt, B=8 sharded, degree 256 | 40/24/12 towers: 0.335/0.313/0.310 ms, or 23,877/25,576/25,789 ct/s |
| bootstrap PtCt, B=8 sharded, degree 4096 | 40/24/12 towers: 0.371/0.342/0.345 ms, or 21,540/23,405/23,202 ct/s |
| Table V Pallas BAT | all eight matrix shapes pass; see `jaxite_word/bat_performance_result.csv` |
| Table IX | all 162 cases pass; exact baseline comparison below |
| PyTorch BSGS convolution | correct; 5.293 s versus 16.9 ms native Conv2d |

PtCt BAT remains an anomaly at about 134 ms, roughly 2,361× slower than the
56.79 µs Montgomery VPU path. The bootstrap performance row is the dominant
sharded linear-transform PtCt kernel, not full `EvalBootstrap` latency.

### NTT and basis conversion

NTT entries are single-device B=8 → eight-device-sharded B=8 total latency,
in microseconds.

| degree / limbs | Montgomery | Barrett | Shoup | BAT-lazy |
|---|---:|---:|---:|---:|
| 4096 / 4 | 6.55 → 69.75 | 9.50 → 174.01 | 15.25 → **13.91** | 16.07 → 71.33 |
| 8192 / 8 | 20.81 → 118.89 | 32.17 → 135.84 | 51.46 → 118.91 | 57.45 → 159.50 |
| 16384 / 16 | 107.42 → 90.31 | 151.75 → **55.30** | 335.77 → 134.05 | 248.20 → 182.06 |
| 65536 / 48 | 1872.10 → **300.18** | 2227.23 → 339.81 | 7483.30 → 387.41 | 3453.71 → 581.06 |

Basis-conversion cells are default / Montgomery / Montgomery-BAT / BAT,
again in microseconds for the complete batch.

| limbs | B=1 | B=256 |
|---|---:|---:|
| 12→28 | 805.93 / 1024.06 / 245.91 / **130.01** | 30855.68 / 39239.52 / 37975.78 / **30275.83** |
| 12→36 | 1044.59 / 1307.85 / 262.77 / **141.15** | **32097.28** / 41765.39 / 43193.40 / 32721.08 |
| 16→40 | 163.95 / 203.34 / 113.01 / **67.96** | 42598.24 / 54855.58 / 42946.67 / **31372.30** |
| 20→48 | 363.96 / 497.52 / 388.15 / **196.93** | 68207.43 / 81840.14 / 56746.84 / **43733.74** |
| 24→56 | 317.33 / 372.45 / 160.90 / **97.29** | 84289.38 / 100455.93 / 61902.06 / **46408.64** |

### Direct comparison with CROSS v2.0.0

The comparison checkout is not a second copy of the current demo suite. It
contains LoLA only, and its inference benchmark requires pickled B=1/B=32
caches that are absent. Its source records historical warm values of about
286 ms at B=1 and 43 ms/image at B=32. Against those non-reproducible legacy
claims, the canonical secure path is about 1.89× slower at B=1 and 9.47×
slower per image at B=32. LeNet and both AlexNet variants have no v2.0.0
implementation, so no honest model speedup can be calculated for them.

For kernels, eleven exact Table IX endpoints were run in both trees on this
same TPU. Values are µs/item; delta is `(current / v2.0.0) - 1`, so negative
is faster. The geometric-mean delta is **+0.07%**, effectively parity.

| exact Table IX case | v2.0.0 | current | delta |
|---|---:|---:|---:|
| automorphism, 19 limbs | 18.572 | 18.583 | +0.06% |
| automorphism, 53 limbs | 20.356 | 20.314 | -0.21% |
| BAT BConv, 7→20 | 4.777 | 4.775 | -0.05% |
| BAT BConv, 15→38 | 8.449 | 8.432 | -0.20% |
| Montgomery NTT, 1 limb | 0.627 | 0.609 | -2.87% |
| Montgomery NTT, 7 limbs | 3.090 | 3.124 | +1.11% |
| Montgomery NTT, 40 limbs | 16.249 | 16.265 | +0.10% |
| VecAdd, 18 limbs | 1.388 | 1.392 | +0.35% |
| VecAdd, 53 limbs | 3.655 | 3.597 | -1.57% |
| `VecMul` row, 18 limbs | 2.015 | 2.059 | +2.16% |
| `VecMul` row, 53 limbs | 5.014 | 5.112 | +1.96% |

The Table IX `VecMul` implementation is add plus Barrett reduction in both
trees; the label is retained only to match the artifact table. Other old
perf-file rows are not normalized into speedups because their public shapes
changed. For example, v2.0.0 HEAdd measures one `(B, limbs, degree)`
polynomial (28.551 µs at CL B=1), while current HEAdd measures a two-element
ciphertext and exposes the reduction backend. Treating 28.551 versus
69.363/37.573 µs as the same workload would be misleading. v2.0.0 also has
no bootstrap performance file matching the current sharded bootstrap-PtCt
benchmark.

Raw current CSVs live under `jaxite_word/log/` and the aggregate CSVs beside
the perf tests. The 171 Xprof directories occupy about 34 GiB; keep that in
mind before launching another full profiler sweep.

---

## Full TPUv6e-8 validation (2026-08-10)

This section records a fresh full-tree run of the 2026-08-10 pre-release tree
on one eight-chip TPU host. It supersedes the latency claims above for the current
secure, canonical Mapping path; older values were measured on retired
pre-unification implementations and are not current model results.

### Environment and coverage

| item | value |
|---|---|
| accelerator | 8 × `TPU v6 lite` |
| backend | TPU |
| Python | 3.13.14 |
| JAX | 0.11.0 |
| tree | 2026-08-10 pre-release (between v2.0.0 and 3.0.0) |

Every current `*_test.py` and `*_perf_test.py` file was executed serially so
that only one process owned the TPU slice. Three demo tests use legacy plain
imports and initially needed the documented `jaxite_word/` `PYTHONPATH`
entry; all three passed when relaunched with that path.

| suite | files | pass | fail |
|---|---:|---:|---:|
| functional tests | 37 | 36 | 1 |
| kernel performance tests | 11 | 11 | 0 |
| secure model performance tests | 3 | 0 | 3 |
| **total** | **51** | **47** | **4** |

The only functional-file failure is
`jaxite_word/nn_test.py`: 168 of 169 tests pass, while
`TemplateSelectionTest.test_a_user_can_register_the_parser_for_a_new_family`
expected a runtime family-registration hook and descriptor type that were not
part of the module's public surface at that commit.

### Current secure model status

All four registered demos vectorize and pack successfully at 128-bit
security. None currently reaches encrypted inference on this host, so there
is no valid current end-to-end model latency to report.

| model | secure plan | real execution result | latency |
|---|---|---|---:|
| LoLA | degree 32768, 6Q, `dnum=3`, depth 5 | setup fails after 430.4 s: a 6.00 GiB TPU reservation has only 5.22 GiB available | n/a |
| LeNet | degree 32768, 8Q, `dnum=4`, depth 7 | setup fails after 479.4 s: an 8.00 GiB allocation has only 6.21 GiB available | n/a |
| AlexNetTiny | degree 32768, 8Q, `dnum=4`, depth 7 | B=1, B=32, and B=8/eight-chip setup all fail on the same 8.00 GiB allocation | n/a |
| AlexNetFull | degree 65536, 16Q, `dnum=8`, depth 15 | setup fails after 671.7 s: `max(r,c)=16384` violates the uint32 BAT accumulator requirement `max(r,c) < 2^14` | n/a |

Only LoLA has a shipped checkpoint. LeNet and both AlexNet variants use
deterministic random parameters, so even a successful run would exercise the
pipeline rather than establish model accuracy.

### HE operator performance

Unless noted otherwise, values below are warm Xprof device time in
microseconds. `B=8 sharded` is the total time for a batch of eight distributed
over all eight chips. Each cell is `Barrett / Montgomery`; bold is faster.

| operator (degree 65536, 51 limbs) | B=1 | B=8 sharded |
|---|---:|---:|
| HEAdd | 69.43 / **36.50** | — |
| HESub | 58.72 / **40.43** | — |
| HEMul, no relinearization | 162.35 / **90.55** | — |
| full HEMul | 3984.63 / **3296.11** | 4167.28 / **3425.74** |
| HERot | 2704.21 / **2448.32** | 2754.28 / **2569.39** |
| Rescale | 660.23 / **499.90** | 809.83 / **556.41** |
| PtCt VPU | 83.41 / **56.65** | — |
| PtCt BAT | 134175.81 / 134134.25 | — |

Montgomery eight-chip throughput is approximately 2,335 HEMul/s, 3,114
HERot/s, and 14,378 Rescale/s. The PtCt BAT path is a major performance
anomaly: 134.1 ms versus 56.65 µs for the VPU path, about **2,368× slower**,
despite its performance test passing.

### NTT performance

Each entry is single-device B=8 → B=8 sharded over eight chips, in
microseconds.

| degree / limbs | Montgomery | Barrett | Shoup | BAT-lazy |
|---|---:|---:|---:|---:|
| 4096 / 4 | 6.63 → 145.61 | 9.58 → 36.67 | 15.08 → 119.58 | 16.02 → 300.98 |
| 8192 / 8 | 20.76 → **8.38** | 32.11 → 31.75 | 51.28 → 50.78 | 57.30 → 40.57 |
| 16384 / 16 | 107.61 → 54.36 | 152.12 → 128.84 | 334.91 → 103.34 | 248.19 → **38.59** |
| 65536 / 48 | 1870.70 → 323.82 | 2226.85 → 310.46 | 7365.31 → **266.18** | 3453.12 → 603.26 |

Sharding loses at degree 4096 because launch and collective overhead dominate.
It becomes beneficial at larger sizes, and all sharded Shoup cases now run.

### Basis-conversion performance

Each cell is `default / Montgomery / Montgomery-BAT / BAT`, in microseconds
for the complete batch.

| limbs | B=1 | B=256 |
|---|---:|---:|
| 12→28 | 805.91 / 1020.67 / 242.76 / **129.94** | 30866.85 / 38990.43 / 37943.93 / **30144.80** |
| 12→36 | 1044.02 / 1303.83 / 259.39 / **141.11** | **32231.28** / 41727.53 / 42738.16 / 32665.18 |
| 16→40 | 163.85 / 202.84 / 112.66 / **67.81** | 42597.78 / 54675.79 / 42506.26 / **31278.06** |
| 20→48 | 364.20 / 493.44 / 382.31 / **196.78** | 68196.76 / 81654.47 / 56781.84 / **43694.80** |
| 24→56 | 318.12 / 372.48 / 159.89 / **97.18** | 84287.18 / 100312.85 / 61655.78 / **46095.52** |

BAT is fastest in 31 of the 35 swept configurations and reaches up to 7.4×
speedup over the default path.

### Other kernel and correctness results

| measurement | result |
|---|---|
| modular multiply, 51 moduli, B=1/B=2 | Montgomery **76.25/146.59 µs**; Barrett 86.63/168.17 µs; Shoup 193.44/718.94 µs; BAT-lazy 493.23/1437.57 µs |
| BSGS matvec, degree 4096 | n=64: **1.356 ms** device; n=256: **3.427 ms** device |
| eight-chip bootstrap PtCt, degree 256 | 40 towers: 0.352 ms / 22,698 ct/s; 24 towers: 0.338 ms / 23,669 ct/s; 12 towers: 0.325 ms / 24,583 ct/s |
| N=256 bootstrap precision | single pass: 15.1 bits minimum; Meta-BTS: 24.3 bits; Montgomery parity: 14.4 bits |
| PyTorch BSGS convolution | correct; 5.293 s versus 16.9 ms for `torch.nn.Conv2d`, so native convolution is 312.9× faster |

The bootstrap performance file measures the dominant sharded linear-transform
PtCt kernel; it does not claim a full `EvalBootstrap` latency.

### Explicit skips

Nine individual cases remain skipped without causing a file failure:

* AlexNetTiny and AlexNetFull accuracy gates: trained checkpoints are absent.
* LeNet trained-weight gate: no checkpoint is shipped.
* LoLA's opt-in end-to-end functional gate: its separately enabled performance
  path was attempted above and failed during Mapping materialization.
* N=4096 Meta-BTS and N=8192 single-pass bootstrap: explicit ~70-minute and
  ~2-hour/~37-GB opt-ins.
* Two zero-matrix sparse-lowering cases have no diagonals to subset.
* The Pallas BAT case skips with the message "designed for TPU only" even on
  this TPU run; that skip condition/message needs follow-up.
