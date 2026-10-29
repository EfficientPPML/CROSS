# SHIP in CROSS

This is a self-contained port of the SHIP algorithm from Google Jaxite PR147
(base `89797d8`, including the local signed-key, independent-output-key, and
overflow fixes). No Jaxite checkout or installation is required. Adapted Google
sources retain their license headers; the upstream Apache 2.0 license is in
`LICENSE`. Algorithm/key-generation code without individual headers is also
derived from that Apache-licensed source. Native CROSS kernels live alongside
the offline key builders.

The context owns SHIP through the same bootstrap facade as CKKS:

```python
from ship import SHIPConfig

# ctx: initialized CROSS CKKSContext, with keys available during trusted setup.
config = SHIPConfig(weight=31, theta=9, gamma=8)
plan = config.plan(ctx)  # key-free input/output metadata
ctx.he_bootstrap.configure(scheme="ship", ship_config=config).setup()
ciphertext = ctx.encrypt(ctx.encode_at_level(values, 1, scale=plan.input_scale))
refreshed = ctx.he_bootstrap.bootstrap(ciphertext)
decoded = ctx.decrypt_slots(refreshed)
```

`configure` validates the plan; `setup` generates evaluation keys, encodes the
input-transform diagonals, and prepares compiled kernels. Online ciphertexts
use CROSS's canonical `(batch, elements, rows, columns, limbs)` layout and
`Polynomial` metadata. The `SHIPBootstrap(ctx, ...)` convenience wrapper
still works, but delegates to this context-owned engine.

An explicit Packing bootstrap operation can use `SHIPConfig` as its argument:
`('refresh', 'bootstrap', ('input_value',), config)`. Mapping validates its
input level/scale/noise scale degree during planning, prepares it through the
same facade, and treats it as a fusion barrier. The frontend does not insert
bootstraps automatically. SHIP's input-transform rotation keys belong to its
private two-Q-limb context; setup prepares them, rather than registering them
as ordinary full-chain Mapping rotations.

`bootstrap` performs encrypted slot-to-coefficient conversion, switches to an
internal sparse signed key, and executes SHIP's half-bootstrap back to the
context's original key. It does not decrypt or re-encrypt. Setup generates the
sparse secret locally and retains only its evaluation keys. The CROSS context
itself retains its normal trusted key-generation state.

## Arithmetic integration

* CROSS `NTTCiphertextBarrettContext`: NTT/INTT with fused twist tables and BAT.
* CROSS `BConvBarrett`: guarded BAT basis extension.
* CROSS `_HERescaleKernel`: centered lifting and Barrett reductions instead of
  SHIP's integer-remainder-based rescaling.
* CROSS `_HEMulKernel`: tensor products, relinearization, and modulus down.
* CROSS `BarrettContext`: plaintext products, selector products, and odd leaves
  in the product tree.
* CROSS hoisted BSGS: the encrypted input transform. Its shared controls and
  array encoding are scoped to this SHIP instance, with no global monkeypatch.

Default trusted setup also uses CROSS's vectorized OS-random uniform and
discrete-Gaussian samplers. The inherited deterministic source is used only
in tests. Host NTT reference tables are cached for larger key-generation jobs.

SHIP-specific selector keys, binary mux orchestration, online phase encoding,
and its 2-by-2 BAT key product remain algorithm-specific. Evaluation keys are
explicit runtime array operands; controls are ordinary
CROSS operators captured by the compiled kernels. There is no custom
ciphertext PyTree or serialized evaluator object in the online path. Private
host records remain only for trusted key generation and reference decoding.
CROSS's mutable multiplication workspaces are copied while tracing so calls
and `vmap` do not retain tracers in evaluator state. Hoisted-transform JIT
wrappers are reused across requests and evicted with CROSS's low-memory caches.

The tensor-product and square primitives also reduce cross products before
adding them: raw two-product sums can exceed Barrett's valid input interval or
overflow uint64. This avoids reintroducing the earlier SHIP accumulation bug.

## Supported configuration

* Real slots, all N/2 slots populated; power-of-two N.
* Standard Barrett representation, `composite_degree=1`.
* Distinct NTT primes below 2^31, uint32 limbs, uint64 intermediates.
* Input: two Q limbs, noise scale degree 1, scale `16*q0/gamma` by default.
* Output: scale `q[-1]`, at least two Q limbs remain.
* Need at least `ceil(log2(weight))+3` Q primes and at least two P primes.
  The P product must also be large enough for the desired noise/accuracy;
  the minimum length alone is not an accuracy guarantee.

The reproduction profile uses N=16384, weight=31, theta=9, gamma=8,
Q=[31,30,30,30,30,30,30,30] bits and nine 30-bit P primes. It is the previous
functional comparison profile, **not a security-qualified HT14 profile**.
Montgomery ciphertexts, composite primes, complex-message bootstrapping and
arbitrary incoming scales are deliberately rejected or outside this API.

## Validation and tracing

From `jaxite_word`:

```sh
python -m pytest ship_test.py -q
python ship_benchmark.py --degree 128 --weight 31 --theta 9
python ship_benchmark.py --degree 16384 --profile-dir /tmp/ship-profiles \
  --output /tmp/ship-result.json
```

The tests cover host-reference NTT and integer rescaling, noisy half-bootstrap
with signed sparse and independent dense keys, full slot-to-slot operation
through CROSS's codec, reused evaluators with different inputs, and arithmetic
near the modulus boundary. The benchmark checks ramp and randomized real
inputs in [-0.125,0.125] against the 2^-10 maximum absolute error gate.

Profiling begins after setup, compilation, and correctness checks. It captures
only warmed full bootstraps. Raw XPlane files must be checked for dropped events
before deriving active device time; host duration or a truncated trace is not
a valid TPU latency measurement. The earlier 1.724 s / 4.736 s comparison
predates this port and must not be reported as its performance.

The native CROSS refactor measured **1.357 s median active TPU time** at
N=16384 and passed both full-size fixtures, compared with 1.576 s for the
first integration on the same parameter profile. The measured revision passed 13 SHIP tests on
CPU and TPU. See [validation results](VALIDATION.md) for samples, accuracy
metrics, comparison limits, and artifact hashes.

Against the earlier locally patched Jaxite SHIP full-bootstrap result
(1.724 s), the native implementation used **21.31% less active TPU time**
(**1.271x speedup**). This is a historical comparison across different v6e
chips/VMs and random keys, with matching arithmetic parameters; it is not a
controlled identical-ciphertext A/B or a comparison with LL13 half-bootstrap.
Exact parameters and samples are committed in [benchmark_results.json](benchmark_results.json).
