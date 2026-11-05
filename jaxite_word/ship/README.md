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
remain algorithm-specific. Selector products use HERot decomposition and key-switch
cores with raw uint32 keys, one lift/drop per mux step, and a two-step P drop.
Keys are assembled on the host before the final device transfer. Evaluation keys are
explicit runtime array operands; controls are ordinary
CROSS operators captured by the compiled kernels. There is no custom
ciphertext PyTree or serialized evaluator object in the online path. Private
host records remain only for trusted key generation and reference decoding.
CROSS's mutable multiplication workspaces are copied while tracing so calls
and `vmap` do not retain tracers in evaluator state. Noncollapsed hoisted-transform JIT wrappers are reused across requests by
level and modulus signature, independent of accessor eviction. Collapsed
transforms still construct their own wrappers. SHIP retains its prepared state.
Tree relinearization keeps its per-level, up-to-eight-partition keys, and
conjugation keeps its selector key: substituting context keys would change
those decomposition schedules. No context-key reuse or corresponding speedup
is claimed.

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

## Accuracy, depth, and security contract

SHIP approximates the map `gamma/(2*pi) * sin(2*pi*m/gamma)`. Its supported
accuracy profile is `gamma=8`, real inputs with `abs(m) <= 0.125`, and a
maximum absolute error target of `2^-10`. Larger messages are not an identity
bootstrap: at `m=1`, the sine approximation alone introduces substantial error.
An evaluator cannot check encrypted message amplitude. `decrypt_slots` checks
imaginary leakage/conjugate symmetry, not accuracy against the original real
message. The benchmark's separate comparison with known plaintext is essential.

The planner checks the actual product-tree recurrence `S_next=S*S/q_dropped`
and final encoder scale, rejecting chains with more than one bit of drift.
It also requires `log2(P/Q) >= max(20, log2(N)+12)`. These conservative admission
rules reject the reviewed equal-Q/P and mixed-bit chains; they are not a proof
that every accepted parameter set meets the error target. Run plaintext-based
correctness gates on each benchmark profile and random key set.

`plan.multiplication_levels` reports the terminal multiplication budget;
`plan.multiplication_levels_before_rebootstrap` reserves level 1 for the next
SHIP input. With weight 31 and eight Q primes, output is level 1: one terminal
multiplication, **zero multiplications between bootstraps**. Nine Q primes
leave level 2; ten leave level 3. Increasing Q requires increasing P to retain
the margin. A second bootstrap also requires the exact `plan.input_scale` and
noise scale degree 1; level headroom alone does not guarantee this. Public
rescale updates both scale and noise scale degree. Low-scale public multiply
uses tensor/relinearize followed by one rescale, rather than rounding its
operands down to scale one. At higher scales it retains CROSS's legacy
pre-rescale convention; combinations that cannot fit the tensor product are
rejected. This does not validate an arbitrary encrypted amplitude.

Public rotations automatically use centered ModDown for tracked Barrett scales
below `2^40`; low-scale BSGS contexts use it for baby and giant rotations.
Untracked raw kernels retain the recorded OpenFHE convention. A dedicated test
checks eight rotation indices at N=16384, where small-ring tests hide the bias.

Default `pack()` / `generate_ring_config()` use composite degree 2 and cannot
produce this SHIP configuration. SHIP currently requires explicit composite
degree 1 parameters; `test_only_parameters` bypasses security selection and
must not be interpreted as a secure deployment configuration.

The historical N=16384 profile has `log2(QP)` about 511 bits, above the 438-bit
128-bit classical uniform-ternary table limit. It is not security qualified at
any level: publishing full-chain keys exposes the full QP modulus. Raising N
to 32768 changes that table limit to 881 bits but does not establish the sparse
key's security, and materially increases setup and resident memory. The
internal weight-31 sparse key is also exposed through switching material over
`q0*p0*p1` (about 91 bits); this secret distribution has no security estimate
here. Weight is an algorithm parameter, not a security floor. These statements
identify missing qualification, not a claimed attack.

SHIP does not invoke the native bootstrap stage checkpoints. `LOW_MEM` does
not automatically evict its constants or keys; inserting such checkpoints
would move offline setup into online calls. Mapping's logical memory estimate
includes SHIP keys and transform arrays, but excludes executable copies,
allocator reservation, and transient buffers.

Plaintext multiplication also has a scale window. For the historical two-limb
output, encode the multiplier at the next dropped prime's scale (`q[1]`)
before public rescale. Tiny plaintext scales lose precision; scales such as
`2^40` can wrap the product modulo the two remaining Q primes even though the
decoder accepts the real result. Reducing to one Q limb is not a supported
precision-preserving multiplication route. As with ordinary CKKS, callers
must bound message magnitudes and test their complete application circuit.

The review also reported executable-cache deserialization failure at Q=10,
P=11. Its cause is unconfirmed; reproduction runs should disable persistent
JAX compilation caching, record the runtime versions, and retain setup logs.
No portable serialized or keyless SHIP setup API is provided in this patch.
