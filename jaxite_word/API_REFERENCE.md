# CROSS frontend, packing, mapping, and context API

A model enters this compiler exactly once, as an ordinary
`torch.nn.Module`, and passes through four owners:

| Phase | Module | Owns |
| --- | --- | --- |
| Vectorize | `jaxite_word.nn` | Traces the module with `torch.fx`, picks a vectorization template per layer, substitutes a polynomial for each non-linear activation, and emits a semantic `VectorizedProgram`. Names no HE concept. |
| Pack | `jaxite_word.packing` | Reconciles slot layouts globally, emits the ordered PP-op DAG with its plaintext constants, and derives the secure `RingConfig` from the program's own slot demand and emitted depth. |
| Map | `jaxite_word.mapping` | Schedules that DAG: level/scale/noise propagation, BSGS plans per operation, compute and virtual-memory trajectories, key and control preparation, device placement, constant encoding, JAX lowering. |
| Evaluate | `jaxite_word.ckks_ctx` | The codec, HE primitives and operators the schedule binds against. |

Every public ciphertext boundary accepts or returns a `Polynomial` with one
canonical rank-5 payload:

```text
(batch, num_elements, r, c, num_moduli), where r * c == degree
```

The CKKS boundary fixes `precision=32` and a `uint32` payload/modulus dtype.
Self-consistent wider wrappers are private implementation values, not a second
supported ciphertext representation. Public codec and evaluator operations
reject raw arrays and noncanonical wrappers.

Low-level operator classes, mapping-backend state, and raw-array kernels are
private. Packed operations are plain immutable tuples rather than public graph
objects. `Mapping.compute_trajectory` records dependency order, input/output
levels, and virtual buffers. `Mapping.memory_trajectory` records each value's
virtual buffer and live interval.

## Static model compilation and inference

Run from the repository root, or install the package on `PYTHONPATH`:

```python
import torch

from jaxite_word import Mapping, nn, packing


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(16, 8)
        self.fc2 = torch.nn.Linear(8, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = x * x          # the one non-linearity CKKS evaluates natively
        return self.fc2(x)


model = Model().double().eval()

# 1. Semantics: what the model computes, per unbatched sample.
program = nn.vectorize(model, (16,))

# 2. PP-ops plus the ring they need. Nothing above chose the ring; it follows
#    from this program's widest live layout and the depth it actually emits.
packed = packing.pack(program)

# 3. The schedule. The ring comes from `packed`, so normal callers pass
#    neither a parameters dict nor a dnum.
mapping = Mapping(packed, global_batch=1)

scores = mapping.infer(model_input)   # encrypt, execute, decrypt, unpack
```

For the example above, `packed` reports:

```text
operations : ('matvec', 'add_plain', 'square', 'matvec', 'add_plain')
depth      : 3
degree     : 16384   num_slots : 8192   num_q : 4   security : 128-bit
```

`packing.pack` materializes every matvec constant eagerly by default, which is
exact but quadratic in ciphertext width. Pass
`packing.PackingPolicy(lazy_constants=True)` to keep the matrix-free source
instead; at demo width the difference is gigabytes.

The public deployment flow is deliberately sequential:

| Stage | Public API | Responsibility |
| --- | --- | --- |
| Vectorize | `program = nn.vectorize(model, input_spec)` | Traces a real `torch.nn.Module`, selects a vectorization template per layer, substitutes activation polynomials, and records a semantic DAG with content-addressed weights. Rejects any node it cannot map rather than skipping it. |
| Pack | `packed = packing.pack(program, policy=None)` | Reconciles layouts, emits the PP-op DAG and its constants at exactly the ring's slot count, and derives the secure `RingConfig`. |
| Map | `mapping = Mapping(packed, global_batch=B, devices=devices, ...)` | Propagates level/scale/noise metadata, chooses a BSGS plan per matvec, builds compute and virtual-memory trajectories, installs exact keys and controls, fixes device placement, encodes constants, and performs JAX lowering. |
| Serve | `mapping.infer(value)` or `mapping.execute(ciphertext)` | Executes the prepared Mapping with no packing, key generation, control generation, constant encoding, or bootstrap setup. |

`Mapping` takes the `packing.Packing` that `packing.pack` returns, by type. A
duck-typed look-alike is refused, because one producer of a packed program is
the point. Its remaining keywords are scheduling only -- `global_batch`,
`devices`, `dnum`, `compile_mode`, `bsgs_n_jobs`, `bsgs_streaming`,
`cache_rotation_keys`, `headroom`, `input_scale`, `input_nsd`, `perf_test`.
`dnum` may only restate the value in the secure ring plan; none of these
options change what the program computes or which ring it runs on.

Both are still accepted in narrow forms. A parameters dict must be wrapped in
`packing.test_only_parameters(...)` -- how a test states that it is
deliberately running on a small insecure ring -- and a plain dict is refused. A
`dnum` is accepted when it restates the ring's planned value and refused when
it contradicts it. Both guards exist for the same reason: the parameters that
execute must be the parameters the security analysis validated.

Mapping construction selects a fused lowering when every operation is
traceable. Bootstrap is a hard temporal barrier and divides execution into
compiled regions; nothing inserts one automatically, and a program too deep
for every supported degree is reported rather than silently segmented.
`compute_trajectory` and `memory_trajectory` are Mapping-level dependency and
virtual-buffer schedules; JAX may optimize the physical instruction order,
device placement, and buffer reuse within a fused region.

`mapping.estimate_live_memory()` returns a model-independent logical byte
estimate for the public input/output ciphertexts, encoded matvec/plaintext
constants, evaluation key, and planned rotation keys. It is the reporting API
for demos; it is not an allocator measurement and may differ from JAX's
physical HBM use. Demos must not inspect `Mapping.operation_plans`,
`Mapping.value_specs`, or CKKSContext parameter caches to reconstruct it.

`mapping.release()` returns a materialized Mapping's device memory. The
compiled executable, with the evaluation keys and encoded constants it
captured, stays resident as long as JAX's tracing caches reference the
compiled program, so dropping the Python object alone does not free HBM.
`release` cuts those references, evicts the codec caches keyed by the
Mapping's context, clears the JAX caches, and leaves the Mapping unusable;
build a new one for the next placement. Call it before materializing another
Mapping in the same process. `CanonicalDemo.release()` and the demo adapters'
`release()` delegate to it.

The Mapping analyzer and executable materializer live in `mapping.py`, which
imports no frontend module. `ckks_ctx.py` owns the codec, primitives and
operators it binds against, and nothing else.

One Mapping owns one packed program and one context-specific deployment. The
minimal model boundary is exactly one input and one output:
`mapping.execute(Polynomial)` returns one `Polynomial`. Branching and
reduction remain internal to the packed operation DAG.

### Encrypted demo conformance

The compatibility entrypoints under `demos/` do not define another compiler
or ciphertext API. Every registered encrypted demo uses the same two shared
owners:

- `CanonicalDemo` owns the ordinary `torch.nn.Module`, calls `nn.vectorize`
  and `packing.pack` exactly once, and materializes exactly one `Mapping` with
  scheduling fixed at construction.
- `CanonicalDemoAdapter` owns the common legacy model lifecycle: bind the
  caller's positional weights, materialize that demo's Mapping, and serve via
  `Mapping.infer`, `Mapping.encrypt_input`, and `Mapping.decrypt_output`.

A model file may declare its positional weight order, preprocess one client
sample before Mapping packs it, validate the ring that packing derived, and
print model-specific setup information. It may not reimplement `infer`, call
`ctx.encrypt_slots*` or `ctx.decrypt_slots*` directly, construct a Mapping,
or reach into Mapping/CKKSContext private state. This keeps LoLA, LeNet,
AlexNetTiny, and AlexNetFull on the same logical input/output boundary while
preserving their historical constructors.

Demo construction accepts only Mapping scheduling fields. Parameters dicts,
key material, and model/ring overrides are not demo configuration; advanced
direct-evaluator and deliberately insecure test setup remain the separate
routes documented below.

### Global batch and device placement

`global_batch` is the public ciphertext batch, and `devices` is the exact
ordered device tuple used to compile the Mapping. The global batch must be
divisible by the number of devices. Mapping derives the private local batch as
`global_batch // len(devices)`. If `devices` is omitted, Mapping selects the
first local JAX device.

Model wrappers may retain a model-level `batch` argument for the number of
inputs accepted by `infer`. Such wrappers must forward it to Mapping as
`global_batch`; `batch` is not a Mapping constructor keyword.

For the common case of `B` independent requests on `B` TPU devices, local
batch is one:

```python
import jax

B = 8
devices = tuple(jax.devices("tpu")[:B])
mapping = Mapping(packed, global_batch=B, devices=devices)

input_ct = mapping.encrypt_input(inputs[:B])
output_ct = mapping.execute(input_ct)
outputs = [mapping.unpack_output(value) for value in
           mapping.ctx.decrypt_slots_batch(output_ct)]
```

Both `input_ct` and `output_ct` are canonical rank-5 `Polynomial` values with
batch `B`. Mapping alone unwraps the payload, privately reshapes it to
`(B, 1, num_elements, r, c, num_moduli)`, invokes its prepared executable,
flattens the output's device and local-batch axes, and wraps the result as a
global-batch `Polynomial`. There is no public raw-payload or separate
multi-device execution API.

Device topology, global batch, local batch, modulus count, degree layout, and
payload dtype are all static compilation inputs. Changing any of them requires
a different Mapping. A single-device Mapping is the same abstraction with one
device; callers still use `mapping.execute`.

Bootstrap operations retain segmented eager barriers and currently require a
single-device Mapping. Constructing a bootstrap Mapping with more than one
device fails before context initialization.

### The frontend: semantics, not primitives

`jaxite_word.nn` describes a model; it builds no slot constant, no diagonal
and no HE operation. Its surface is small:

- `nn.vectorize(model, input_spec)` -- the only way in. Traces a real
  `torch.nn.Module` and returns a `VectorizedProgram`.
- `nn.VectorizedProgram` / `nn.VectorizedLayer` -- the semantic DAG. A layer's
  kind is one of `linear_transform`, `polynomial_activation`, `add` or
  `layout`. Rescale and bootstrap are deliberately not expressible: they are
  HE realizations that packing materializes and mapping schedules.
- `nn.LayoutTemplate` plus `nn.register_layout_template` -- how a layer is
  vectorized, and the condition under which that applies. Several templates
  may claim one operator; the highest-priority one whose condition holds wins.
- `nn.ActivationPolynomial` plus `nn.register_activation` -- what replaces a
  non-linear activation. ReLU and SiLU substitute to `x**2` by default, and
  every substitution warns, naming the polynomial it chose.
- `nn.WeightTable` / `nn.WeightTableBuilder` -- content-addressed weights, so
  a program digests deterministically and survives serialization.

Adding a template takes two registrations, one per phase: the frontend one
above, and `packing.register_template_lowering(template_id, alias=...)` for
how it is realized physically. `alias` reuses an existing lowering, which is
the common case when only the condition or priority differs; a genuinely new
realization supplies a `recipe_factory` instead. A template registered without
a lowering vectorizes and then fails to pack, naming the id.

Importing `jaxite_word.nn` pulls in neither torch, JAX, nor the crypto stack;
torch is imported inside `vectorize` alone.

### Packing: layouts, PP-ops, and the security plan

`packing.pack(program, policy=None)` returns a `packing.Packing`:

- `operations` -- the ordered PP-op DAG, each entry
  `(value_id, kind, inputs, argument)`. Kinds are `matvec`, `square`, `mul`,
  `add`, `sub`, `add_plain`, `mul_plain`, `rotate`, `rescale`, `level_reduce`
  and `bootstrap`. The reserved input value is named `input`.
- `ring_config` -- the security parameter set, chosen by `he_params` as the
  smallest supported degree meeting this program's slot demand and depth.
  CKKS packs `degree // 2` slots, so the degree is never the next power of two
  of the slot count.
- `num_slots`, `depth`, `fingerprint`, and the logical shapes and coordinate
  maps needed to pack an input and unpack an output.

Every ciphertext value has one physical layout, `(num_slots,)`. Logical shapes
travel as metadata, and a consumer expecting a different layout gets a
concrete repack matrix -- or a precise refusal when the slot counts cannot
match.

Depth is recomputed from the operations actually emitted, never taken on trust
from the frontend, and measured as the longest path through the DAG: the two
arms of a residual cost what the deeper arm costs, not their sum. Where they
meet, the shallower arm consumes its missing levels the way the deeper one
did, so both reach the join at equal level, equal scale and `nsd` back at 1.

`packing` records no BSGS ratio. The baby-step/giant-step split is a
scheduling choice Mapping makes per matvec, from the diagonals that operation
actually has.


### Advanced direct-evaluator setup

Code that intentionally invokes individual `ctx.he_*` evaluators without a
compiled `nn` model must initialize their shared key/parameter cache directly:

```python
ctx = CKKSContext(params)
ctx.program_initialization(
    total_rotation_indices=[1, 2],
    dnum=3,
    r=4,
    c=4,
    batch=1,
)
```

This advanced route requires the caller to know the complete rotation set.
Static inference should construct `Mapping(packing.pack(program),
global_batch=..., devices=...)`, which computes the set and creates every
operation-specific control before serving. Direct accessors are level-indexed
and lazy; obtain every accessor during offline setup—not on a latency-sensitive
first request:

```python
mul_at_level = ctx.he_mul[output_level]
rotate_at_level = ctx.he_rot[level, rotation_index]
# Later request execution reuses these already-materialized evaluators.
```

The read-only evaluator accessors are:

```text
ctx.he_add[level]
ctx.he_sub[level]
ctx.he_mul[level]
ctx.he_rot[level, rotation_index]
ctx.he_rescale[source_level, destination_level]
ctx.ptct_mul[level]
ctx.bsgs_matvec[level, dimension]
ctx.he_bootstrap
```

## Codec operations

The staged codec methods operate on `Polynomial` values; the `_slots` methods
combine slot encoding/decoding with encryption/decryption:

| API | Input | Output | Intended use |
| --- | --- | --- | --- |
| `ctx.encode(slots)` | One host slot vector | A batch-1, one-element plaintext `Polynomial` in NTT/evaluation form | Offline plaintext construction. No encryption key is required. |
| `ctx.encrypt(plaintext)` | A batch-1, one-element NTT plaintext `Polynomial` | A batch-1, two-element ciphertext `Polynomial` | Encrypt an already encoded plaintext; it does not encode slots. |
| `ctx.decrypt(ciphertext)` | A batch-1, two-element ciphertext `Polynomial` | A batch-1, one-element plaintext `Polynomial` in coefficient form | Obtain a plaintext object for a separate decode step. |
| `ctx.decode(plaintext)` | A batch-1, one-element coefficient-form plaintext `Polynomial` | A JAX array of decoded slots | Decode the output of `decrypt`; use `is_ntt=True` for the NTT-form output of `encode`. |
| `ctx.encrypt_slots(slots, scale=None)` | One host slot vector | A batch-1, two-element ciphertext `Polynomial` | Preferred end-to-end input path for one ciphertext. |
| `ctx.decrypt_slots(ciphertext, scale=None, slots_to_decode=None)` | One batch-1 ciphertext `Polynomial` | A host NumPy array | Preferred end-to-end output path for one ciphertext. |
| `ctx.encrypt_slots_batch(slot_vectors, scale=None)` | An iterable of `B` host slot vectors | One ciphertext `Polynomial` with batch dimension `B` | Preferred batched input path. |
| `ctx.decrypt_slots_batch(ciphertext_batch, scale=None, slots_to_decode=None)` | One ciphertext `Polynomial` with batch dimension `B` | A list of `B` host NumPy arrays | Preferred batched output path. |

`encode`, `encrypt`, `decrypt`, `decode`, `encrypt_slots`, and `decrypt_slots`
support exactly one batch item. Use the `_slots_batch` pair for `B > 1`. The
batch APIs use one scale for the entire batch. They reduce transfers and
wrapper overhead but do not promise one vectorized computation across the
batch.

`encode` always uses `ctx.scaling_factor`. `encrypt_slots` and its batch form
use that value by default or an explicit scale. Fresh wrappers track the actual
scale; encryption and decryption preserve it. The decode helpers use an
explicit scale when supplied, otherwise the tracked scale, then
`ctx.output_scale` as a fallback. Rotation and add/sub preserve scale metadata,
rescale updates it, and Mapping execution restores the exact planned metadata
after multiplication, plaintext multiplication, `matvec`, and bootstrap.

The `_slots` paths avoid an intermediate plaintext wrapper and are the normal
inference boundary. Staged methods remain useful for tests, diagnostics, and
direct operations requiring a plaintext object. `slots_to_decode` trims only
the returned values; the decoder still processes all configured slots.

Construct the Mapping with `global_batch=B`. On `D` devices, its context and
evaluator cache use local batch `B // D`; with one device, local and global
batch are the same. For whole-program bootstrap-free execution, omitting
`devices` automatically chooses the widest prefix of visible devices whose
count divides `B`; explicit device tuples remain authoritative. The context's
batch codec methods still create and consume
the global-batch public `Polynomial`. For advanced direct-evaluator use without
Mapping, pass the desired evaluator batch directly to
`program_initialization`.

There are no public module-level `fast_*` codec functions.

## Direct evaluator operations

### Add and subtract

```python
summed = ctx.he_add[level].add(left, right)
difference = ctx.he_sub[level].sub(left, right)
```

Both operands must be canonical `Polynomial` ciphertexts at the declared
level, with identical layout, modulus prefix, batch size, element count, and
compatible scale metadata. These operations do not perform implicit alignment.

### Ciphertext multiplication

```python
result = ctx.he_mul[output_level].mul(left, right)
```

The unified call owns tensor multiplication, relinearization, and rescaling.
Schedules that deliberately own those stages can control them explicitly:

```python
multiply = ctx.he_mul[output_level]
three_elements = multiply.hemul_no_relin(left, right)
two_elements = multiply.relinearize(three_elements)
result = ctx.he_rescale[
    output_level + 1, output_level
].rescale(two_elements)
```

`hemul_no_relin` performs only the tensor product and `relinearize` performs
only key switching; neither drops a modulus level. Their controls are
materialized when `ctx.he_mul[output_level]` is obtained, so direct callers
must obtain the accessor during offline setup. The accessor index denotes the
full multiplication output level, making `ctx.he_mul[ctx.max_level]` invalid.

### Rotation

```python
result = ctx.he_rot[level, rotation_index].rotate(ciphertext)
```

For a static network, Mapping analysis discovers the exact rotation set and
Mapping construction installs it on the owned context. Direct evaluator users
must include the index during direct setup. Materializing
`ctx.he_rot[level, rotation_index]` performs the offline HERot phases:
rotation-independent controls are generated first, then that rotation's
evaluation key and automorphism map are bound. `rotate` itself only executes
the prepared runtime path.

By default, one max-level rotation key is reused at lower levels: inactive Q
limbs are removed, every P limb and the max-level HYBRID partition boundaries
are retained, and empty active-level partitions are discarded by the
evaluator. The diagnostic low-memory
`CROSS_SKIP_TOPLEVEL_ROTKEYS=1` mode instead generates a level-specific,
operator-owned key when the accessor is materialized. That key is not placed
in the global raw/formatted key caches, but it remains owned by the cached
accessor; it is therefore on-demand rather than request-time ephemeral.

### Rescale

```python
result = ctx.he_rescale[source_level, destination_level].rescale(ciphertext)
```

This drops `(source_level - destination_level) * composite_degree` RNS towers.

### Plaintext-ciphertext multiplication

```python
result = ctx.ptct_mul[level].mul(ciphertext, plaintext_ntt)
```

The two-argument form is stateless and canonical. For a `mul_plain` PP-op,
Mapping construction encodes and captures the plaintext once so request
execution does not reinstall or re-encode it. When both operands carry tracked
CKKS scales, the direct result tracks their product and increments its
noise-scale degree.

### BSGS matrix-vector multiplication

```python
op = ctx.bsgs_matvec[level, dimension]
op = ctx.bsgs_matvec[level, dimension, n1, n2]

op.preprocess(matrix)
result = op.matvec(ciphertext)
```

`preprocess` accepts a dense matrix, a sparse `{diagonal_index: values}`
mapping, or a matrix-free source. Pass `memory_bounded=True` to encode and
multiply one giant step at a time. Preprocessing is offline and `matvec`
consumes one logical level.
Mapping analysis selects the BSGS factorization and aggregates its exact
rotations for the matvec PP-ops packing emits. Direct users are responsible
for planning and installing their required rotations before accessing the
operator.

### Bootstrap

Mapping-owned bootstrap execution delegates to the repository's single private
engine through its context.

There is currently no way to request one from the public pipeline. A model is
a `torch.nn.Module`, which has no notion of a bootstrap, and `packing.pack`
never inserts one: it reports a program too deep for every supported degree
rather than segmenting it silently. `bootstrap` remains a PP-op kind that
Mapping schedules, so a packed program that contains one is executed
normally -- but constructing that program is not something the public API
offers today. The direct standalone route below is the supported way to
refresh a ciphertext.

When a packed program does contain a bootstrap, Mapping construction computes
its rotations, freezes the input level/scale/noise contract, encodes scalar
plaintexts, and materializes all evaluator controls. Bootstrap is a hard temporal barrier between compiled
regions. A request outside that contract is rejected instead of generating
parameters at runtime. `mode="meta"` uses the same private engine for two-pass
refinement.

For an advanced standalone call, configure and set up the context's same
bootstrap accessor with a precomputed bootstrap configuration, then call its
`bootstrap` method. There is no second bootstrap engine.

## Levels

One logical level is one composite rescale step:

```text
max_level = (len(q_towers) - 1) // composite_degree
num_q_at_level(L) = len(q_towers)
                    - (max_level - L) * composite_degree
```

Fresh ciphertexts use `max_level`. Operations reject wrappers whose modulus
metadata does not match the selected level.

## Private fused regions

During Mapping construction, trusted backend code may bind raw-array hooks such
as `_square_array`, `_rotate_array`, and `_matvec_array` into one JAX transform.
Bootstrap divides the transform into compiled segments.
For non-bootstrap mappings, Mapping compiles its private raw callable for the
device tuple supplied at construction. Device-axis reshaping and placement are
part of that private materialization.

Those hooks are unstable implementation details and may accept raw
`jax.numpy.ndarray` values. `mapping.execute` remains the public boundary and
accepts or returns only canonical `Polynomial` ciphertexts. Private fusion is
an execution lowering, not another ciphertext representation. JAX may optimize
physical scheduling within a fused region; it may not move work across a
bootstrap barrier.

Kernel naming distinguishes API role rather than Python value type:
`rotate`-style names are complete `Polynomial` operations;
`_<step>_array` names are validated raw-array composition boundaries; and
`_<verb>_<object>` names are implementation-only helpers. A helper may
manipulate JAX arrays but intentionally omits `_array` when it is not a
supported fusion boundary. Setup/configuration names likewise omit `_array`.
