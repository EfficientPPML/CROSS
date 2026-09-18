# CROSS Context-Level API Reference

## Overview

This document describes the context-level HE operation API that wraps HEMul, HERot,
HERescale, and HEPtCtMul into level-indexed accessors on `CKKSContext`.

### Architecture

```
CKKSContext
  ├── program_initialization()    # Offline: creates cache + accessors
  │     ├── HEParameterCache      # Shared NTT/Barrett/psi, per-level BConv
  │     │     ├── SlicedNTTContext      (Q-only, shares parent twiddles)
  │     │     ├── SlicedBarrettContext  (Q-only, shares parent arrays)
  │     │     ├── BConvBarrett          (per-level, not shareable)
  │     │     └── LevelCiphertextHelpers (pre-allocated ct objects)
  │     ├── HEMulAccessor    → ctx.he_mul[level]
  │     ├── HERotAccessor    → ctx.he_rot[level, rot_index]
  │     ├── HERescaleAccessor→ ctx.he_rescale[src, dst]
  │     └── HEPtCtMulAccessor→ ctx.ptct_mul[level]
  └── encode / encrypt / decrypt / decode  (unchanged)
```

### Level Semantics

One level = one composite rescale step.
- `max_level = (len(q_towers) - 1) // composite_degree`
- `num_q_at_level(L) = len(q_towers) - (max_level - L) * composite_degree`
- Fresh ciphertexts are at `max_level`. Each multiply consumes one level.

---

## Files Modified

| File | Change |
|------|--------|
| `hemul.py` | Added `hemul_no_relin()`, `relinearize()` methods |
| `finite_field.py` | Added `SlicedBarrettContext` class |
| `ntt_mm.py` | Added `SlicedNTTContext` class |
| `ciphertext.py` | Added `ntt_ctx` injection parameter in `__init__` |
| `ckks_ctx.py` | Added `program_initialization()`, `max_level` property, imports |

## Files Created

| File | Purpose |
|------|---------|
| `he_params.py` | `HEParameterCache` — shared parameter storage, per-level BConv |
| `he_ops.py` | Operator wrappers (`HEMulAtLevel`, `HERotAtLevel`, `HERescaleOp`, `HEPtCtMulAtLevel`) and accessor classes |
| `ptct_mul.py` | `HEPtCtMul` — ciphertext-plaintext multiply with VPU and BAT/MXU paths |

---

## API Usage

### Setup

```python
ctx = CKKSContext(params)
ctx.program_initialization(
    total_hemul_levels=3,         # max multiplication depth
    total_rotation_indices=[1,2], # rotation indices to support
    dnum=3, r=4, c=4, batch=1)
```

### Multiply (ct × ct)

```python
# Full multiply (rescale + tensor mul + relin):
result = ctx.he_mul[level].mul(ct1, ct2)

# Split into tensor multiply and relinearization:
ct_3elem = ctx.he_mul[level].hemul_no_relin(ct1, ct2)  # 3-element output
result = ctx.he_mul[level].relinearize(ct_3elem)        # 2-element output
```

Input: two 2-element ciphertexts at level+1.
Output: 2-element ciphertext at level (after rescale).

### Rotate

```python
result = ctx.he_rot[level, rot_index].rotate(ct)
```

Input/output: 2-element ciphertext at the same level.

### Rescale

```python
result = ctx.he_rescale[src_level, dst_level](ct)
```

Drops `(src - dst) * composite_degree` moduli from the ciphertext.

### Ciphertext-Plaintext Multiply

```python
op = ctx.ptct_mul[level]
op.set_plaintext(pt_ntt)                  # plaintext in NTT domain
result = op.mul(ct)                       # VPU path (default)

op.precompute_bat(pt_ntt)                 # offline: BAT precomputation
result = op.mul(ct, use_bat=True)         # MXU path
```

No relinearization needed. Output stays 2-element.

---

## TPU Optimization Notes

### Data Layout
Ciphertexts use shape `(batch, num_elements, r, c, num_moduli)` where `r × c = degree`.
The `(r, c)` layout maps to TPU's `(8, 128)` SIMD granularity at production degrees.

### Compute Mapping
| Operation | TPU Unit | Mechanism |
|-----------|----------|-----------|
| NTT/INTT twiddle multiply | MXU | BAT: 8-bit matmul on decomposed twiddles |
| Element-wise modmul | VPU | 32-bit multiply + Barrett reduction |
| ct-pt multiply (VPU) | VPU | Same as modmul |
| ct-pt multiply (BAT) | MXU + VPU | Plaintext byte-decomposed, 8-bit einsum; Barrett on VPU |
| Barrett reduction | VPU | 32-bit shifts and multiplies |
| BConv basis change | MXU | BAT einsum on QHatModp matrix |

### Parameter Sharing
- **NTT twiddles**: Computed once at max level; sliced for lower levels via `SlicedNTTContext`.
  Q-only operations use sliced views (zero-copy). Q+P operations require fresh contexts per level.
- **Barrett parameters**: Sliced from max-level `BarrettContext` via `SlicedBarrettContext`.
- **Psi/inv_psi arrays**: Computed once for all Q+P moduli; sliced per level.
- **Eval/rotation keys**: Stored at max level; sliced via `concat(key[..., :num_q], key[..., -num_p:])`.
- **BConv parameters**: NOT shareable — `QHatInvModq` depends on the full product of moduli at each level.
  One BConv instance per level.
- **Q+P Barrett**: Constructed via `ConcatBarrettContext(sliced_q, p)` — concatenates
  the sliced-Q Barrett arrays with the fixed P Barrett arrays. No NTT twiddle factors
  needed for Q+P since it's only used for `mod_reduce`.

---

## Bug Fixes

### Automorphism index bug (rotation)
`find_automorphism_index_2n_complex(rot_index, m)` requires `m = 2*degree` (cyclotomic
order), not `m = degree`. The function was ported correctly from OpenFHE, but most
callsites passed `degree` instead of `2*degree`. This caused incorrect rotations for
`rot_index >= 2` at small degrees. Fixed in: `ckks_ctx.py`, `matvec.py`,
`ckks_ctx_test.py`, `herot_test.py`, `herot_perf_test.py`, `matvec_test.py`, `tabIX.py`.

### Q+P context optimization
Per-level Q+P Barrett/NTT contexts were unnecessarily recomputed from scratch. Since
Barrett reduction is purely element-wise per modulus and the Q+P context is only used
for `mod_reduce` (no NTT/INTT), added `slice()` and `concat()` methods directly to
`BarrettContext` and `NTTCiphertextContextBase`:

```python
# Slice Q-only view from max-level context (zero-copy JAX array slice)
barrett_q = barrett_q_max.slice(num_q_at_level)
ntt_q = ntt_q_max.slice(num_q_at_level, barrett_q)

# Build Q+P context by concatenation (no recomputation)
barrett_qp = barrett_q.concat(barrett_p)
ntt_qp = NTTCiphertextContextBase.mod_reduce_only(barrett_qp)
```

No ad-hoc subclasses needed — the operations live on the base classes themselves.

---

## Cross-Validation Summary

All operations verified against OpenFHE (`HEMul_ref.cpp`, `HERot_ref.cpp`,
`HEPtCtMul_ref.cpp`, `HERot_multiindex_ref.cpp`).

| Operation | composite_degree=1 | composite_degree=2 |
|-----------|--------------------|--------------------|
| `he_mul[L].mul` | PASS (5Q, 8Q, 9Q) | PASS (8Q skip_rescale + with_rescale) |
| `hemul_no_relin + relinearize` | Bit-exact with `mul` | Bit-exact with `mul` |
| `he_rot[L, idx]` all indices 1-7, -1 to -3 | PASS | PASS (rot 1 and 2) |
| `he_rescale[L, L-1]` | PASS | PASS (drops 2 moduli) |
| `ptct_mul` VPU | PASS, matches OpenFHE | — |
| `ptct_mul` BAT | Bit-exact with VPU | — |
