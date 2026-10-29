# Adapted from Google Jaxite PR147 for CROSS. See README.md and LICENSE.
"""Native CROSS array kernels for SHIP; ciphertext payloads stay rank five.

Controls are ordinary CROSS operator instances. Evaluation keys are explicit
array arguments to compiled kernels, never fields in an opaque kernel PyTree.
"""

import copy

import bconv
import finite_field
import hemul
import jax
import jax.numpy as jnp
import ntt_mm
import numpy as np
import polynomial
import rescale

from . import _host_types as types
from . import blind_rotate_utils, bootstrapping_utils, key_gen


def _ntt(moduli, r, c):
  return ntt_mm.get_shared_barrett_ntt_context(list(moduli), {"r": r, "c": c})


def _rescale(moduli, count, r, c, batch=1):
  kernel = rescale._HERescaleKernel(batch, 2, list(moduli), r, c)
  kernel.control_gen(composite_degree=count)
  return kernel


def _working_mul(kernel):
  """Detach CROSS's mutable Polynomial workspaces, sharing immutable tables."""
  result = copy.copy(kernel)
  for name, value in vars(kernel).items():
    if isinstance(value, polynomial.Polynomial):
      setattr(result, name, copy.copy(value))
    elif (
      isinstance(value, list) and value and isinstance(value[0], polynomial.Polynomial)
    ):
      setattr(result, name, [copy.copy(item) for item in value])
  return result


class _BATSwitch:
  """SHIP's encrypted two-component selector product, using CROSS controls."""

  def __init__(self, q, p, r, c):
    self.q, self.p = list(q), list(p)
    self.r, self.c, self.degree = r, c, r * c
    self.ntt_q, self.ntt_p = _ntt(q, r, c), _ntt(p, r, c)
    self.ff = finite_field.BarrettContext(q + p)
    self.bc = bconv.BConvBarrett(q + p)
    self.bc.control_gen([[list(range(len(q))), list(range(len(q), len(q) + len(p)))]])
    self.drop = _rescale(q + p, len(p), r, c)
    self.block = min(128, self.degree)

  def format_key(self, key0, key1):
    """Trusted setup: convert two encrypted selector columns to BAT bytes."""
    data = jnp.stack([key0, key1], axis=1).transpose(2, 3, 0, 1).astype(jnp.uint64)
    shifted = jnp.stack([data << (8 * i) for i in range(4)])
    reduced = (
      shifted % jnp.asarray(self.q + self.p, jnp.uint64).reshape(1, 1, -1, 1, 1)
    ).astype(jnp.uint32)
    bat = jax.lax.bitcast_convert_type(reduced, jnp.uint8).transpose(1, 2, 3, 4, 0, 5)
    return bat.reshape(
      self.degree // self.block, self.block, len(self.q + self.p), 2, 2, 4, 4
    )

  def apply_array(self, ct, key):
    finite_field.check_rank5_array(
      ct,
      "SHIP selector switch",
      batch=1,
      num_elements=2,
      degree_layout=(self.r, self.c),
      num_moduli=len(self.q),
    )
    coeffs = self.ntt_q.intt(ct.reshape(2, self.r, self.c, len(self.q)))
    lifted = self.bc.basis_change(coeffs.reshape(2, self.degree, len(self.q)))
    p_ntt = self.ntt_p.ntt(lifted.reshape(2, self.r, self.c, len(self.p)))
    qp = jnp.concatenate([ct, p_ntt[None]], axis=-1)
    vector = jnp.moveaxis(qp[:, jnp.array([1, 0])], 1, -1).reshape(
      self.degree // self.block, self.block, len(self.q + self.p), 2
    )
    digits = jax.lax.bitcast_convert_type(vector, jnp.uint8)
    product = jnp.einsum(
      "ikjvq,ikjuvqp->ikjup", digits, key, preferred_element_type=jnp.uint32
    )
    wide = jnp.sum(
      product.astype(jnp.uint64) << jnp.array([0, 8, 16, 24], jnp.uint32), axis=-1
    )
    wide = jnp.moveaxis(wide.reshape(self.r, self.c, len(self.q + self.p), 2), -1, 0)[
      None
    ]
    return self.drop._rescale_array(self.ff.modular_reduction(wide))


class _SHIPKernel:
  """Prepared coefficient-to-slot SHIP evaluator with native CROSS controls."""

  def __init__(self, q, p, degree, weight, theta, gamma, layout):
    self.q, self.p = list(q), list(p)
    self.degree, self.weight, self.theta, self.gamma = degree, weight, theta, gamma
    self.r, self.c = layout
    self.scale = float(q[-1])
    self.ntt0 = _ntt(q[:1], *layout)
    self.ntt_qp = _ntt(q + p, *layout)
    self.ff_qp = self.ntt_qp.ff_ctx
    self.selector = _BATSwitch(q, p, *layout)
    self.fft = bootstrapping_utils.SpecialInverseFFT()
    self.fft.precompute_constants(degree)
    self.baby_maps = tuple(
      blind_rotate_utils.compute_automorphism_indices(degree, pow(5, -t, 2 * degree))
      for t in range(theta)
    )
    steps = ((degree // 2 + theta - 1) // theta - 1).bit_length()
    self.mux_maps = tuple(
      blind_rotate_utils.compute_automorphism_indices(
        degree, pow(5, -(2**i) * theta, 2 * degree)
      )
      for i in range(steps)
    )
    self.tree = []
    count = weight
    current = list(q)
    while count > 1:
      pairs = count // 2
      op = hemul._HEMulKernel(
        pairs, *layout, min(8, len(current)), 1, current, p, composite_degree=1
      )
      op.control_gen(skip_rescale=True, keygen_sizeQ=len(current))
      self.tree.append(
        (
          op,
          _rescale(current, 1, *layout, batch=pairs),
          _rescale(current, 1, *layout),
          len(current),
        )
      )
      current = current[:-1]
      count = pairs + count % 2
    self.final_ntt = _ntt(current, *layout)
    self.final_drop = _rescale(current, 1, *layout)
    self.conjugate = _BATSwitch(current[:-1], p, *layout)
    self.output_q = current[:-1]

  def setup_keys(self, sparse_coeffs, dense_key, rng):
    """Trusted host setup. Neither input support nor secret is retained."""
    cm_keys, mux_keys = [], []
    for index in np.flatnonzero(sparse_coeffs):
      cm, mux = key_gen.gen_hybrid_key(
        dense_key,
        int(index),
        int(index),
        int(sparse_coeffs[index]),
        self.theta,
        self.q,
        self.p,
        rng,
      )
      cm_keys.append(
        jnp.stack([jnp.stack([ct.data for ct in row]) for row in cm]).astype(jnp.uint32)
      )
      mux_keys.append(
        tuple(
          (
            self.selector.format_key(pair[0].key0.data, pair[0].key1.data),
            self.selector.format_key(pair[1].key0.data, pair[1].key1.data),
          )
          for pair in mux.keys
        )
      )
    relin = []
    for _, _, _, count in self.tree:
      sk = types.SecretKey(dense_key.data[:, :count], dense_key.moduli[:count])
      key = key_gen.gen_evaluation_key(sk, self.q[:count], self.p, min(8, count), rng)
      relin.append((jnp.asarray(key.a, jnp.uint32), jnp.asarray(key.b, jnp.uint32)))
    count = len(self.output_q)
    sk = types.SecretKey(dense_key.data[:, :count], dense_key.moduli[:count])
    conj = key_gen.gen_conjugate_key(sk, self.output_q, self.p, 1, rng)
    one = key_gen.gen_key1(sk, self.output_q, self.p, rng)
    return {
      "cm": jnp.stack(cm_keys).reshape(
        self.weight, 4, self.theta, 2, self.r, self.c, -1
      ),
      "mux": tuple(
        (jnp.stack([k[i][0] for k in mux_keys]), jnp.stack([k[i][1] for k in mux_keys]))
        for i in range(len(self.mux_maps))
      ),
      "relin": tuple(relin),
      "conjugate": self.conjugate.format_key(
        jnp.stack([conj.b[0], conj.a[0]]), one.data
      ),
    }

  def _encode_array(self, slots, scale, ntt):
    values = self.fft(slots)
    coefficients = jnp.concatenate([values.real, values.imag])
    moduli = jnp.asarray(ntt.moduli, jnp.uint32)
    poly = (jnp.round(coefficients * scale)[:, None] % moduli).astype(jnp.uint32)
    return ntt.ntt(poly.reshape(1, self.r, self.c, len(ntt.moduli)))[0]

  def _blind_rotate_array(self, plaintexts, cm, mux):
    current = jnp.zeros((1, 2, self.r, self.c, len(self.q + self.p)), jnp.uint32)
    moduli = jnp.asarray(self.q + self.p, jnp.uint32)
    for k in range(4):
      for t in range(self.theta):
        plain = jnp.take(
          plaintexts[k].reshape(self.degree, -1), self.baby_maps[t], axis=0
        )
        product = cm[k, t].astype(jnp.uint64) * plain.reshape(
          self.r, self.c, -1
        ).astype(jnp.uint64)
        total = current + self.ff_qp.modular_reduction(product)[None]
        current = jnp.where(total >= moduli, total - moduli, total)
    current = self.selector.drop._rescale_array(current)
    q = jnp.asarray(self.q, jnp.uint32)
    for i, permutation in enumerate(self.mux_maps):
      rotated = jnp.take(
        current.reshape(1, 2, self.degree, -1), permutation, axis=2
      ).reshape(current.shape)
      left = self.selector.apply_array(rotated, mux[i][0])
      right = self.selector.apply_array(current, mux[i][1])
      total = left + right
      current = jnp.where(total >= q, total - q, total)
    return current

  def apply_array(self, ciphertext, keys):
    finite_field.check_rank5_array(
      ciphertext,
      "SHIP half bootstrap",
      batch=1,
      num_elements=2,
      degree_layout=(self.r, self.c),
      num_moduli=1,
    )
    coeffs = self.ntt0.intt(ciphertext.reshape(2, self.r, self.c, 1)).reshape(
      2, self.degree
    )
    v0 = bootstrapping_utils.compute_v0_slots(
      coeffs[0], self.q[0], self.degree // 2, self.gamma
    )
    a, b = bootstrapping_utils.compute_a_slots(
      coeffs[1, : self.degree // 2], coeffs[1, self.degree // 2 :], self.q[0]
    )
    slots = jnp.stack([a, jnp.conj(a), b, jnp.conj(b)])
    plaintexts = jax.vmap(lambda s: self._encode_array(s, self.scale, self.ntt_qp))(
      slots
    )
    leaves = jax.vmap(lambda cm, mux: self._blind_rotate_array(plaintexts, cm, mux))(
      keys["cm"], keys["mux"]
    ).reshape(self.weight, 2, self.r, self.c, len(self.q))
    scale = self.scale
    for index, (base, drop, odd_drop, count) in enumerate(self.tree):
      pairs = leaves.shape[0] // 2
      op = _working_mul(base)
      op.setup_relinearization(*keys["relin"][index])
      paired = leaves[: pairs * 2].reshape(pairs, 4, self.r, self.c, count)
      product = op._hemul_no_relin_array(paired)
      current = drop._rescale_array(op._relinearize_array(product))
      if leaves.shape[0] % 2:
        ff = base.ct_obj.ntt_ctx.ff_ctx
        multiplier = ff.modular_reduction(jnp.asarray(round(scale), jnp.uint64))
        odd = ff.modular_reduction(
          leaves[-1:].astype(jnp.uint64) * multiplier.astype(jnp.uint64)
        )
        current = jnp.concatenate([current, odd_drop._rescale_array(odd)], axis=0)
      leaves = current
      scale = scale**2 / self.q[count - 1]
    scale0 = self.final_ntt.moduli[-1] * self.scale / scale
    pt = self._encode_array(v0, scale0, self.final_ntt)
    root = self.final_ntt.ff_ctx.modular_reduction(
      leaves.astype(jnp.uint64) * pt.astype(jnp.uint64)
    )
    root = self.final_drop._rescale_array(root)
    flipped = jnp.flip(root.reshape(1, 2, self.degree, -1), axis=2).reshape(root.shape)
    conjugate = self.conjugate.apply_array(flipped, keys["conjugate"])
    total = root + conjugate
    q = jnp.asarray(self.output_q, jnp.uint32)
    return jnp.where(total >= q, total - q, total)
