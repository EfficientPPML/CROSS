"""CROSS Polynomial interface for the full real-slot SHIP bootstrap."""

import copy

import bootstrapping as cross_bootstrapping
import ckks_ctx
import finite_field
import jax
import jax.numpy as jnp
import numpy as np
import polynomial

from . import _host_types as types
from . import key_gen, ntt_cpu, random
from .kernel import _BATSwitch, _SHIPKernel
from .plan import SHIPConfig


class _SharedRotations:
  """Share immutable controls within this bridge, without global patches."""

  def __init__(self, accessor):
    self.accessor = accessor
    self.templates = {}
    self._instances = {}

  def __getitem__(self, index):
    if index not in self._instances:
      level, rotation = index
      if level not in self.templates:
        result = self.accessor[index]
        self.templates[level] = result
      else:
        result = copy.copy(self.templates[level])
        result.rot_index = rotation
        result._herot = copy.copy(result._herot)
        a, b, mapping = result.cache.get_rot_key(rotation, level)
        count = result._herot.numPartQl
        result._herot.setup_rotation(a[:count], b[:count], mapping)
      self._instances[index] = result
    return self._instances[index]

  def clear(self):
    self._instances.clear()
    self.templates.clear()


class _InputTransform(cross_bootstrapping.Bootstrap):
  """Reuse CROSS's hoisted BSGS with SHIP's transform scale."""

  def __init__(self, ctx, plaintext_scale):
    super().__init__(ctx)
    self.plaintext_scale = plaintext_scale

  def _openfhe_scale_at_level(self, level):
    return self.plaintext_scale

  def _encode_diagonal(self, diag, moduli, num_q, scale=None):
    # Preserve arrays throughout offline encoding at large ring sizes.
    slots = np.asarray(diag, dtype=np.complex128)
    scale = self.plaintext_scale if scale is None else scale
    codec = ckks_ctx._get_or_make_encode_ctx(
      list(moduli),
      self._degree,
      scale,
      31,
      self.ctx.parameters.get("max_bits_value", (1 << 63) - (1 << 9) - 1),
    )
    encoded = ckks_ctx._fast_encode(
      slots, codec, scale=scale, noise_scale_degree=1, allow_complex=True
    )
    return jnp.asarray(encoded, dtype=jnp.uint32).reshape(
      *self._cache.degree_layout, num_q
    )


class _SHIPBootstrap:
  """Context-owned SHIP engine: control_gen → setup_key → bootstrap.

  Ciphertexts are native CROSS Polynomial objects at boundaries and canonical
  rank-five arrays inside compiled regions. Only trusted offline key builders
  use the private host key records.
  """

  def __init__(self, ctx):
    self.ctx = ctx
    self.plan = None
    self._is_setup = False

  def control_gen(self, config):
    self.plan = config.plan(self.ctx)
    cache = getattr(self.ctx, "_param_cache", None)
    if cache is None:
      raise ValueError("Initialize the CROSS context before SHIP configuration")
    if not isinstance(cache.get_sliced_ntt_q(1).ff_ctx, finite_field.BarrettContext):
      raise TypeError("SHIP currently supports Barrett contexts only")
    if cache.batch != 1:
      raise ValueError("SHIP currently supports one ciphertext per request")
    self.input_scale = self.plan.input_scale
    self.output_scale = self.plan.output_scale
    self.output_level = self.plan.expected_output_level
    self.layout = tuple(cache.degree_layout)

  def setup_key(self, random_source=None):
    if self.plan is None:
      raise RuntimeError("Configure SHIP before setup")
    if self._is_setup:
      return
    ctx = self.ctx
    if ctx.secret_key is None or ctx.public_key is None:
      raise ValueError("SHIP setup requires trusted secret/public key material")
    q, p = list(ctx.q_towers), list(ctx.p_towers)
    n = ctx.degree
    config = self.plan.config
    r, c = self.layout
    rng = random_source or random.CrossRandomSource()
    sparse = np.zeros(n, dtype=np.int64)
    support = rng.rng.sample(range(n), config.weight)
    sparse[support] = 1
    sparse[support[: config.weight // 2]] = -1
    sk0 = types.SecretKey(
      ntt_cpu.ntt_negacyclic_poly((sparse[:, None] % q[0]).astype(np.uint64), q[:1]),
      np.asarray(q[:1], np.uint32),
    )
    dense = types.SecretKey(
      np.asarray(ctx.secret_key, np.uint64).T, np.asarray(q, np.uint32)
    )
    self.kernel = _SHIPKernel(
      q, p, n, config.weight, config.theta, config.gamma, self.layout
    )
    self.keys = self.kernel.setup_keys(sparse, dense, rng)
    self.switch = _BATSwitch(q[:1], p[:2], r, c)
    src = types.SecretKey(dense.data[:, :1], dense.moduli[:1])
    evk = key_gen.gen_key_switching_key(src, sk0, q[:1], p[:2], 1, rng)
    one = key_gen.gen_key1(sk0, q[:1], p[:2], rng)
    self.switch_key = self.switch.format_key(jnp.stack([evk.b[0], evk.a[0]]), one.data)
    params = dict(
      ctx.parameters,
      q_towers=q[:2],
      p_towers=p[:3],
      scaling_factor=self.input_scale,
      output_scale=self.input_scale,
      public_key=[list(v[:2]) for v in ctx.public_key],
      secret_key=[list(v) for v in ctx.secret_key[:2]],
      composite_degree=1,
      noise_scale_degree=1,
    )
    self.input_context = ckks_ctx.CKKSContext(params)
    self.input_context.program_initialization(
      total_rotation_indices=[], dnum=2, r=r, c=c, batch=1
    )
    self.input_context._he_rot = _SharedRotations(self.input_context.he_rot)
    self.transform = _InputTransform(
      self.input_context, q[1] / config.input_scale_factor
    )
    self.transform._all_rot_indices = list(self.plan.input_rotation_indices)
    self.transform.setup_key()
    rows = np.arange(n // 2, dtype=np.int64)
    roots = np.asarray([pow(5, i, 2 * n) for i in range(n // 2)], np.int64)
    self.diagonals = {
      k: np.exp(2j * np.pi * ((roots * ((rows + k) % (n // 2))) % (2 * n)) / (2 * n))
      for k in range(n // 2)
    }
    self.output_template = polynomial.Polynomial(
      {
        "batch": 1,
        "num_elements": 2,
        "degree": n,
        "precision": 32,
        "num_moduli": self.output_level + 1,
        "degree_layout": self.layout,
      },
      {
        "moduli": q[: self.output_level + 1],
        "ntt_ctx": ctx._param_cache.get_sliced_ntt_q(self.output_level),
      },
    )
    # Compile with evaluation keys as runtime operands. The compiler sees
    # CROSS controls, not a custom serialization of the evaluator object.
    zero = jnp.zeros((1, 2, r, c, 1), jnp.uint32)
    self._switch = (
      jax.jit(self.switch.apply_array).lower(zero, self.switch_key).compile()
    )
    self._half = jax.jit(self.kernel.apply_array).lower(zero, self.keys).compile()
    # Materialize every transform key, plaintext and executable offline.
    warm = polynomial.Polynomial(
      {
        "batch": 1,
        "num_elements": 2,
        "degree": n,
        "precision": 32,
        "num_moduli": 2,
        "degree_layout": self.layout,
      },
      {"moduli": q[:2], "ntt_ctx": self.input_context._param_cache.get_sliced_ntt_q(1)},
    )
    warm._ckks_scale = self.input_scale
    warm._ckks_nsd = 1
    transformed = self.transform._eval_linear_transform_hoisted(warm, self.diagonals, 1)
    transformed = self.transform._he_op_rescale(1, 0, transformed)
    jax.block_until_ready(transformed.polynomial)
    self._is_setup = True

  def bootstrap(self, ciphertext):
    if not self._is_setup:
      raise RuntimeError("Call setup before SHIP bootstrap")
    finite_field.check_ct_operand(
      finite_field.BarrettContext,
      2,
      ciphertext,
      "SHIP bootstrap",
      batch=1,
      num_elements=2,
      degree_layout=self.layout,
      moduli=self.ctx.q_towers[:2],
    )
    self.plan.validate_input(
      1, getattr(ciphertext, "_ckks_scale", None), getattr(ciphertext, "_ckks_nsd", 1)
    )
    transformed = self.transform._eval_linear_transform_hoisted(
      ciphertext, self.diagonals, 1
    )
    transformed = self.transform._he_op_rescale(1, 0, transformed)
    switched = self._switch(transformed.polynomial, self.switch_key)
    out = self._half(switched, self.keys)
    result = self.output_template._clone_with_payload(out)
    result._ckks_scale = self.output_scale
    result._ckks_nsd = 1
    return result


class SHIPBootstrap:
  """Compatibility convenience wrapper; evaluation is owned by ctx.he_bootstrap."""

  def __init__(
    self,
    ctx,
    *,
    weight=31,
    theta=9,
    gamma=8.0,
    input_scale_factor=16.0,
    random_source=None,
  ):
    config = SHIPConfig(weight, theta, gamma, input_scale_factor)
    self.ctx = ctx
    self.facade = ctx.he_bootstrap.configure(scheme="ship", ship_config=config)
    self.facade.setup(random_source=random_source)
    self.input_scale = self.facade._engine.input_scale
    self.output_scale = self.facade._engine.output_scale
    self.output_level = self.facade._engine.output_level
    self.gamma = gamma

  def encrypt(self, values):
    if np.iscomplexobj(values) and np.any(np.imag(values)):
      raise ValueError("SHIP supports real inputs only")
    return self.ctx.encrypt(
      self.ctx.encode_at_level(list(np.real(values)), 1, scale=self.input_scale)
    )

  def bootstrap(self, ciphertext):
    return self.facade.bootstrap(ciphertext)
