"""Exact arithmetic and noisy encrypted regressions for the CROSS SHIP port."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ship import _host_types as types
from ship import encode, encrypt, key_gen, ntt_cpu, random
from ship import math as ship_math
from ship.kernel import _ntt, _rescale, _SHIPKernel


def primes(n, q_count=6, p_count=7):
  # Search downward to keep the same scale/noise regime as the N=16384 run.
  used = set()

  def take(bits, count):
    candidate = ((1 << bits) - 2) // (2 * n) * (2 * n) + 1
    result = []
    while len(result) < count:
      if candidate not in used and ship_math.is_prime(candidate):
        used.add(candidate)
        result.append(candidate)
      candidate -= 2 * n
    return result

  return take(31, 1) + take(30, q_count - 1), take(30, p_count)


def secret(coefficients, moduli):
  coeffs = np.column_stack([coefficients % q for q in moduli]).astype(np.uint64)
  return types.SecretKey(
    ntt_cpu.ntt_negacyclic_poly(coeffs, moduli), np.array(moduli, dtype=np.uint32)
  )


@pytest.mark.parametrize("n", [16, 32, 256])
def test_cross_ntt_matches_host(n):
  q, _ = primes(n, 2, 1)
  r = 1 << ((n.bit_length() - 1) // 2)
  k = _ntt(q, r, n // r)
  values = np.random.default_rng(1).integers(0, q, (n, len(q)), dtype=np.uint32)
  expected = ntt_cpu.ntt_negacyclic_poly(values, q)
  actual = jax.jit(k.ntt)(jnp.array(values.reshape(1, r, n // r, -1)))
  np.testing.assert_array_equal(np.asarray(actual).reshape(n, -1), expected)
  back = jax.jit(k.intt)(actual)
  np.testing.assert_array_equal(np.asarray(back).reshape(n, -1), values)


@pytest.mark.parametrize("drops", [1, 2, 3])
def test_cross_rescale_exact_centered_integer_reference(drops):
  n = 16
  q, _ = primes(n, 5, 1)
  rng = np.random.default_rng(2)
  values = rng.integers(0, q, (2, n, len(q)), dtype=np.uint32)
  expected = values.astype(object)
  for last in range(len(q) - 1, len(q) - drops - 1, -1):
    centered = expected[..., last]
    centered = np.where(centered > q[last] // 2, centered - q[last], centered)
    expected = np.stack(
      [
        ((expected[..., j] - centered) * pow(q[last], -1, q[j])) % q[j]
        for j in range(last)
      ],
      axis=-1,
    )
  data = np.stack([ntt_cpu.ntt_negacyclic_poly(v, q) for v in values]).astype(np.uint32)
  k = _rescale(q, drops, 4, 4)
  out = jax.jit(k._rescale_array)(jnp.asarray(data).reshape(1, 2, 4, 4, len(q)))
  actual = np.stack(
    [
      ntt_cpu.intt_negacyclic_poly(v, q[:-drops])
      for v in np.asarray(out).reshape(2, n, -1)
    ]
  )
  np.testing.assert_array_equal(actual, np.asarray(expected, dtype=np.uint64))


@pytest.mark.parametrize("seed,weight", [(7, 5), (19, 4)])
def test_noisy_half_bootstrap_independent_dense_key(seed, weight):
  n = 32
  q, p = primes(n)
  rng = random.TestRandomSource(seed)
  s = np.zeros(n, dtype=np.int64)
  support = rng.rng.sample(range(n), weight)
  s[support] = 1
  s[support[: weight // 2]] = -1
  sk0 = secret(s, q[:1])
  dense = secret(np.array([rng.rng.choice([-1, 0, 1]) for _ in range(n)]), q)
  kernel = _SHIPKernel(q, p, n, weight, 4, 8.0, (4, 8))
  keys = kernel.setup_keys(s, dense, rng)
  assert not hasattr(kernel, "nonzero_idx")
  mu = np.random.default_rng(seed).uniform(-0.125, 0.125, n // 2)
  coefficients = np.pad(np.rint(mu * q[0] / 8).astype(np.int64), (0, n // 2))
  data = ntt_cpu.ntt_negacyclic_poly(
    (coefficients[:, None] % q[0]).astype(np.uint64), q[:1]
  )
  pt = types.Plaintext(
    jnp.array(data, dtype=jnp.uint32), jnp.array(q[:1], dtype=jnp.uint32)
  )
  pk, _ = key_gen.keygen(n, q[:1], random_source=rng, sk=sk0)
  ct = encrypt.Encrypt(pk).encrypt(pt, random_source=rng)
  fn = jax.jit(kernel.apply_array)
  payload = jax.block_until_ready(fn(ct.data.reshape(1, 2, 4, 8, 1), keys))
  out = types.Ciphertext(
    payload.reshape(2, n, -1), jnp.asarray(kernel.output_q, jnp.uint32)
  )
  actual = np.asarray(
    encode.Decode(float(q[-1]), n // 2).decode(encrypt.Decrypt(dense).decrypt(out))
  )
  assert out.data.dtype == jnp.uint32
  np.testing.assert_allclose(actual, mu, atol=2**-10, rtol=0)


@pytest.mark.parametrize("n,weight,theta", [(32, 5, 4), (128, 31, 9)])
def test_full_bootstrap_cross_polynomial_api(n, weight, theta, monkeypatch):
  import ckks_ctx
  import key_gen as cross_keys
  from ship import SHIPBootstrap

  q, p = primes(
    n, q_count=(weight - 1).bit_length() + 3, p_count=(weight - 1).bit_length() + 4
  )
  keys = cross_keys.gen_pke_pair(q, p, n)
  ctx = ckks_ctx.CKKSContext(
    dict(
      degree=n,
      num_slots=n // 2,
      scaling_factor=q[-1],
      output_scale=q[-1],
      q_towers=q,
      p_towers=p,
      p=30,
      CKKS_M_FACTOR=1,
      max_bits_in_word=31,
      noise_scale_degree=1,
      composite_degree=1,
      **keys,
    )
  )
  r = 1 << ((n.bit_length() - 1) // 2)
  ctx.program_initialization(total_rotation_indices=[], dnum=2, r=r, c=n // r, batch=1)
  bs = SHIPBootstrap(
    ctx, weight=weight, theta=theta, random_source=random.TestRandomSource(29)
  )

  with pytest.raises(ValueError, match="real inputs"):
    bs.encrypt(np.full(n // 2, 0.1j))

  # Setup must finish encoding and key generation even before the first real input.
  def no_offline_work(*args, **kwargs):
    raise AssertionError("offline work attempted during online bootstrap")

  engine = ctx.he_bootstrap._engine
  monkeypatch.setattr(engine.transform, "_encode_diagonal", no_offline_work)
  monkeypatch.setattr(key_gen, "gen_hybrid_key", no_offline_work)
  monkeypatch.setattr(cross_keys, "gen_rotation_key", no_offline_work)
  assert ctx.he_bootstrap.setup() is ctx.he_bootstrap
  with pytest.raises(ValueError, match="Meta-BTS"):
    ctx.he_bootstrap.meta_bootstrap(None, 10)
  mu = np.random.default_rng(31).uniform(-0.125, 0.125, n // 2)
  ct = bs.encrypt(mu)
  before = np.array(ct.polynomial)
  out = bs.bootstrap(ct)
  jax.block_until_ready(out.polynomial)
  np.testing.assert_array_equal(ct.polynomial, before)
  assert out.num_moduli == 2
  assert out.polynomial.dtype == jnp.uint32
  actual = ctx.decrypt_slots(out)
  np.testing.assert_allclose(actual, mu, atol=2**-10, rtol=0)
  # A second input exercises reused kernels/caches, not just a warm replay.
  mu2 = np.linspace(-0.125, 0.125, n // 2)
  native_input = ctx.encrypt(ctx.encode_at_level(list(mu2), 1, scale=bs.input_scale))
  out2 = bs.bootstrap(native_input)
  np.testing.assert_allclose(ctx.decrypt_slots(out2), mu2, atol=2**-10, rtol=0)
  # Independently decode unprojected complex slots, including imaginary error.
  plain = ctx.decrypt(out2)
  raw = types.Plaintext(
    plain.polynomial.reshape(n, out2.num_moduli),
    jnp.asarray(out2.moduli, dtype=jnp.uint32),
  )
  unprojected = np.asarray(encode.Decode(bs.output_scale, n // 2).decode(raw))
  np.testing.assert_allclose(unprojected, mu2, atol=2**-10, rtol=0)
  bad = ct._clone_with_payload(ct.polynomial)
  bad._ckks_scale = bs.input_scale * 2
  with pytest.raises(ValueError, match="scale"):
    bs.bootstrap(bad)


def test_mapping_ship_bootstrap_native_lifecycle(monkeypatch):
  from types import SimpleNamespace

  import ckks_ctx
  import key_gen as cross_keys
  import mapping
  import packing
  from ship import SHIPConfig

  n = 32
  q, p = primes(n)
  params = dict(
    degree=n,
    num_slots=n // 2,
    scaling_factor=q[-1],
    output_scale=q[-1],
    q_towers=q,
    p_towers=p,
    p=30,
    CKKS_M_FACTOR=1,
    max_bits_in_word=31,
    noise_scale_degree=1,
    composite_degree=1,
    **cross_keys.gen_pke_pair(q, p, n),
  )
  config = SHIPConfig(weight=5, theta=4)
  ctx = ckks_ctx.CKKSContext(params)
  ctx.program_initialization(total_rotation_indices=[], dnum=2, r=4, c=8, batch=1)
  plan = config.plan(ctx)
  with pytest.raises(RuntimeError, match="Configure"):
    ctx.he_bootstrap.setup()
  with pytest.raises(ValueError, match="level 1"):
    ctx.he_bootstrap.configure(scheme="ship", ship_config=config, input_level=0)
  with pytest.raises(ValueError, match="scale"):
    ctx.he_bootstrap.configure(scheme="ship", ship_config=config, input_scale=1.0)
  with monkeypatch.context() as patch:

    def forbidden(*args, **kwargs):
      raise AssertionError("planning attempted key generation")

    patch.setattr(key_gen, "gen_hybrid_key", forbidden)
    ctx.he_bootstrap.configure(scheme="ship", ship_config=config)
    assert ctx.he_bootstrap.required_rotation_indices == ()
    with pytest.raises(RuntimeError, match="setup"):
      ctx.he_bootstrap.bootstrap(None)
  program = packing.Packing(
    operations=(
      ("consume", "level_reduce", ("input",), len(q) - 2),
      ("refresh", "bootstrap", ("consume",), config),
    ),
    ring_config=SimpleNamespace(degree=n, num_slots=n // 2, dnum=2),
    input_shape=(n // 2,),
    input_packing="slots",
    logical_input_shape=(n // 2,),
    logical_output_shape=(n // 2,),
    input_coordinate_map=tuple(range(n // 2)),
    output_coordinate_map=tuple(range(n // 2)),
    layer_shapes=(),
    output="refresh",
    num_slots=n // 2,
    depth=0,
  )
  compiled = mapping.Mapping(
    program, packing.test_only_parameters(params), input_scale=plan.input_scale
  )
  assert compiled.regions == ((True, ("consume",)), (False, ("refresh",)))
  assert compiled.context.he_bootstrap._scheme == "ship"
  assert compiled.output_spec.level == plan.expected_output_level
  for values in (
    np.linspace(-0.125, 0.125, n // 2),
    np.random.default_rng(61).uniform(-0.125, 0.125, n // 2),
  ):
    out = compiled.execute(compiled.encrypt_input(values))
    assert out.polynomial.dtype == jnp.uint32
    assert out.polynomial.ndim == 5
    np.testing.assert_allclose(
      compiled.decrypt_output(out), values, atol=2**-10, rtol=0
    )


@pytest.mark.contract
@pytest.mark.parametrize(
  "kwargs",
  [
    {"weight": 0},
    {"weight": True},
    {"theta": -1},
    {"theta": 1.5},
    {"gamma": True},
    {"gamma": "8"},
    {"gamma": float("nan")},
    {"input_scale_factor": 0},
    {"input_scale_factor": float("inf")},
  ],
)
def test_invalid_ship_configuration(kwargs):
  from ship import SHIPConfig

  with pytest.raises(ValueError):
    SHIPConfig(**kwargs)


@pytest.mark.contract
@pytest.mark.parametrize(
  "case",
  [
    "composite_degree",
    "non_prime",
    "duplicate",
    "too_wide",
    "scale_overflow",
  ],
)
def test_ship_plan_rejects_incompatible_context(case):
  from types import SimpleNamespace

  from ship import SHIPConfig

  q, p = primes(32)
  ctx = SimpleNamespace(
    degree=32, num_slots=16, q_towers=q, p_towers=p, composite_degree=1
  )
  config = SHIPConfig(weight=5, theta=4)
  if case == "composite_degree":
    ctx.composite_degree = 2
  elif case == "non_prime":
    ctx.q_towers[0] = 65  # Congruent to 1 mod 2N, but not a prime.
  elif case == "duplicate":
    ctx.p_towers[0] = ctx.q_towers[0]
  elif case == "too_wide":
    ctx.q_towers[0] = 2**32 + 1
  else:
    config = SHIPConfig(weight=5, theta=4, gamma=1e-300, input_scale_factor=1e300)
  with pytest.raises(ValueError):
    config.plan(ctx)


@pytest.mark.contract
@pytest.mark.parametrize("first", ["ship", "jaxite_word.ship"])
def test_ship_configuration_identity_across_import_styles(first):
  import os
  import subprocess
  import sys
  from pathlib import Path

  source = f"""
from {first} import SHIPConfig
from ship.plan import SHIPConfig as Flat
from jaxite_word.ship.plan import SHIPConfig as Qualified
from jaxite_word import SHIPConfig as Root
assert SHIPConfig is Flat is Qualified is Root
"""
  root = Path(__file__).resolve().parent.parent
  env = dict(
    os.environ, PYTHONPATH=os.pathsep.join((str(root), str(root / "jaxite_word")))
  )
  subprocess.run(
    [sys.executable, "-c", source], env=env, check=True, capture_output=True, text=True
  )
