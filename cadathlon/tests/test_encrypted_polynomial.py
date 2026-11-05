"""Public approximation contracts and opt-in real 128-bit CKKS evaluation.

Run the actual encrypted integration check with:
  CROSS_TEST_SECURE_POLYNOMIAL=1 JAX_PLATFORMS=cpu python -m pytest \\
    cadathlon/tests/test_encrypted_polynomial.py -q
The secure test requires several minutes of cold JAX compilation. CPU success
is not TPU validation and does not make this a complete transformer reference.
"""
import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from encrypted_polynomial import EncryptedPolynomialEvaluator, PolynomialApproximation, nonlinear_approximation


def test_inverse_sqrt_domain_and_grid_accuracy():
    approximation = nonlinear_approximation("inverse_sqrt", (.5, 1.5), 8)
    values = np.linspace(.5, 1.5, 10001)
    relative_error = np.max(np.abs(approximation.evaluate_public(values) * np.sqrt(values) - 1))
    assert relative_error < 3.4e-6
    assert approximation.depth == 8
    for bad in ([.49], [1.51], [float('nan')]):
        with pytest.raises(ValueError, match="outside"):
            approximation.evaluate_public(bad)


@pytest.mark.parametrize('name,domain', [('inverse_sqrt', (0, 1)), ('reciprocal', (-1, 1)), ('gelu', (2, 1))])
def test_invalid_domains_are_rejected(name, domain):
    with pytest.raises(ValueError):
        nonlinear_approximation(name, domain, 8)


@pytest.mark.parametrize('name,expected,error', [
    ('gelu', lambda x: .5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2))), .0001),
    ('silu', lambda x: x / (1 + np.exp(-x)), .0001),
    ('tanh', np.tanh, .001),
])
def test_nonlinear_public_approximation(name, expected, error):
    x = np.linspace(-2, 2, 10001)
    approximation = nonlinear_approximation(name, (-2, 2), 12)
    assert np.max(np.abs(approximation.evaluate_public(x) - expected(x))) < error


def test_input_power_basis_is_limited_to_bounded_centered_quadratics():
    from dataclasses import replace
    evaluator = object.__new__(EncryptedPolynomialEvaluator)
    evaluator.strategy = 'power'
    silu = nonlinear_approximation('silu', (-2, 2), 2)
    direct = replace(silu, input_power_basis=True)
    assert evaluator.polynomial_depth(silu) == 3
    assert evaluator.polynomial_depth(direct) == 2
    grid = np.linspace(-2, 2, 1001)
    np.testing.assert_array_equal(direct.evaluate_public(grid), silu.evaluate_public(grid))
    for invalid in (nonlinear_approximation('inverse_sqrt', (.005, .35), 2),
                    nonlinear_approximation('silu', (-3, 3), 2),
                    nonlinear_approximation('silu', (-2, 2), 4)):
        with pytest.raises(ValueError, match='centered quadratic'):
            replace(invalid, input_power_basis=True)


def test_refresh_scheduler_preserves_capacity_and_deduplicates_square_operand():
    # This is scheduling-only, independent of the opt-in real bootstrap proof.
    from types import SimpleNamespace
    refreshed, checkpoints = [], []
    low, enough = SimpleNamespace(level=1), SimpleNamespace(level=9)
    evaluator = object.__new__(EncryptedPolynomialEvaluator)
    evaluator._level = lambda value: value.level
    evaluator.checkpoint = lambda value, label: checkpoints.append(label)
    def refresh(value):
        refreshed.append(value)
        return SimpleNamespace(level=8)
    evaluator.refresher = SimpleNamespace(output_level=8, refresh=refresh)
    left, right, retained = evaluator._ensure_capacity((low, low, enough), 1)
    assert refreshed == [low]
    assert left is right and left.level == 8 and retained is enough
    assert checkpoints == ['homomorphic refresh starting', 'homomorphic refresh complete']
    # A polynomial must retain one level for a future genuine bootstrap.
    with pytest.raises(ValueError, match='operation needs 9'):
        evaluator._ensure_capacity((low,), 8)
    assert evaluator._ensure_capacity((enough,), 8) == (enough,)
    evaluator.refresher = None
    assert evaluator._ensure_capacity((low,), 20) == (low,)


def test_refresh_scheduler_rejects_short_output():
    from types import SimpleNamespace
    evaluator = object.__new__(EncryptedPolynomialEvaluator)
    evaluator._level = lambda value: value.level
    evaluator.checkpoint = lambda value, label: value
    evaluator.refresher = SimpleNamespace(output_level=8, refresh=lambda value: SimpleNamespace(level=1))
    with pytest.raises(ValueError, match='insufficient capacity'):
        evaluator._ensure_capacity((SimpleNamespace(level=1),), 1)


@pytest.mark.slow
@pytest.mark.secure_profile
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('CROSS_TEST_SECURE_POLYNOMIAL') != '1', reason='opt-in secure CKKS compilation')
def test_secure_inverse_sqrt_and_rmsnorm_ciphertexts():
    import jax
    from jaxite_word import CKKSContext, he_params, key_gen, util
    from jaxite_word.composite_prime_gen import composite_prime_gen
    from support import require_secure_execution

    jax.config.update('jax_enable_x64', True)
    degree, depth = 32768, 12
    q = composite_prime_gen(2, 2 * (depth + 1), 61, 50, 2 * degree, 31)
    p = util.generate_p_towers(q, 4, degree, aux_bits=30)
    assert he_params.qp_is_secure(q, p, degree, he_params.SecurityTarget(128, 'classical'))
    keys = key_gen.gen_pke_pair(q, [], degree)
    evaluation = key_gen.gen_evaluation_key(keys['secret_key'], q=q, P=p, dnum=4,
                                            noise_std=he_params.DEFAULT_SIGMA, noise_scale=1)
    scale = float(q[-2]) * q[-1]
    params = dict(degree=degree, num_slots=degree // 2, scaling_factor=scale, output_scale=scale,
                  q_towers=q, p_towers=p, composite_degree=2, degree_layout=(128, 256), p=50,
                  CKKS_M_FACTOR=1, max_bits_in_word=61, noise_scale_degree=1)
    client = CKKSContext(dict(params, **keys))
    # A physically separate context: no secret or public encryption key reaches
    # the evaluator, which needs only its public relinearization key.
    server = CKKSContext(dict(params, evaluation_key=[
        jax.numpy.asarray(evaluation[name], dtype='uint32').transpose(0, 2, 1) for name in ('a', 'b')]))
    server.program_initialization([], dnum=4, r=128, c=256, batch=1)
    assert server.secret_key is None and server._param_cache.secret_key is None
    evaluator = EncryptedPolynomialEvaluator(server)
    approximation = nonlinear_approximation('inverse_sqrt', (.5, 1.5), 8)
    values = np.tile([.5, .6, .7, .8, 1., 1.2, 1.4, 1.5], degree // 16)
    encrypted = client.encrypt_slots(values)
    result = evaluator.evaluate(encrypted, approximation)
    require_secure_execution(evaluator, result)
    np.testing.assert_allclose(client.decrypt_slots(result), 1 / np.sqrt(values), rtol=0, atol=1e-5)

    features = [np.tile([.5, .7, .8, 1., 1.2, 1.3, .8, 1.], degree // 16),
                np.tile([1., 1.1, 1., 1., 1., .5, 1.1, .9], degree // 16)]
    results = evaluator.normalize([client.encrypt_slots(value) for value in features], approximation, epsilon=1e-6)
    require_secure_execution(evaluator, results)
    variance = np.mean(np.asarray(features) ** 2, axis=0) + 1e-6
    for result, feature in zip(results, features):
        np.testing.assert_allclose(client.decrypt_slots(result), feature / np.sqrt(variance), rtol=0, atol=1e-5)
    # Also exercise learned-network nonlinearities through ciphertexts, rather
    # than relying on the public polynomial interpolation tests above.
    values = np.tile(np.linspace(-2, 2, 8), degree // 16)
    gelu = nonlinear_approximation('gelu', (-2, 2), 12)
    actual = client.decrypt_slots(evaluator.evaluate(client.encrypt_slots(values), gelu))
    expected = .5 * values * (1 + np.vectorize(math.erf)(values / np.sqrt(2)))
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)

    centered_features = [np.full(degree // 2, value) for value in (-1., 0., 1.)]
    layer_outputs = evaluator.normalize([client.encrypt_slots(value) for value in centered_features],
                                        approximation, epsilon=1e-6, center=True)
    for result, feature in zip(layer_outputs, centered_features):
        np.testing.assert_allclose(client.decrypt_slots(result), feature / np.sqrt(2 / 3 + 1e-6),
                                   rtol=0, atol=1e-5)
    depleted = server.he_level_reduce[server.max_level, 1].level_reduce(encrypted)
    with pytest.raises(ValueError, match='needs 8 levels'):
        evaluator.evaluate(depleted, approximation)


@pytest.mark.integration
@pytest.mark.secure_profile
@pytest.mark.skipif(os.environ.get('CROSS_TEST_SECURE_POLYNOMIAL') != '1', reason='opt-in real-context codec check')
def test_uniform_constants_match_generic_encoder_bit_for_bit():
    import jax
    from jaxite_word import CKKSContext, key_gen, util, he_params
    from jaxite_word.composite_prime_gen import composite_prime_gen
    jax.config.update('jax_enable_x64', True)
    degree = 8192
    q = composite_prime_gen(2, 4, 61, 56, 2 * degree, 31)
    p = util.generate_p_towers(q, 2, degree, aux_bits=30)
    assert he_params.qp_is_secure(q, p, degree, he_params.SecurityTarget(128, 'classical'))
    keys = key_gen.gen_pke_pair(q, [], degree)
    evaluation = key_gen.gen_evaluation_key(keys['secret_key'], q=q, P=p, dnum=2,
        noise_std=he_params.DEFAULT_SIGMA, noise_scale=1)
    scale = float(q[-2]) * q[-1]
    context = CKKSContext(dict(degree=degree, num_slots=degree // 2,
        scaling_factor=scale, output_scale=scale, q_towers=q, p_towers=p,
        composite_degree=2, degree_layout=(64, 128), p=56, CKKS_M_FACTOR=1,
        max_bits_in_word=61, noise_scale_degree=1, evaluation_key=[
            jax.numpy.asarray(evaluation[name], dtype='uint32').transpose(0, 2, 1) for name in ('a', 'b')]))
    context.program_initialization([], dnum=2, r=64, c=128, batch=1)
    evaluator = EncryptedPolynomialEvaluator(context)
    for level in (0, 1):
        for encode_scale in (scale, scale * scale):
            for scalar in (0., .2, -3., 167., -167., 1.5 / encode_scale, -1.5 / encode_scale):
                slots = np.full(degree // 2, scalar)
                direct = evaluator._encode_public(slots, level, encode_scale)
                generic = context.encode_at_level(slots, level, scale=encode_scale)
                np.testing.assert_array_equal(direct.polynomial, generic.polynomial)
    assert len(evaluator._constant_cache) <= 8


@pytest.mark.integration
@pytest.mark.secure_profile
@pytest.mark.skipif(os.environ.get('CROSS_TEST_SECURE_POLYNOMIAL') != '1', reason='opt-in secure balanced-power circuit')
def test_balanced_power_inverse_sqrt_with_bounded_caches():
    """Exercise Qwen's ill-conditioned public polynomial through real CKKS."""
    import jax
    from jaxite_word import CKKSContext, key_gen, util, he_params, polynomial
    from jaxite_word.composite_prime_gen import composite_prime_gen
    jax.config.update('jax_enable_x64', True)
    degree, depth, dnum = 16384, 6, 8
    q = composite_prime_gen(2, 2 * (depth + 1), 61, 44, 2 * degree, 31)
    p = util.generate_p_towers(q, dnum, degree, aux_bits=30)
    assert he_params.qp_is_secure(q, p, degree, he_params.SecurityTarget(128, 'classical'))
    keys = key_gen.gen_pke_pair(q, [], degree)
    evaluation = key_gen.gen_evaluation_key(keys['secret_key'], q=q, P=p, dnum=dnum,
        noise_std=he_params.DEFAULT_SIGMA, noise_scale=1)
    scale = float(q[-2]) * q[-1]
    params = dict(degree=degree, num_slots=degree // 2, scaling_factor=scale,
        output_scale=scale, q_towers=q, p_towers=p, composite_degree=2,
        degree_layout=(128, 128), p=44, CKKS_M_FACTOR=1, max_bits_in_word=61, noise_scale_degree=1)
    client = CKKSContext(dict(params, **keys))
    context = CKKSContext(dict(params, evaluation_key=[
        jax.numpy.asarray(evaluation[name], dtype='uint32').transpose(0, 2, 1) for name in ('a', 'b')]))
    context.program_initialization([], dnum=dnum, r=128, c=128, batch=1)
    evaluator = EncryptedPolynomialEvaluator(context, strategy='power', low_memory=True,
                                             compiled_multiply=True, compiled_linear=True)
    approximation = nonlinear_approximation('inverse_sqrt', (.005, .35), 12)
    values = np.tile([.005, .008, .01, .03, .05, .1, .2, .35], degree // 16)
    encrypted = client.encrypt_slots(values)
    other = client.encrypt_slots(values * .7, scale=scale * 1.125)
    direct = context.he_mul[depth - 1].mul(encrypted, other)
    compiled = evaluator.multiply(encrypted, other)
    np.testing.assert_array_equal(compiled.polynomial, direct.polynomial)
    assert compiled._ckks_scale == direct._ckks_scale
    assert compiled._ckks_nsd == getattr(direct, '_ckks_nsd', 1) == 1
    assert compiled.moduli == direct.moduli
    evaluator.checkpoint(compiled, 'release compiled controls')
    assert not evaluator._multiplication_bindings
    compilation_count = evaluator._multiplication_compiler.compile_count
    repeated = evaluator.multiply(encrypted, other)
    np.testing.assert_array_equal(repeated.polynomial, direct.polynomial)
    assert evaluator._multiplication_compiler.compile_count == compilation_count
    del other, direct, compiled, repeated
    result = evaluator.evaluate(encrypted, approximation)
    assert evaluator._level(result) == 0
    assert context.secret_key is None and context._param_cache.secret_key is None
    # Compare to the explicit approximation separately from its public analytic
    # approximation error. These endpoints expose the power-form cancellation.
    np.testing.assert_allclose(client.decrypt_slots(result), approximation.evaluate_public(values),
                               rtol=0, atol=1e-4)
    payload = np.asarray(result.polynomial).copy()
    evaluator.checkpoint([result], 'test completed')
    assert not context._param_cache.level_params
    assert not polynomial._POLYNOMIAL_REGISTRY
    np.testing.assert_array_equal(result.polynomial, payload)
    # Resolve the actual primitive module, not an independent package alias.
    import importlib
    import weakref
    ntt_module = importlib.import_module(type(context._param_cache.ntt_q_max).__module__)
    ntt_module.get_shared_barrett_ntt_context(context.q_towers[:2],
        {'r': 128, 'c': 128, 'finite_field_context': context._param_cache.get_sliced_ff_q(0)})
    assert ntt_module._barrett_ntt_context_cache
    evaluator.checkpoint(result, 'clear actual primitive cache')
    assert not ntt_module._barrett_ntt_context_cache
    new_ntt = context._param_cache.get_sliced_ntt_q(0)
    assert new_ntt is not result.ntt_ctx
    duplicate = result._clone_with_payload(result.polynomial, ntt_ctx=new_ntt)
    original_payload = duplicate.polynomial
    metadata = (duplicate._ckks_scale, duplicate._ckks_nsd, tuple(duplicate.moduli),
                duplicate.degree_layout, duplicate.precision)
    evaluator._finish(duplicate)
    assert duplicate.ntt_ctx is result.ntt_ctx
    assert duplicate.polynomial is original_payload
    assert metadata == (duplicate._ckks_scale, duplicate._ckks_nsd, tuple(duplicate.moduli),
                        duplicate.degree_layout, duplicate.precision)
    temporary = evaluator._finish(evaluator._lower(encrypted, 1))
    retired_context = weakref.ref(temporary.ntt_ctx)
    del temporary
    evaluator.checkpoint(result, 'retire unused public level')
    assert retired_context() is None
    # A live ciphertext remains usable after global compilation caches and
    # retired pytree records have been released.
    doubled = evaluator.add(result, result)
    np.testing.assert_allclose(client.decrypt_slots(doubled), 2 * approximation.evaluate_public(values),
                               rtol=0, atol=2e-4)
