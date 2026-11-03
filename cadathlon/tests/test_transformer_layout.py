"""Numerical planning tests for ciphertext-layout algebra, not HE execution.

The NumPy test double lives only here. The runtime module has no plaintext
backend, decryption path, or fallback.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from encrypted_polynomial import EncryptedPolynomialEvaluator, nonlinear_approximation
from encrypted_transformer import EncryptedTensorOps, PairLayout, public_toy_approximations


@pytest.mark.parametrize('depth,dnum,q_count,p_count', [(38, 2, 78, 37), (50, 5, 102, 20), (67, 4, 100, 24)])
def test_exact_transformer_profiles_generate_secure_covered_rings(depth, dnum, q_count, p_count):
    from encrypted_reference import reference_ring_profile
    from jaxite_word import util, he_params
    from jaxite_word.composite_prime_gen import composite_prime_gen
    chain, selected = reference_ring_profile(depth)
    assert selected == dnum
    q = composite_prime_gen(2, 2 * (chain + 1), 61, 56, 262144, 31)
    p = util.generate_p_towers(q, dnum, 131072, aux_bits=30)
    assert (len(q), len(p)) == (q_count, p_count)
    assert he_params.qp_is_secure(q, p, 131072, he_params.SecurityTarget(128, 'classical'))
    he_params.assert_p_coverage(q, p, dnum)


def test_unsharded_reference_rejects_proven_impossible_pretrained_payloads():
    from encrypted_reference import check_device_payload_capacity
    device = SimpleNamespace(memory_stats=lambda: {'bytes_limit': 32 * 2**30})
    check_device_payload_capacity(device, degree=131072, num_q=100,
                                  input_features=24, output_features=32)
    with pytest.raises(ValueError, match='input ciphertext payload'):
        check_device_payload_capacity(device, degree=131072, num_q=100,
                                      input_features=1024, output_features=32)
    with pytest.raises(ValueError, match='output ciphertext payload'):
        check_device_payload_capacity(device, degree=131072, num_q=100,
                                      input_features=16, output_features=151936)
    # Absence of a device limit cannot establish an impossibility bound.
    check_device_payload_capacity(SimpleNamespace(memory_stats=lambda: None),
        degree=131072, num_q=100, input_features=1024, output_features=151936)


class PublicPlanningArithmetic:
    def __init__(self, slots):
        self.ctx = SimpleNamespace(num_slots=slots)

    def add(self, left, right, subtract=False):
        return left - right if subtract else left + right

    def rotate(self, value, shift):
        return np.roll(value, -shift)

    def weighted_sum(self, values, weights, bias=0.):
        return sum(value * weight for value, weight in zip(values, weights)) + bias

    def multiply(self, left, right):
        return left * right

    def evaluate(self, value, approximation, output_weights=1.):
        return approximation.evaluate_public(value) * output_weights

    def normalize(self, features, approximation, epsilon, center=False):
        values = np.asarray(features)
        if center:
            values = values - values.mean(axis=0, keepdims=True)
        variance = np.mean(values * values, axis=0) + epsilon
        inverse = approximation.evaluate_public(variance)
        return list(values * inverse)


def test_linear_refreshes_shared_depleted_features_once_for_all_output_rows():
    calls = []
    evaluator = object.__new__(EncryptedPolynomialEvaluator)
    evaluator.ctx = SimpleNamespace(num_slots=4)
    evaluator._level = lambda value: value.level
    evaluator.checkpoint = lambda value, label: value
    def refresh(value):
        calls.append(value)
        return SimpleNamespace(level=8, value=value.value)
    evaluator.refresher = SimpleNamespace(output_level=8, refresh=refresh)
    def weighted_sum(values, weights, *, bias):
        values = evaluator._ensure_capacity(values, 1)
        return sum(value.value * weight for value, weight in zip(values, weights)) + bias
    evaluator.weighted_sum = weighted_sum
    features = [SimpleNamespace(level=1, value=value) for value in (2., -3.)]
    weights = np.arange(64, dtype=float).reshape(32, 2)
    ops = EncryptedTensorOps(evaluator, PairLayout(1, 1, 4))
    np.testing.assert_allclose(ops.linear(features, weights), weights @ [2., -3.])
    assert len(calls) == 2  # two shared inputs, not 2 * 32 output rows


def test_packing_transpose_and_key_sum_with_non_power_of_two_dimensions():
    layout = PairLayout(3, 6, 512)
    values = np.random.default_rng(4).normal(size=(3, 6, 5))
    features = layout.pack_client_embeddings(values)
    np.testing.assert_array_equal(layout.unpack_client_features(features), values)
    ops = EncryptedTensorOps(PublicPlanningArithmetic(layout.slots), layout)
    source = np.arange(layout.slots, dtype=float)
    expected = source.reshape(-1, layout.width, layout.width).transpose(0, 2, 1).reshape(-1)
    np.testing.assert_allclose(ops.transpose_pairs(source), expected)
    expected_sum = np.repeat(source.reshape(-1, layout.width).sum(-1), layout.width)
    np.testing.assert_allclose(ops.sum_keys(source), expected_sum)


@pytest.mark.parametrize('causal', [False, True])
def test_grouped_attention_layout_matches_independent_dense_oracle(causal):
    layout = PairLayout(2, 6, 256)
    rng = np.random.default_rng(81)
    query, key, value = [rng.normal(0, .1, (2, 6, n)) for n in (8, 4, 4)]
    ops = EncryptedTensorOps(PublicPlanningArithmetic(layout.slots), layout)
    public_mask = np.ones((2, 6), bool)
    if not causal:
        public_mask[:, -1] = False
    mask, counts = layout.attention_mask(public_mask, causal=causal)
    outputs = ops.attention(*[layout.pack_client_embeddings(x) for x in (query, key, value)],
        heads=2, kv_heads=1, head_dim=4, mask=mask, counts=counts,
        exponential=nonlinear_approximation('exp', (-1, 1), 12),
        reciprocal=nonlinear_approximation('reciprocal', (.8, 1.2), 12))
    actual = layout.unpack_client_features(outputs)
    q = query.reshape(2, 6, 2, 4).transpose(0, 2, 1, 3)
    k = np.repeat(key[:, None], 2, axis=1)
    v = np.repeat(value[:, None], 2, axis=1)
    scores = q @ k.swapaxes(-1, -2) / 2
    valid = np.broadcast_to(public_mask[:, None, None, :], scores.shape).copy()
    if causal:
        valid &= np.tril(np.ones((6, 6), bool))
    scores = np.where(valid, scores, -np.inf)
    exp = np.exp(scores - scores.max(-1, keepdims=True))
    expected = ((exp / exp.sum(-1, keepdims=True)) @ v).transpose(0, 2, 1, 3).reshape(2, 6, 8)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_rope_preserves_qwen_half_rotation_convention():
    layout = PairLayout(1, 6, 128)
    inputs = np.random.default_rng(7).normal(size=(1, 6, 8))
    ops = EncryptedTensorOps(PublicPlanningArithmetic(layout.slots), layout)
    result = layout.unpack_client_features(ops.rope(layout.pack_client_embeddings(inputs), 4, 1000000.))
    shaped = inputs.reshape(1, 6, 2, 4)
    angles = np.arange(6)[:, None] * np.array([1., .001])[None, :]
    angles = np.concatenate([angles, angles], -1)[None, :, None]
    rotated = np.concatenate([-shaped[..., 2:], shaped[..., :2]], -1)
    expected = (shaped * np.cos(angles) + rotated * np.sin(angles)).reshape(1, 6, 8)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


def test_attention_rejects_fully_masked_rows():
    with pytest.raises(ValueError, match='masked'):
        PairLayout(1, 2, 8).attention_mask([[False, False]])


@pytest.mark.parametrize('case_number', [2, 3, 4, 5, 6])
def test_softmax_planner_observes_actual_masked_and_padded_polynomial_inputs(case_number):
    """Compare every polynomial argument, including work later masked out."""
    import importlib.util
    import json
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('masked_domain_planner', root / 'validation/plan_transformer_polynomials.py')
    planner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planner)
    case = json.loads((root / f'agent_pset/p2_bert/test/test{case_number}.in').read_text())
    weights, embeddings, public_mask, _ = planner.BERT.make_case(case)
    batch, sequence, hidden = embeddings.shape
    heads = case['config']['num_attention_heads']
    head_dim = hidden // heads
    prefix = 'bert.encoder.layer.0.attention.self.'
    query, key, value = [planner.BERT.linear(embeddings, weights, prefix + name)
                         for name in ('query', 'key', 'value')]
    slots = (1 << (batch - 1).bit_length()) * (1 << (sequence - 1).bit_length()) ** 2
    layout = PairLayout(batch, sequence, slots)

    class RecordingArithmetic(PublicPlanningArithmetic):
        def __init__(self):
            super().__init__(slots)
            self.arguments = {'exp': [], 'reciprocal': []}

        def evaluate(self, values, approximation, output_weights=1.):
            self.arguments[approximation.name].append(np.asarray(values).reshape(-1))
            return super().evaluate(values, approximation, output_weights)

    class RecordingPlan(planner.Plan):
        def __init__(self):
            super().__init__('bert')
            self.arguments = {}

        def apply(self, name, values, label=None):
            self.arguments[name] = np.asarray(values).reshape(-1)
            return super().apply(name, values, label)

    arithmetic = RecordingArithmetic()
    ops = EncryptedTensorOps(arithmetic, layout)
    contracts = public_toy_approximations('bert')
    mask, counts = layout.attention_mask(public_mask)
    output = ops.attention(*[layout.pack_client_embeddings(x) for x in (query, key, value)],
        heads=heads, kv_heads=heads, head_dim=head_dim, mask=mask, counts=counts,
        exponential=contracts['exp'], reciprocal=contracts['reciprocal'])
    q, k, v = [x.reshape(batch, sequence, heads, head_dim).transpose(0, 2, 1, 3)
               for x in (query, key, value)]
    plan = RecordingPlan()
    probabilities = plan.softmax(q @ k.swapaxes(-1, -2) / np.sqrt(head_dim), public_mask[:, None, None, :])
    for name in ('exp', 'reciprocal'):
        runtime_arguments = np.concatenate(arithmetic.arguments[name])
        np.testing.assert_allclose(np.sort(plan.arguments[name]), np.sort(runtime_arguments), rtol=0, atol=1e-12)
        assert plan.contracts[name] == (contracts[name].domain, len(contracts[name].coefficients) - 1)
    expected = (probabilities @ v).transpose(0, 2, 1, 3).reshape(batch, sequence, hidden)
    np.testing.assert_allclose(layout.unpack_client_features(output), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize('problem', ['bert', 'qwen'])
@pytest.mark.parametrize('case_number', range(1, 7))
def test_complete_ciphertext_operator_graph_public_planning(problem, case_number):
    """Only a NumPy test double: graph semantics, not encrypted execution."""
    import importlib.util
    import json
    from encrypted_transformer import bert_ciphertext_forward, qwen_ciphertext_forward
    folder = Path(__file__).resolve().parents[1] / 'agent_pset' / ('p2_bert' if problem == 'bert' else 'p4_qwen')
    spec = importlib.util.spec_from_file_location(f'oracle_{problem}', folder / 'src/reference.py')
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    case = json.loads((folder / f'test/test{case_number}.in').read_text())
    result = oracle.make_case(case)
    weights, embeddings, expected = result[0], result[1], result[-1]
    batch, sequence = embeddings.shape[:2]
    slots = (1 << (batch - 1).bit_length()) * (1 << (sequence - 1).bit_length()) ** 2
    layout = PairLayout(batch, sequence, slots)
    ops = EncryptedTensorOps(PublicPlanningArithmetic(slots), layout)
    approximations = public_toy_approximations(problem)
    features = layout.pack_client_embeddings(embeddings)
    if problem == 'bert':
        outputs = bert_ciphertext_forward(ops, features, case['config'], weights, result[2], approximations)
        actual = layout.unpack_client_features(outputs)[:, 0]
        assert np.all(np.abs(actual - expected) <= .02 + .01 * np.abs(expected))
        np.testing.assert_array_equal(actual[:, 1] >= actual[:, 0], expected[:, 1] >= expected[:, 0])
    else:
        outputs = qwen_ciphertext_forward(ops, features, case['config'], weights, approximations)
        actual = layout.unpack_client_features(outputs)[:, -1]
        assert np.max(np.abs(actual - expected)) <= .20
        assert np.sqrt(np.mean((actual - expected) ** 2)) / max(np.sqrt(np.mean(expected ** 2)), .01) <= .10
