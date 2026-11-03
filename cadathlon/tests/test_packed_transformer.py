"""Independent packed-transformer plaintext/layout/depth checks; no HE passes."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'cadathlon/common'))
from encrypted_polynomial import EncryptedPolynomialEvaluator
from encrypted_transformer import (PairLayout, EncryptedTensorOps, public_toy_approximations,
                                   bert_ciphertext_forward, qwen_ciphertext_forward)
from encrypted_packed_qwen import MatrixDiagonals
from support import load_module, metrics

PACKED_CASE_OBSERVATIONS = {}


@dataclass
class _Value:
    slots: np.ndarray
    level: int
    _ckks_scale: float = 1.


class _SubAtLevel:
    def __init__(self, arithmetic):
        self.arithmetic = arithmetic

    def __getitem__(self, level):
        return SimpleNamespace(sub=lambda a, b: self.arithmetic.add(a, b, subtract=True))


class _AddAtLevel:
    def __init__(self, arithmetic):
        self.arithmetic = arithmetic

    def __getitem__(self, level):
        return SimpleNamespace(add_plain=self.arithmetic.add)


class PlanningArithmetic(EncryptedPolynomialEvaluator):
    """Actual runtime polynomial composition over NumPy and symbolic levels."""
    def __init__(self, slots):
        self.ctx = SimpleNamespace(num_slots=slots, he_sub=_SubAtLevel(self), he_add=_AddAtLevel(self))
        self.strategy, self.refresher, self.scale = 'power', None, 1.
        self.domains = {}
        self.rotations = set()

    def _level(self, value):
        return value.level

    def _lower(self, value, level):
        assert level <= value.level
        return _Value(value.slots, level)

    def add(self, left, right, subtract=False):
        return _Value(left.slots-right.slots if subtract else left.slots+right.slots,
                      min(left.level, right.level))

    def rotate(self, value, shift):
        if shift % self.ctx.num_slots:
            self.rotations.add(shift % self.ctx.num_slots)
        return _Value(np.roll(value.slots, -shift), value.level)

    def weighted_sum(self, values, weights, bias=0.):
        values, weights = tuple(values), tuple(weights)
        assert values and len(values) == len(weights)
        return _Value(sum(value.slots*weight for value, weight in zip(values, weights))+bias,
                      min(value.level for value in values)-1)

    def multiply(self, left, right):
        return _Value(left.slots*right.slots, min(left.level, right.level)-1)

    def checkpoint(self, value, label):
        return value

    def _prepare(self, level):
        pass

    def _finish(self, value):
        return value

    def _encode_public(self, slots, level, scale):
        return _Value(np.asarray(slots), level)

    def evaluate(self, value, approximation, *, output_weights=1.):
        low, high = approximation.domain
        observed_low, observed_high = float(np.min(value.slots)), float(np.max(value.slots))
        assert np.isfinite(value.slots).all()
        assert observed_low >= low and observed_high <= high, (
            approximation.name, (observed_low, observed_high), approximation.domain)
        old = self.domains.setdefault(approximation.name, dict(min=observed_low, max=observed_high,
                                                             domain=approximation.domain, calls=0))
        old.update(min=min(old['min'], observed_low), max=max(old['max'], observed_high), calls=old['calls']+1)
        return super().evaluate(value, approximation, output_weights=output_weights)


def _runtime():
    from encrypted_packed_transformer import PackedPairLayout, PackedTransformer, packed_feature_width
    return PackedPairLayout, PackedTransformer, packed_feature_width


def planning_packed(layout):
    _, PackedTransformer, _ = _runtime()
    class PlanningPacked(PackedTransformer):
        def __init__(self, layout):
            self.layout, self.features = layout, layout.features
            self.arithmetic = PlanningArithmetic(layout.slots)
            self.ctx = self.arithmetic.ctx
            self.phase_results = []
    return PlanningPacked(layout)


def test_packed_pair_client_codec_preserves_nonpower_padding_and_feature_zeros():
    PackedPairLayout, _, _ = _runtime()
    batch, sequence, features = 3, 5, 64
    period = 4*8*8*features
    layout = PackedPairLayout(batch, sequence, features, 2*period)
    embeddings = np.arange(batch*sequence*24, dtype=float).reshape(batch, sequence, 24)
    packed = layout.pack_client_embeddings(embeddings)
    assert len(packed) == 1 and packed[0].shape == (2*period,)
    expected = np.zeros((4, 8, 8, features))
    for b in range(4):
        for q in range(8):
            expected[b, q, :, :24] = embeddings[min(b, batch-1), min(q, sequence-1)]
    np.testing.assert_array_equal(packed[0], np.tile(expected.reshape(-1), 2))
    np.testing.assert_array_equal(layout.unpack_client_features(packed, output_features=24), embeddings)


@pytest.mark.parametrize('rows,columns', [(5, 7), (7, 5)])
def test_packed_linear_input_mask_uses_source_feature_coordinates(rows, columns):
    PackedPairLayout, _, _ = _runtime()
    layout = PackedPairLayout(1, 2, 8, 32)
    graph = planning_packed(layout)
    random = np.random.default_rng(20261102 + rows)
    inputs = random.normal(size=(4, 8))
    mask = random.uniform(-1., 1., size=(4, 8))
    mask[::2, 1::2] = 0.
    weight = random.normal(size=(rows, columns))
    bias = random.normal(size=rows)
    actual = graph.linear(_Value(inputs.reshape(-1), 10), weight, bias,
                          input_mask=mask.reshape(-1))
    expected = np.zeros_like(inputs)
    expected[:, :rows] = (inputs[:, :columns] * mask[:, :columns]) @ weight.T + bias
    np.testing.assert_allclose(actual.slots.reshape(inputs.shape), expected, rtol=0, atol=1e-12)
    assert actual.level == 9


@pytest.mark.parametrize('model,number', [(m, n) for m in ('bert', 'qwen') for n in range(1, 7)])
def test_all_public_packed_graphs_match_feature_oracle_and_domain_contracts(model, number):
    PackedPairLayout, _, feature_width = _runtime()
    problem = 'p2_bert' if model == 'bert' else 'p4_qwen'
    folder = ROOT/'cadathlon/agent_pset'/problem
    oracle = load_module(folder/'src/reference.py')
    case = json.loads((folder/f'test/test{number}.in').read_text())
    created = oracle.make_case(case)
    weights, embeddings, exact = created[0], created[1], created[-1]
    mask = created[2] if model == 'bert' else None
    golden = np.asarray(json.loads((folder/f'test/test{number}.out').read_text())['logits'])
    np.testing.assert_allclose(exact, golden, atol=1e-10, rtol=1e-10)
    batch, sequence, hidden = embeddings.shape
    features = feature_width(model, case['config'])
    period = (1 << (batch-1).bit_length())*(1 << (sequence-1).bit_length())**2*features
    layout = PackedPairLayout(batch, sequence, features, max(period, 2*features))
    graph = planning_packed(layout)
    approximations = public_toy_approximations(model)
    output = graph.evaluate(_Value(layout.pack_client_embeddings(embeddings)[0], 1000), model,
                            case['config'], weights, approximations, attention_mask=mask)
    actual = layout.unpack_client_features([output.slots], output_features=golden.shape[-1])[:, 0 if model == 'bert' else -1]

    # Compare against the structurally independent per-feature runtime graph,
    # using the same actual polynomial composition over public NumPy values.
    reference_slots = (1 << (batch-1).bit_length())*(1 << (sequence-1).bit_length())**2
    reference_layout = PairLayout(batch, sequence, reference_slots)
    reference_arithmetic = PlanningArithmetic(reference_slots)
    reference_ops = EncryptedTensorOps(reference_arithmetic, reference_layout)
    inputs = [_Value(slots, 1000) for slots in reference_layout.pack_client_embeddings(embeddings)]
    outputs = (bert_ciphertext_forward(reference_ops, inputs, case['config'], weights, mask, approximations)
               if model == 'bert' else qwen_ciphertext_forward(reference_ops, inputs, case['config'], weights, approximations))
    reference = reference_layout.unpack_client_features([value.slots for value in outputs])[:, 0 if model == 'bert' else -1]
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-10)
    errors = metrics(actual, golden)
    if model == 'bert':
        assert np.all(np.abs(actual-golden) <= .02+.01*np.abs(golden))
        assert np.array_equal(actual[:, 1] >= actual[:, 0], golden[:, 1] >= golden[:, 0])
    else:
        assert errors['nrmse'] <= .10 and errors['max_abs_error'] <= .20
    assert graph.arithmetic.domains
    assert output.level >= 0
    from encrypted_packed_transformer import plan_packed_transformer
    planned = plan_packed_transformer(model, case['config'], weights, batch=batch, sequence=sequence,
                                      slots=layout.slots, attention_mask=mask)
    assert planned['depth'] == 1000-output.level
    assert set(planned['rotations']) == graph.arithmetic.rotations
    full_ring_plan = plan_packed_transformer(model, case['config'], weights, batch=batch, sequence=sequence,
                                            slots=65536, attention_mask=mask)
    assert full_ring_plan['depth'] == planned['depth']
    PACKED_CASE_OBSERVATIONS[f'{model}/test{number}.in'] = {
        'plaintext_numeric_gate': True,
        'feature_width': features, 'period': layout.period,
        'symbolic_packed_depth': 1000-output.level,
        'symbolic_feature_depth': 1000-min(value.level for value in outputs),
        'max_error_vs_feature_polynomial': float(np.max(np.abs(actual-reference))),
        'error_vs_exact_fixture': errors,
        'polynomial_input_domains_including_inactive_slots': graph.arithmetic.domains,
        'planned_rotation_inventory_matches_graph': True,
        'full_ring_rotation_indices': full_ring_plan['rotations'],
        'full_ring_rotation_count': len(full_ring_plan['rotations']),
    }


@pytest.mark.parametrize('model,number', [('bert', 3), ('qwen', 2)])
@pytest.mark.parametrize('packed_pairs', [False, True])
def test_unvalidated_refresh_graphs_fail_before_key_generation(monkeypatch, model, number, packed_pairs):
    from encrypted_reference import create_toy_session
    from jaxite_word import key_gen
    problem = 'p2_bert' if model == 'bert' else 'p4_qwen'
    folder = ROOT/'cadathlon/agent_pset'/problem
    case = json.loads((folder/f'test/test{number}.in').read_text())
    created = load_module(folder/'src/reference.py').make_case(case)
    weights, embeddings = created[:2]
    def forbidden_keygen(*args, **kwargs):
        raise AssertionError('unvalidated refresh must fail before key generation')
    monkeypatch.delenv('CROSS_SKIP_TOPLEVEL_ROTKEYS', raising=False)
    monkeypatch.setattr(key_gen, 'gen_pke_pair', forbidden_keygen)
    with pytest.raises(ValueError, match='requires bounded homomorphic refresh'):
        create_toy_session(model, case['config'], weights, batch=embeddings.shape[0],
                           sequence_length=embeddings.shape[1], devices=(SimpleNamespace(),), packed_pairs=packed_pairs)
