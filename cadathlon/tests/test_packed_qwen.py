"""Plaintext layout and symbolic-depth tests, not encrypted inference evidence."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'cadathlon/common'))
from encrypted_packed_qwen import MatrixDiagonals, PackedQwen1, SingleTokenFeatureLayout
from encrypted_polynomial import EncryptedPolynomialEvaluator
from encrypted_transformer import (EncryptedTensorOps, PairLayout,
                                   public_toy_approximations, qwen_ciphertext_forward)
from support import load_module, metrics


@pytest.mark.parametrize('changes,batch,sequence', [
    ({}, 2, 1), ({}, 1, 2), ({'hidden_size': 32}, 1, 1),
    ({'attention_bias': True}, 1, 1), ({'tie_word_embeddings': False}, 1, 1),
    ({'hidden_act': 'relu'}, 1, 1),
])
def test_unsupported_packed_session_fails_before_client_key_generation(monkeypatch, changes, batch, sequence):
    from encrypted_reference import create_toy_session
    from jaxite_word import key_gen
    config = json.loads((ROOT/'cadathlon/agent_pset/p4_qwen/test/test1.in').read_text())['config']
    config.update(changes)
    def forbidden_keygen(*args, **kwargs):
        raise AssertionError('unsupported packed workloads must fail before key generation')
    monkeypatch.setattr(key_gen, 'gen_pke_pair', forbidden_keygen)
    with pytest.raises(ValueError, match='one-token|packed Qwen'):
        create_toy_session('qwen', config, {}, batch=batch, sequence_length=sequence,
                           devices=(SimpleNamespace(),), packed=True)


def test_single_token_client_codec_has_one_periodic_input_and_one_output():
    layout = SingleTokenFeatureLayout(128)
    embeddings = np.arange(16, dtype=float).reshape(1, 1, 16)/16
    packed = layout.pack_client_embeddings(embeddings)
    assert len(packed) == 1
    expected_block = np.pad(embeddings.reshape(-1), (0, 16))
    np.testing.assert_array_equal(packed[0], np.tile(expected_block, 4))
    decoded = np.arange(128, dtype=float)
    np.testing.assert_array_equal(layout.unpack_client_features([decoded]),
                                  decoded[:32].reshape(1, 1, 32))
    for malformed in (np.zeros((1, 2, 16)), np.full((1, 1, 16), np.nan)):
        with pytest.raises(ValueError, match='client input'):
            layout.pack_client_embeddings(malformed)
    with pytest.raises(ValueError, match='one full-slot'):
        layout.unpack_client_features([decoded, decoded])


@pytest.mark.parametrize('rows,columns', [(16, 24), (24, 16), (16, 32), (32, 16)])
def test_rectangular_signed_diagonals_keep_feature_blocks_separate(rows, columns):
    rng = np.random.default_rng(207)
    weight = rng.normal(size=(rows, columns))
    # Omitted positive and negative diagonals must not become required keys.
    for displacement in (-3, 7):
        for row in range(rows):
            if 0 <= row+displacement < columns:
                weight[row, row+displacement] = 0.
    inputs = rng.normal(size=(4, columns))
    packed = np.pad(inputs, ((0, 0), (0, 32-columns))).reshape(-1)
    source = MatrixDiagonals(weight, len(packed))
    assert (-3) % len(packed) not in source.active
    assert 7 not in source.active
    actual = sum(diagonal*np.roll(packed, -index)
                 for index, diagonal in source.materialize_diagonals(source.active).items())
    expected = np.pad(inputs@weight.T, ((0, 0), (0, 32-rows))).reshape(-1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@dataclass
class _Value:
    slots: np.ndarray
    level: int
    _ckks_scale: float = 1.


class _PlanningArithmetic(EncryptedPolynomialEvaluator):
    """Run actual polynomial composition over public NumPy/symbolic values."""
    def __init__(self, slots):
        self.ctx = SimpleNamespace(num_slots=slots)
        self.strategy, self.refresher, self.scale = 'power', None, 1.

    def _level(self, value):
        return value.level

    def _lower(self, value, level):
        assert level <= value.level
        return _Value(value.slots, level)

    def add(self, left, right, subtract=False):
        return _Value(left.slots-right.slots if subtract else left.slots+right.slots,
                      min(left.level, right.level))

    def rotate(self, value, shift):
        return _Value(np.roll(value.slots, -shift), value.level)

    def weighted_sum(self, values, weights, bias=0.):
        values, weights = tuple(values), tuple(weights)
        assert len(values) == len(weights) and values
        return _Value(sum(value.slots*weight for value, weight in zip(values, weights))+bias,
                      min(value.level for value in values)-1)

    def multiply(self, left, right):
        return _Value(left.slots*right.slots, min(left.level, right.level)-1)

    def checkpoint(self, value, label):
        return value


class _PlanningPackedQwen(PackedQwen1):
    def __init__(self, slots):
        self.ctx = SimpleNamespace(num_slots=slots)
        self.features = 32
        self.arithmetic = _PlanningArithmetic(slots)
        self.linear_shapes = []

    def linear(self, value, weight, bias=None):
        source = MatrixDiagonals(weight, self.ctx.num_slots)
        result = sum(diagonal*np.roll(value.slots, -index)
                     for index, diagonal in source.materialize_diagonals(source.active).items())
        if bias is not None:
            result += self.public_feature(bias)
        self.linear_shapes.append(weight.shape)
        return _Value(result, value.level-1)


def test_complete_packed_graph_matches_feature_graph_and_original_fixture_at_depth38():
    folder = ROOT / 'cadathlon/agent_pset/p4_qwen'
    oracle = load_module(folder/'src/reference.py')
    case = json.loads((folder/'test/test1.in').read_text())
    weights, embeddings, exact = oracle.make_case(case)
    golden = np.asarray(json.loads((folder/'test/test1.out').read_text())['logits'])
    np.testing.assert_allclose(exact, golden, atol=1e-10, rtol=1e-10)
    approximations = public_toy_approximations('qwen')
    layout = SingleTokenFeatureLayout(64)
    packed = _PlanningPackedQwen(64)
    output = packed.evaluate(_Value(layout.pack_client_embeddings(embeddings)[0], 38),
                             case['config'], weights, approximations)
    assert output.level == 0
    actual = layout.unpack_client_features([output.slots])[:, 0]

    # The independently structured per-feature graph must produce the same
    # polynomial result. Both use public planning values; neither executes HE.
    feature_layout = PairLayout(1, 1, 1)
    feature_ops = EncryptedTensorOps(_PlanningArithmetic(1), feature_layout)
    feature_inputs = [_Value(slots, 38) for slots in feature_layout.pack_client_embeddings(embeddings)]
    feature_outputs = qwen_ciphertext_forward(feature_ops, feature_inputs, case['config'], weights, approximations)
    assert all(value.level == 0 for value in feature_outputs)
    feature_result = feature_layout.unpack_client_features([value.slots for value in feature_outputs])[:, 0]
    np.testing.assert_allclose(actual, feature_result, rtol=0, atol=1e-12)
    errors = metrics(actual, golden)
    assert errors['max_abs_error'] <= .20
    assert errors['nrmse'] <= .10
    # GQA expansion is fused into the existing first V projection, not an
    # additional linear transform or a scalar-feature reinterpretation.
    assert packed.linear_shapes == [(32, 16), (16, 32), (24, 16), (24, 16), (16, 24), (32, 16)]
