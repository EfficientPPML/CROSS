"""Complete128-key plaintext references with actual binary64 scale equations."""
from pathlib import Path
import json
import math
from types import SimpleNamespace
import numpy as np
import pytest
from test_routing import Arithmetic, Value, backend_free, SLOTS
from encrypted_bert_attention_tile import BertAttentionTile

PROFILE = json.loads((Path(__file__).parent/'p60_profile_and_refresh.json').read_text())['profile']
Q, P = PROFILE['q'], PROFILE['p']
SCALE = PROFILE['canonical_scale']


def value(data, level, scale):
    result = Value(data, level)
    result._ckks_scale = scale
    result.moduli = tuple(Q[:2*(level+1)])
    result.degree_layout = (256, 512)
    return result


class ActualScaleArithmetic(Arithmetic):
    scale = SCALE
    def __init__(self):
        super().__init__()
        self.ctx.dnum = 8
        self.ctx.degree_layout = (256, 512)
        cache = self.ctx._param_cache
        cache.composite_degree = 2
        cache.dnum = 8
        cache.degree_layout = (256, 512)
        cache.q_towers, cache.p_towers = list(Q), list(P)
        keys = BertAttentionTile.required_rotation_indices()
        cache.raw_rotation_keys = {k: {'a': object(), 'b': object()} for k in keys}
        cache.coef_maps = {k: object() for k in keys}
    def rotate(self, source, shift):
        assert shift in self.ctx._param_cache.raw_rotation_keys
        self.calls.append(('rotate', shift))
        return value(np.roll(source.data, -shift), source.level, source._ckks_scale)
    def add(self, left, right):
        self.calls.append(('add',))
        assert type(left._ckks_scale) is type(right._ckks_scale) and left._ckks_scale == right._ckks_scale
        return value(left.data+right.data, min(left.level, right.level), left._ckks_scale)
    def multiply(self, left, right):
        self.calls.append(('multiply',))
        level = min(left.level, right.level)
        scale = left._ckks_scale*right._ckks_scale/math.prod(Q[2*level:2*level+2])
        return value(left.data*right.data, level-1, scale)
    def weighted_sum(self, values, weights):
        self.calls.append(('weighted_sum',))
        level = min(v.level for v in values)
        divisor = math.prod(Q[2*level:2*level+2])
        encoded_scales = [(self.scale*divisor)/v._ckks_scale for v in values]
        products = [float(v._ckks_scale)*float(encoded) for v, encoded in zip(values, encoded_scales)]
        assert all(x == products[0] for x in products)
        scale = products[0]/divisor
        return value(sum(v.data*w for v, w in zip(values, weights)), level-1, scale)


def feature_tiles(matrix, *, level=48, scale=SCALE, seed=1):
    assert matrix.shape[1] == 768 and matrix.shape[0] % 16 == 0
    rng = np.random.default_rng(seed)
    result = []
    for start in range(0, matrix.shape[0], 16):
        packed = rng.uniform(-100, 100, (16, 4096))
        packed[:, :768] = matrix[start:start+16]
        result.append(value(packed.reshape(-1), level, scale))
    return result


def test_all128_scores_match_independent_dense_heads_and_preserve_inputs():
    rng = np.random.default_rng(2026110301)
    q = rng.normal(size=(16, 768))
    k = rng.normal(size=(128, 768))
    query, = feature_tiles(q, scale=SCALE*.93, seed=2)
    keys = feature_tiles(k, scale=SCALE*1.07, seed=3)
    originals = [v.data.copy() for v in (query, *keys)]
    arithmetic = ActualScaleArithmetic()
    graph = BertAttentionTile(arithmetic)
    output = graph.scores(query, keys)
    # The reference has no routing masks, offsets or rotations.
    reference = np.einsum('qhd,khd->qkh', q.reshape(16, 12, 64), k.reshape(128, 12, 64))/8
    expected = np.zeros((16, 4096))
    expected[:, :2048].reshape(16, 128, 16)[:, :, :12] = reference
    np.testing.assert_allclose(output.data, expected.reshape(-1), rtol=0, atol=2e-14)
    assert output.level == 45 and math.isclose(output._ckks_scale, SCALE, rel_tol=1e-12)
    assert len(BertAttentionTile.required_rotation_indices()) == 89
    assert graph.last_counts == dict(row_alignment_rotations=120, ciphertext_products=128,
        head_rotations=768, head_additions=768, route_rotations=3440,
        route_public_multiplications=3584, route_ciphertext_additions=3328,
        route_actual_composite_rescales=256, route_public_multiplication_levels=256,
        accumulator_additions=127)
    for source, before in zip((query, *keys), originals):
        np.testing.assert_array_equal(source.data, before)
        assert source._public_bound == 'deliberately stale'
    assert not hasattr(output, '_public_bound')


def test_all128_value_terms_match_independent_dense_reference_with_hostile_padding():
    rng = np.random.default_rng(2026110302)
    raw = rng.uniform(.01, 1., (16, 128, 12))
    probabilities = raw/raw.sum(axis=1, keepdims=True)
    packed = rng.uniform(-100, 100, (16, 4096))
    packed[:, :2048].reshape(16, 128, 16)[:, :, :12] = probabilities
    probability = value(packed.reshape(-1), 25, math.nextafter(SCALE, math.inf))
    matrix = rng.normal(size=(128, 768))
    values = feature_tiles(matrix, scale=math.nextafter(SCALE, -math.inf), seed=4)
    originals = [v.data.copy() for v in (probability, *values)]
    graph = BertAttentionTile(ActualScaleArithmetic())
    output = graph.mix_values(probability, values)
    reference = np.einsum('qkh,khd->qhd', probabilities, matrix.reshape(128, 12, 64)).reshape(16, 768)
    expected = np.zeros((16, 4096))
    expected[:, :768] = reference
    np.testing.assert_allclose(output.data, expected.reshape(-1), rtol=0, atol=2e-15)
    assert output.level == 22 and graph.last_counts['ciphertext_products'] == 128
    assert graph.last_counts['route_rotations'] == 3440 and graph.last_counts['accumulator_additions'] == 127
    for source, before in zip((probability, *values), originals):
        np.testing.assert_array_equal(source.data, before)
        assert source._public_bound == 'deliberately stale'
    assert not hasattr(output, '_public_bound')


def inputs():
    source = value(np.ones(SLOTS), 48, SCALE)
    tiles = [value(np.ones(SLOTS), 48, SCALE) for _ in range(8)]
    return source, tiles


@pytest.mark.parametrize('operation', ['scores', 'mix_values'])
@pytest.mark.parametrize('target_event', [1, 128])
@pytest.mark.parametrize('mutation', ['input_nsd', 'input_payload', 'result_nsd', 'result_payload', 'context', 'key', 'owner'])
def test_callback_corruption_stops_before_another_primitive(operation, target_event, mutation):
    arithmetic = ActualScaleArithmetic()
    source, tiles = inputs()
    calls = None
    event = 0
    def progress(label, result, counts):
        nonlocal calls, event
        event += 1
        if event != target_event:
            return
        calls = len(arithmetic.calls)
        if mutation == 'input_nsd': tiles[-1]._ckks_nsd = True
        elif mutation == 'input_payload': source.polynomial = source.polynomial.copy()
        elif mutation == 'result_nsd': result._ckks_nsd = 1.0
        elif mutation == 'result_payload': result.polynomial = result.polynomial.copy()
        elif mutation == 'context': arithmetic.ctx._param_cache.batch = True
        elif mutation == 'key': arithmetic.ctx._param_cache.coef_maps.pop(1)
        elif mutation == 'owner': graph.arithmetic = ActualScaleArithmetic()
    graph = BertAttentionTile(arithmetic, progress=progress)
    with pytest.raises(ValueError):
        getattr(graph, operation)(source, tiles)
    assert calls is not None and len(arithmetic.calls) == calls and not graph._active


@pytest.mark.parametrize('mutation', ['tile_count', 'level', 'scale', 'nsd', 'missing_key', 'secret', 'owner'])
def test_invalid_source_rejected_before_work(mutation):
    arithmetic = ActualScaleArithmetic()
    graph = BertAttentionTile(arithmetic)
    source, tiles = inputs()
    if mutation == 'tile_count': tiles.pop()
    elif mutation == 'level': tiles[-1].level = 47
    elif mutation == 'scale': tiles[-1]._ckks_scale = math.nextafter(SCALE, math.inf)
    elif mutation == 'nsd': tiles[-1]._ckks_nsd = True
    elif mutation == 'missing_key': arithmetic.ctx._param_cache.raw_rotation_keys.pop(1)
    elif mutation == 'secret': arithmetic.ctx.parameters['secret_key'] = object()
    elif mutation == 'owner': graph.arithmetic = ActualScaleArithmetic()
    with pytest.raises(ValueError): graph.scores(source, tiles)
    assert not arithmetic.calls and not graph._active


def test_reentry_fails_and_releases_active_flag():
    arithmetic = ActualScaleArithmetic()
    source, tiles = inputs()
    def callback(*args): graph.scores(source, tiles)
    graph = BertAttentionTile(arithmetic, progress=callback)
    with pytest.raises(ValueError, match='reentered'):
        graph.scores(source, tiles)
    assert not graph._active and graph.last_counts is None
