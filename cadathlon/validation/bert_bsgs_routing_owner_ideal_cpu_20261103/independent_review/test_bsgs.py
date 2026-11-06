"""Actual owner loops with explicit ideal primitive/security doubles; no HE."""
from pathlib import Path
from types import SimpleNamespace
import json
import math
import sys
import numpy as np
import pytest
from encrypted_factored_score_routing import FactoredScoreLayout, SLOTS
from encrypted_shallow_score_routing import ShallowScoreLayout
from encrypted_bsgs_score_routing import BSGSScoreRouting, DELTA, required_rotation_indices

PROFILE = json.loads((Path(__file__).parent/'p60_profile_and_refresh.json').read_text())['profile']
Q, P = tuple(PROFILE['q']), tuple(PROFILE['p'])


class Payload:
    dtype = np.dtype('uint32')
    def __init__(self, data, level):
        self.data = np.frombuffer(np.asarray(data, dtype='float64').tobytes(), dtype='float64')
        self.shape = (1, 2, 256, 512, 2*(level+1))
    def devices(self):
        return {'ideal-cpu'}


class Value:
    batch, num_elements, degree = 1, 2, 131072
    degree_layout, precision, _ckks_nsd = (256, 512), 32, 1
    def __init__(self, data, level=25, scale=DELTA):
        self.level = level
        self._ckks_scale = scale
        self.moduli = Q[:2*(level+1)]
        self.num_moduli = len(self.moduli)
        self.polynomial = Payload(data, level)
        self.ntt_ctx = object()
        self._public_bound = 'caller annotation'


class Helper:
    def __init__(self, context):
        self.ctx = context
    def rotate(self, *args):
        raise AssertionError('ideal facade handles the payload')
    multiply = rotate
    multiply_encoded = rotate
    rescale = rotate
    def clear(self):
        pass


class Arithmetic:
    scale, low_memory, refresher, progress = DELTA, True, None, None
    def __init__(self):
        values = dict(degree=131072, num_slots=65536, composite_degree=2, dnum=8,
                      max_level=49, scaling_factor=DELTA, degree_layout=(256, 512),
                      q_towers=Q, p_towers=P, secret_key=None, batch=1)
        self.ctx = SimpleNamespace(**values, parameters={}, _param_cache=SimpleNamespace(**values))
        self.rotation = Helper(self.ctx)
        self._plaintext_multiply = Helper(self.ctx)
        self._rescale = Helper(self.ctx)
        self.calls = []
    def _level(self, value):
        return value.level
    def rotate(self, value, shift):
        self.calls.append(('rotate', shift))
        return Value(np.roll(value.polynomial.data, -shift), value.level, value._ckks_scale)
    def weighted_sum(self, values, weights):
        self.calls.append(('weighted', len(values)))
        level = min(v.level for v in values)
        # Independent literal production-order recurrence. Production AST tests
        # separately execute the actual methods; no scale planner is called.
        divisor = math.prod(Q[2*level:2*level+2])
        source = values[0]._ckks_scale
        scale = (source*((DELTA*divisor)/source))/divisor
        output = sum(v.polynomial.data*w for v, w in zip(values, weights))
        return Value(output, level-1, scale)
    def add(self, left, right):
        self.calls.append(('add',))
        return Value(left.polynomial.data+right.polynomial.data, min(left.level, right.level), left._ckks_scale)
    def _finish(self, value):
        return value
    def _prepare(self, *args):
        pass
    _lower = _prepare
    _ensure_capacity = _prepare
    _encode_public = _prepare
    _public_slots = _prepare
    _clear_working_caches = _prepare


@pytest.fixture(autouse=True)
def doubles(monkeypatch):
    barriers = []
    monkeypatch.setitem(sys.modules, 'encrypted_packed_qwen', SimpleNamespace(CompiledArithmetic=Arithmetic))
    monkeypatch.setitem(sys.modules, 'jaxite_word', SimpleNamespace(Polynomial=Value))
    monkeypatch.setitem(sys.modules, 'support', SimpleNamespace(require_secure_execution=lambda *args: None))
    monkeypatch.setitem(sys.modules, 'jax', SimpleNamespace(block_until_ready=lambda arrays: barriers.extend(arrays)))
    return barriers


def scatter(data, tile, offset, inverse, gain):
    result = np.zeros(SLOTS)
    for row in range(16):
        for head in range(12):
            source = 4096*row+64*head
            target = 4096*row+16*(16*tile+(row+offset)%16)+head
            result[source if inverse else target] = data[target if inverse else source]*(1. if inverse else gain)
    return result


@pytest.mark.parametrize('tile', range(8))
@pytest.mark.parametrize('offset', range(16))
@pytest.mark.parametrize('inverse,gain', [(False, 1.), (False, .125), (True, 1.)])
def test_every_map_all_slots(tile, offset, inverse, gain):
    rng = np.random.default_rng(89100+tile*16+offset)
    for level, factor in ((49, 1.), (25, .73)):
        data = rng.uniform(-100, 100, SLOTS)
        value = Value(data, level, DELTA*factor)
        arithmetic = Arithmetic()
        owner = BSGSScoreRouting(arithmetic, FactoredScoreLayout(tile, offset))
        out = owner.inverse(value) if inverse else owner.forward(value, gain=gain)
        np.testing.assert_array_equal(out.polynomial.data, scatter(data, tile, offset, inverse, gain))
        np.testing.assert_array_equal(value.polynomial.data, data)
        assert out.level == level-2
        assert value._public_bound == 'caller annotation'
        assert not hasattr(out, '_public_bound')
        assert owner.counts == dict(rotations=11+bool(tile), public_multiplications=28,
            ciphertext_additions=26, actual_composite_rescales=7, public_multiplication_levels=2)
        assert sum(c[0] == 'rotate' for c in arithmetic.calls) == 11+bool(tile)
        assert sum(c[0] == 'weighted' for c in arithmetic.calls) == 7
        assert sum(c[0] == 'add' for c in arithmetic.calls) == 5
        assert owner.last_run['counts'] == dict(owner.counts)
        assert not owner._retained and not owner._active


def test_key_inventory():
    original = set().union(*(set(FactoredScoreLayout(t, o).rotation_indices()) for t in range(8) for o in range(16)))
    proposed = set(required_rotation_indices())
    assert len(original) == 66 and len(proposed) == 36 and proposed <= original
    observed = set()
    for tile in range(8):
        a = Arithmetic();r = BSGSScoreRouting(a, FactoredScoreLayout(tile, 3));v = Value(np.arange(SLOTS))
        r.forward(v);r.inverse(v)
        observed.update(call[1] for call in a.calls if call[0] == 'rotate')
    assert observed == proposed


@pytest.mark.parametrize('gain', [True, False, 1, 0., 2., float('nan'), float('inf'), .5])
def test_bad_gain_before_work(gain):
    a = Arithmetic();r = BSGSScoreRouting(a, FactoredScoreLayout(0, 0))
    with pytest.raises(ValueError):r.forward(None, gain=gain)
    assert not a.calls


@pytest.mark.parametrize('mutation', ['nsd', 'level', 'scale', 'precision', 'shape', 'q_order', 'layout'])
def test_bad_input_before_work(mutation):
    a = Arithmetic();r = BSGSScoreRouting(a, FactoredScoreLayout(0, 0));v = Value(np.ones(SLOTS))
    if mutation == 'nsd':v._ckks_nsd = True
    if mutation == 'level':v.level = 1
    if mutation == 'scale':v._ckks_scale = float('inf')
    if mutation == 'precision':v.precision = 64
    if mutation == 'shape':v.polynomial.shape = (1, 2, 65536)
    if mutation == 'q_order':v.moduli = tuple(reversed(v.moduli))
    if mutation == 'layout':v.degree_layout = [256, 512]
    with pytest.raises(ValueError):r.forward(v)
    assert not a.calls


@pytest.mark.parametrize('target', ['input', 'output'])
@pytest.mark.parametrize('field', ['_ckks_scale', '_ckks_nsd', 'precision', 'moduli', 'polynomial'])
def test_callback_retained_payload_and_metadata(target, field, doubles):
    value = Value(np.arange(SLOTS));original = value.polynomial
    saved = []
    def callback(label, out, counts):
        selected = value if target == 'input' else out
        saved.append(selected.polynomial)
        setattr(selected, field, Payload(np.zeros(SLOTS), selected.level) if field == 'polynomial' else 0)
    owner = BSGSScoreRouting(Arithmetic(), FactoredScoreLayout(0, 0), progress=callback)
    with pytest.raises(ValueError, match='retained'):owner.forward(value)
    assert original in doubles and saved[0] in doubles
    assert not owner._active and not owner._retained


def test_zero_alias_callback_preserves_original_storage(doubles):
    value = Value(np.arange(SLOTS));original = value.polynomial
    def callback(label, out, counts):
        assert label == 'key tile extraction' and out is value
        out.polynomial = Payload(np.zeros(SLOTS), out.level)
    owner = BSGSScoreRouting(Arithmetic(), FactoredScoreLayout(0, 0), progress=callback)
    with pytest.raises(ValueError, match='retained'):owner.inverse(value)
    assert doubles == [original]


@pytest.mark.parametrize('name', ['rotate', 'weighted_sum', 'add', '_level', '_finish', '_encode_public'])
def test_accessor_replacement(name):
    a = Arithmetic();v = Value(np.arange(SLOTS))
    def callback(*args):setattr(a, name, lambda *args: None)
    owner = BSGSScoreRouting(a, FactoredScoreLayout(0, 0), progress=callback)
    with pytest.raises(ValueError, match='accessor'):owner.forward(v)


@pytest.mark.parametrize('field', ['_ckks_scale', 'precision', '_ckks_nsd', 'moduli'])
def test_bad_primitive_metadata(field):
    a = Arithmetic();original = a.weighted_sum
    def bad(values, weights):
        out = original(values, weights)
        setattr(out, field, math.nextafter(out._ckks_scale, math.inf) if field == '_ckks_scale' else 0)
        return out
    a.weighted_sum = bad
    owner = BSGSScoreRouting(a, FactoredScoreLayout(0, 0))
    with pytest.raises((ValueError, TypeError)):owner.forward(Value(np.ones(SLOTS)))


def test_reentry_rejected():
    value = Value(np.ones(SLOTS));errors = []
    owner = None
    def callback(*args):
        with pytest.raises(ValueError, match='reentered'):owner.forward(value)
        errors.append(True)
    owner = BSGSScoreRouting(Arithmetic(), FactoredScoreLayout(1, 0), progress=callback)
    owner.forward(value)
    assert len(errors) == 3


def test_primary_preserved_and_failed_barrier_quarantined(monkeypatch):
    value = Value(np.ones(SLOTS));original = value.polynomial;primary = KeyboardInterrupt('primary')
    def callback(*args):raise primary
    def fail(arrays):raise RuntimeError('barrier')
    monkeypatch.setattr(sys.modules['jax'], 'block_until_ready', fail)
    owner = BSGSScoreRouting(Arithmetic(), FactoredScoreLayout(0, 0), progress=callback)
    with pytest.raises(KeyboardInterrupt) as caught:owner.forward(value)
    assert caught.value is primary and original in owner._failed_arrays[0]
    assert any('synchronization' in note for note in primary.__notes__)
    with pytest.raises(ValueError, match='unresolved'):owner.forward(value)


@pytest.mark.parametrize('where', ['ctx', 'cache', 'parameters', 'helper', 'refresh', 'low_memory', 'layout'])
def test_borrowed_state_rejected(where):
    a = Arithmetic();r = BSGSScoreRouting(a, FactoredScoreLayout(0, 0))
    if where == 'ctx':a.ctx.secret_key = object()
    if where == 'cache':a.ctx._param_cache.secret_key = object()
    if where == 'parameters':a.ctx.parameters['secret_key'] = object()
    if where == 'helper':a.rotation = Helper(a.ctx)
    if where == 'refresh':a.refresher = object()
    if where == 'low_memory':a.low_memory = False
    if where == 'layout':object.__setattr__(r.layout, 'offset', True)
    with pytest.raises(ValueError):r.forward(Value(np.ones(SLOTS)))
    assert not a.calls


@pytest.mark.parametrize('cls,name', [(FactoredScoreLayout, 'tile_shift'),
    (ShallowScoreLayout, 'row_keys'), (ShallowScoreLayout, 'head_diagonals'),
    (ShallowScoreLayout, 'key_diagonals')])
def test_inherited_layout_accessors_bound(monkeypatch, cls, name):
    a = Arithmetic();r = BSGSScoreRouting(a, FactoredScoreLayout(1, 0))
    monkeypatch.setattr(cls, name, property(lambda self: 0) if name in ('tile_shift', 'row_keys') else lambda *args: ())
    with pytest.raises(ValueError, match='inherited accessor'):r.forward(Value(np.ones(SLOTS)))
    assert not a.calls
