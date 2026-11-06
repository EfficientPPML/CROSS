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


class Barrett:
    computation_format_is_standard = True
    def __init__(self, moduli):
        self.moduli = tuple(moduli)


def add_key_inventory(arithmetic, field=Barrett):
    k = arithmetic.ctx._param_cache
    k.ff_context_cls = field
    k.per_level_rotation_keys = False
    k.raw_rotation_keys = {i: {n: SimpleNamespace(shape=(8, 114, 131072), dtype=np.dtype('uint32'))
                              for n in ('a', 'b')} for i in required_rotation_indices()}
    k.coef_maps = {i: SimpleNamespace(shape=(131072,), dtype=np.dtype('int32')) for i in required_rotation_indices()}


class Payload:
    dtype = np.dtype('uint32')
    def __init__(self, data, level):
        self.data = np.frombuffer(np.asarray(data, dtype='float64').tobytes(), dtype='float64')
        self.shape = (1, 2, 256, 512, 2*(level+1))
    def devices(self):
        return {'ideal-cpu'}


class Value:
    batch, num_elements, degree = 1, 2, 131072
    r, c = 256, 512
    degree_layout, precision, _ckks_nsd = (256, 512), 32, 1
    def __init__(self, data, level=25, scale=DELTA):
        self.level = level
        self._ckks_scale = scale
        self.moduli = Q[:2*(level+1)]
        self.num_moduli = len(self.moduli)
        self.polynomial = Payload(data, level)
        self.ntt_ctx = SimpleNamespace(moduli=self.moduli, ff_ctx=Barrett(self.moduli))
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
        add_key_inventory(self)
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
    monkeypatch.setitem(sys.modules, 'finite_field', SimpleNamespace(BarrettContext=Barrett))
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


@pytest.mark.parametrize('mutation', ['missing_key', 'missing_map', 'key_container', 'map_container',
    'pair', 'a', 'b', 'map', 'key_shape', 'key_dtype', 'map_shape', 'map_dtype', 'per_level', 'field'])
def test_required_key_and_field_binding(mutation):
    a = Arithmetic();r = BSGSScoreRouting(a, FactoredScoreLayout(0, 0));k = a.ctx._param_cache
    index = required_rotation_indices()[0]
    if mutation == 'missing_key':del k.raw_rotation_keys[index]
    elif mutation == 'missing_map':del k.coef_maps[index]
    elif mutation == 'key_container':k.raw_rotation_keys = dict(k.raw_rotation_keys)
    elif mutation == 'map_container':k.coef_maps = dict(k.coef_maps)
    elif mutation == 'pair':k.raw_rotation_keys[index] = dict(k.raw_rotation_keys[index])
    elif mutation in ('a', 'b'):k.raw_rotation_keys[index][mutation] = SimpleNamespace(shape=(8, 114, 131072), dtype=np.dtype('uint32'))
    elif mutation == 'map':k.coef_maps[index] = SimpleNamespace(shape=(131072,), dtype=np.dtype('int32'))
    elif mutation == 'key_shape':k.raw_rotation_keys[index]['a'].shape = (1, 114, 131072)
    elif mutation == 'key_dtype':k.raw_rotation_keys[index]['a'].dtype = np.dtype('uint64')
    elif mutation == 'map_shape':k.coef_maps[index].shape = (65536,)
    elif mutation == 'map_dtype':k.coef_maps[index].dtype = np.dtype('uint32')
    elif mutation == 'per_level':k.per_level_rotation_keys = True
    elif mutation == 'field':k.ff_context_cls = object
    with pytest.raises(ValueError):r.forward(Value(np.ones(SLOTS)))
    assert not a.calls


def test_missing_key_rejected_at_construction():
    a = Arithmetic();a.ctx._param_cache.raw_rotation_keys.clear()
    with pytest.raises(ValueError, match='missing'):BSGSScoreRouting(a, FactoredScoreLayout(0, 0))


def test_callback_tags_stripped_from_every_owned_retained_result():
    captured = [];source = Value(np.ones(SLOTS))
    def callback(label, out, counts):
        if captured:captured[0]._public_callback_tag = 'stale'
        out._public_callback_tag = 'stale'
        captured.append(out)
    r = BSGSScoreRouting(Arithmetic(), FactoredScoreLayout(0, 0), progress=callback)
    out = r.forward(source)
    assert source._public_bound == 'caller annotation'
    assert all(not any(n.startswith('_public_') for n in vars(v)) for v in captured)


@pytest.mark.parametrize('field', ['ntt_moduli', 'field_moduli', 'field_family', 'standard', 'r', 'c'])
@pytest.mark.parametrize('during_callback', [False, True])
def test_nested_ntt_metadata_guard(field, during_callback):
    a = Arithmetic();v = Value(np.ones(SLOTS))
    def mutate():
        if field == 'ntt_moduli':v.ntt_ctx.moduli = tuple(reversed(v.moduli))
        elif field == 'field_moduli':v.ntt_ctx.ff_ctx.moduli = tuple(reversed(v.moduli))
        elif field == 'field_family':v.ntt_ctx.ff_ctx = SimpleNamespace(moduli=v.moduli, computation_format_is_standard=True)
        elif field == 'standard':v.ntt_ctx.ff_ctx.computation_format_is_standard = False
        elif field == 'r':v.r = True
        else:v.c = 256
    r = BSGSScoreRouting(a, FactoredScoreLayout(0, 0), progress=(lambda *args:mutate()) if during_callback else None)
    if not during_callback:mutate()
    with pytest.raises(ValueError):r.forward(v)
    if not during_callback:assert not a.calls
