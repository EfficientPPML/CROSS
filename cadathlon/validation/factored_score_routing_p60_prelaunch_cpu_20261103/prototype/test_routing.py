import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'checkout/cadathlon/common'))
from encrypted_shallow_score_routing import ShallowScoreLayout, ShallowScoreRouting, SLOTS


class Value:
    batch = 1
    degree = 2*SLOTS
    num_elements = 2
    _ckks_nsd = 1
    _ckks_scale = 2.**56
    def __init__(self, data, level=5):
        self.data = np.array(data, copy=True)
        self.polynomial = self.data
        self.ntt_ctx = object()
        self.level = level
        self._public_bound = 'deliberately stale'


class Arithmetic:
    scale = 2.**56
    refresher = None
    def __init__(self):
        self.calls = []
        self.ctx = SimpleNamespace(degree=2*SLOTS, num_slots=SLOTS, composite_degree=2,
            secret_key=None, parameters={}, _param_cache=SimpleNamespace(secret_key=None,batch=1))
    def _level(self, value):
        if value._ckks_nsd != 1:
            raise ValueError('invalid noise scale degree')
        return value.level
    def rotate(self, value, shift):
        self.calls.append('rotate')
        return Value(np.roll(value.data, -shift), value.level)
    def weighted_sum(self, values, weights):
        self.calls.append('weighted_sum')
        return Value(sum(v.data*w for v,w in zip(values, weights)), min(v.level for v in values)-1)


@pytest.fixture(autouse=True)
def backend_free(monkeypatch):
    monkeypatch.setitem(sys.modules, 'jaxite_word', SimpleNamespace(Polynomial=Value))
    monkeypatch.setitem(sys.modules, 'support', SimpleNamespace(require_secure_execution=lambda *args: None))
    monkeypatch.setitem(sys.modules, 'encrypted_packed_qwen', SimpleNamespace(CompiledArithmetic=Arithmetic))


@pytest.mark.parametrize('tile', range(8))
@pytest.mark.parametrize('offset', range(16))
def test_all_maps_against_independent_gather(tile, offset):
    rng = np.random.default_rng(710000 + tile*16+offset)
    data = rng.uniform(-100,100,SLOTS)
    source = Value(data)
    layout = ShallowScoreLayout(tile,offset)
    router = ShallowScoreRouting(Arithmetic(),layout)
    # Independent coordinates; neither tested target_indices nor diagonal mask
    # code participates in the direct reference.
    expected = np.zeros(SLOTS)
    anchors = np.zeros(SLOTS)
    for row in range(16):
        for head in range(12):
            src = row*4096+head*64
            dst = row*4096+(tile*16+(row+offset)%16)*16+head
            expected[dst] = data[src]/8
            anchors[src] = data[src]/8
    out = router.forward(source,gain=.125)
    np.testing.assert_array_equal(out.data,expected)
    recovered = router.inverse(out)
    np.testing.assert_array_equal(recovered.data,anchors)
    # Inverse must independently reject all hostile non-target slots, rather
    # than relying on zeros produced by forward.
    hostile = rng.uniform(-100,100,SLOTS)
    inverse_expected = np.zeros(SLOTS)
    for row in range(16):
        for head in range(12):
            inverse_expected[row*4096+head*64] = hostile[
                row*4096+(tile*16+(row+offset)%16)*16+head]
    np.testing.assert_array_equal(router.inverse(Value(hostile)).data,inverse_expected)
    np.testing.assert_array_equal(source.data,data)
    assert source._public_bound == 'deliberately stale'
    assert not hasattr(out,'_public_bound') and not hasattr(recovered,'_public_bound')
    assert out.level == 3 and recovered.level == 1
    assert router.counts['actual_composite_rescales'] == 6
    assert router.counts['public_multiplications'] == 84


def test_worst_profile_and_progress():
    layout = ShallowScoreLayout(7,15)
    assert len(layout.rotation_indices(include_inverse=False)) == 27
    assert len(layout.rotation_indices()) == 54
    events=[]
    router=ShallowScoreRouting(Arithmetic(),layout,progress=lambda *a:events.append(a))
    out=router.forward(Value(np.arange(SLOTS)))
    router.inverse(out)
    assert len(events)==4
    assert router.counts == {'rotations':54,'public_multiplications':56,
        'ciphertext_additions':52,'actual_composite_rescales':4,'public_multiplication_levels':4}


def test_bound_aware_owner_rejected_before_work():
    arithmetic=Arithmetic()
    arithmetic.declare_bound=lambda *args:None
    with pytest.raises(TypeError,match='bound-aware'):
        ShallowScoreRouting(arithmetic,ShallowScoreLayout(0,0))
    with pytest.raises(TypeError,match='CompiledArithmetic'):
        ShallowScoreRouting(object(),ShallowScoreLayout(0,0))


@pytest.mark.parametrize('tile,offset', [(True,0),(0,False),(-1,0),(8,0),(0,-1),(0,16),(1.,0)])
def test_bad_layout(tile,offset):
    with pytest.raises(ValueError): ShallowScoreLayout(tile,offset)


@pytest.mark.parametrize('gain', [True,False,complex(1),float('nan'),float('inf'),0.,2.,1e-300])
def test_bad_gain_rejected_before_ciphertext(gain):
    with pytest.raises(ValueError): ShallowScoreRouting(Arithmetic(),ShallowScoreLayout(0,0)).forward(None,gain=gain)


@pytest.mark.parametrize('where', ['ctx','cache','parameters'])
def test_secret_injection_is_rechecked(where):
    arithmetic=Arithmetic();router=ShallowScoreRouting(arithmetic,ShallowScoreLayout(0,0))
    if where=='ctx': arithmetic.ctx.secret_key=object()
    elif where=='cache': arithmetic.ctx._param_cache.secret_key=object()
    else: arithmetic.ctx.parameters['secret_key']=object()
    with pytest.raises(ValueError): router.forward(Value(np.ones(SLOTS)))


@pytest.mark.parametrize('mutation', ['level','scale','nsd','refresher','batch','degree','elements'])
def test_invalid_input(mutation):
    arithmetic=Arithmetic();router=ShallowScoreRouting(arithmetic,ShallowScoreLayout(0,0))
    value=Value(np.ones(SLOTS))
    if mutation=='level':value.level=1
    elif mutation=='scale':value._ckks_scale=123.
    elif mutation=='nsd':value._ckks_nsd=2
    elif mutation=='refresher':arithmetic.refresher=object()
    elif mutation=='batch':value.batch=2
    elif mutation=='degree':value.degree=16384
    elif mutation=='elements':value.num_elements=3
    with pytest.raises(ValueError):router.forward(value)


@pytest.mark.parametrize('exception_type',[KeyboardInterrupt,SystemExit])
def test_cleanup_only_cancellation_preserved_after_all_actions(exception_type):
    from lifecycle import CleanupTracker
    cancellation=exception_type('cleanup interrupted')
    tracker=CleanupTracker(None);completed=[]
    def fail():raise cancellation
    tracker.attempt('first',fail)
    tracker.attempt('second',lambda:completed.append(True))
    assert completed==[True] and len(tracker.errors)==1
    with pytest.raises(exception_type) as caught:tracker.finish()
    assert caught.value is cancellation


@pytest.mark.parametrize('exception_type',[KeyboardInterrupt,SystemExit])
def test_primary_failure_survives_cleanup_cancellation(exception_type):
    from lifecycle import CleanupTracker
    primary=ValueError('original numerical failure')
    tracker=CleanupTracker(primary)
    def fail():raise exception_type('cleanup interrupted')
    tracker.attempt('cleanup',fail)
    tracker.finish()
    assert any(exception_type.__name__ in note for note in primary.__notes__)


def test_cleanup_failure_recorded_without_skipping_later_actions():
    from lifecycle import CleanupTracker
    tracker=CleanupTracker(None);completed=[]
    def fail():raise RuntimeError('close failed')
    tracker.attempt('close',fail)
    tracker.attempt('next',lambda:completed.append(True))
    tracker.finish()
    assert completed==[True] and 'close failed' in tracker.errors[0]
