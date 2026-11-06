"""Early boundary regressions; arithmetic spy proves rejection before work."""
import numpy as np
import pytest
from test_first_attention_ideal import arrays,primitives,setup
from test_initial_ideal import ideal_primitives
from test_softmax_tile import refresh_and_tracked
from ideal_primitives import DELTA

@pytest.mark.parametrize('kind',('float','numpy_int'))
def test_original_positions_type_alias_fails_before_any_primitive(arrays,monkeypatch,kind):
    owner,ops,refresh,sources,model=setup(arrays,15)
    convert=float if kind=='float' else np.int64
    owner.positions=tuple(tuple(convert(x) for x in row) for row in owner.positions)
    def forbidden(*a,**k):raise AssertionError('arithmetic reached')
    monkeypatch.setattr(owner.linear,'linear_tile',forbidden)
    with pytest.raises(ValueError):owner.evaluate(sources)
    assert not ops.calls and not refresh.calls

@pytest.mark.parametrize('scale',(np.nextafter(DELTA,0.),np.nextafter(DELTA,np.inf),np.float64(DELTA)))
def test_fresh_input_schema_requires_exact_typed_canonical_scale(arrays,monkeypatch,scale):
    owner,ops,refresh,sources,model=setup(arrays,15)
    for value in sources:value._ckks_scale=scale
    def forbidden(*a,**k):raise AssertionError('arithmetic reached')
    monkeypatch.setattr(owner.linear,'linear_tile',forbidden)
    with pytest.raises(ValueError):owner.evaluate(sources)
    assert not ops.calls and not refresh.calls
