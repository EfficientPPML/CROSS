import pytest
from test_rmsnorm import setup,gammas

@pytest.mark.parametrize('which',['refresh','compiled_helper'])
def test_replaced_refresh_accessor_must_fail_before_next_primitive(which,gammas):
    o,a,r=setup('10.input.inverse_sqrt',gammas)
    if which=='refresh':
        original=r.refresh;r.refresh=lambda *args,**kwargs:original(*args,**kwargs)
    else:r.compiled_primitives[2].rescale=lambda *args,**kwargs:None
    with pytest.raises(ValueError):o._rms_guard()
    assert not a.calls
