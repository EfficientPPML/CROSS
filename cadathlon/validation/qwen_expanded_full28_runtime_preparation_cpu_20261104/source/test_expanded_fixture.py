"""Exact synthetic placeholder inventory; no key payloads or crypto execution."""
import json
import numpy as np
import pytest
from pathlib import Path
from join_ideal import Arithmetic,Value

@pytest.mark.parametrize('shift',[0,65536,-65536])
def test_period_alias_is_real_noop(shift):
    a=Arithmetic();value=Value(np.arange(65536,dtype=float))
    assert a.rotate(value,shift)is value and not a.calls

def test_exact_inventory_and_unplanned_rotation_rejected():
    a=Arithmetic();plan=json.loads((Path(__file__).parent/'inputs/actual_key_plan.json').read_bytes())
    expected=set(plan['combined_keys'])
    assert len(expected)==166 and set(a.ctx._param_cache.raw_rotation_keys)==set(a.ctx._param_cache.coef_maps)==expected
    missing=next(k for k in range(1,65536) if k not in expected);value=Value(np.arange(65536,dtype=float))
    with pytest.raises(ValueError,match='absent'):a.rotate(value,missing)
    assert not a.calls
    key=next(k for k in sorted(expected) if 0<k<65536)
    np.testing.assert_array_equal(a.rotate(value,key).polynomial,np.roll(value.polynomial,-key))

def test_removing_planned_key_rejects_before_primitive():
    a=Arithmetic();key=next(k for k in a.ctx._param_cache.raw_rotation_keys if 0<k<65536)
    del a.ctx._param_cache.raw_rotation_keys[key]
    with pytest.raises(ValueError,match='absent'):a.rotate(Value(np.zeros(65536)),key)
    assert not a.calls

