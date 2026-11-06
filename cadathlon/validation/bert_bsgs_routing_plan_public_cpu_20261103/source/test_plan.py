import itertools, sys
import numpy as np
import pytest
import plan

CASES = list(itertools.product(range(8), range(16), ('forward','scaled','inverse')))

@pytest.mark.parametrize('tile,offset,mode', CASES)
def test_all_physical_slots_against_direct_scatter(tile, offset, mode):
    # Dense hostile non-selected slots make missing/shifted masks observable.
    rng = np.random.default_rng(7103)
    for source in (np.arange(65536,dtype=np.float64)+1, rng.normal(size=65536)):
        original=source.tobytes();inverse=mode=='inverse';gain=.125 if mode=='scaled' else 1.
        actual=plan.evaluate(source,tile,offset,inverse=inverse,gain=gain)
        expected=np.zeros(65536)
        for row in range(16):
            key=16*tile+(row+offset)%16
            for head in range(12):
                hidden=row*4096+64*head;score=row*4096+16*key+head
                if inverse:expected[hidden]=source[score]
                else:expected[score]=gain*source[hidden]
        np.testing.assert_array_equal(actual,expected)
        assert source.tobytes()==original

def test_plan_and_existing_keys():
    p=plan.public_plan();assert len(p['key_inventory']['original'])==66
    assert len(p['key_inventory']['proposed'])==36
    assert p['full_k128_routing_maps']==24576
    assert p['saved_internal_rotations']==368640 and p['extra_rescale_calls']==122880
    assert p['proposed_total_rotations_excluding_bootstrap']==538189

@pytest.mark.parametrize('change', ('shape','dtype','nonfinite','direction','gain','inverse_gain'))
def test_public_contract_rejection(change):
    value=np.ones(65536);kwargs={}
    if change=='shape':value=value[:-1]
    if change=='dtype':value=value.astype(np.float32)
    if change=='nonfinite':value[1]=np.nan
    if change=='direction':kwargs['inverse']=1
    if change=='gain':kwargs['gain']=True
    if change=='inverse_gain':kwargs.update(inverse=True,gain=.125)
    with pytest.raises(ValueError):plan.evaluate(value,0,0,**kwargs)

def test_no_backend_imported():
    assert not any(n=='jax' or n.startswith(('jax.','jaxite_word','torch','transformers')) for n in sys.modules)
