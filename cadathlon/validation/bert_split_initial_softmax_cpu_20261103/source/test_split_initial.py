"""Independent all-slot prefix/tail tests; no encrypted arithmetic or real refresh."""
from pathlib import Path
from collections import Counter
import copy,json,math
import numpy as np
import pytest
from test_initial_ideal import ideal_primitives,Q,Value as BaseValue
from test_softmax_tile import refresh_and_tracked
from ideal_primitives import Arithmetic,Value,DELTA
from ideal_refresh import IdentityRefresh,install_public_bootstrap_plan
from encrypted_score_reduction import ScoreReductionLayout
from encrypted_bert_split_initial import BertSplitInitialSoftmax,LayerInitialContract,prefix_scale_plan,tail_scale_plan,FIELDS
HERE=Path(__file__).resolve().parent;RAW=(HERE/'candidate_manifest.json').read_bytes()

@pytest.fixture(autouse=True)
def primitives(refresh_and_tracked,monkeypatch):
    install_public_bootstrap_plan(monkeypatch)
    # Earlier arithmetic/refresh doubles return this base slot class. Supply
    # the real profile's immutable public layout metadata, not fake RNS data.
    monkeypatch.setattr(BaseValue,'degree_layout',(256,512),raising=False)

def setup(layer,K,valid=16,progress=None):
    ops=Arithmetic();refresh=IdentityRefresh(ops)
    keys=np.arange(128);mask=np.array([((keys+13*row+17)%128)<K for row in range(16)])
    layout=ScoreReductionLayout(mask,valid_rows=valid)
    positions=tuple(range(128-valid,128))
    owner=BertSplitInitialSoftmax(ops,layout,RAW,refresh,layer=layer,original_positions=positions,progress=progress)
    return owner,ops,refresh,layout

def fixture(owner,layout,K):
    values=np.resize(np.array([-4096.,4096.]),65536).reshape(16,4096)
    amplitude=.15*min(-owner.contract.centered_domain[0],owner.contract.centered_domain[1])
    for row in range(layout.valid_rows):
        keys=np.flatnonzero(layout.key_mask[row])
        for head in range(12):
            values[row,keys*16+head]=row*23+head*7.25+amplitude*np.sin((keys+1)*(head+3))
    return values.reshape(-1)

def oracle(values,owner,layout,*,invalid_t=1.75):
    mean=np.zeros(65536);t=np.zeros(65536);base=np.zeros(65536);den=np.ones(65536)
    coeff=owner.contract.power_coefficients
    for row in range(layout.valid_rows):
        keys=np.flatnonzero(layout.key_mask[row])
        for head in range(12):
            chosen=row*4096+keys*16+head;broadcast=row*4096+np.arange(128)*16+head
            mu=float(np.mean(values[chosen]));centered=values[chosen]-mu
            lo,hi=owner.contract.centered_domain
            assert centered.min()>=lo and centered.max()<=hi
            mean[broadcast]=mu;t[chosen]=centered/owner.contract.normalized_divisor
            base[chosen]=np.polynomial.polynomial.polyval(t[chosen],coeff)
            den[broadcast]=base[chosen].mean()
    assert np.all((den>=.5)&(den<=2.))
    inverse=den*(-.8*.8)+2*.8
    for _ in range(5):inverse*=2-den*inverse
    probabilities=base*inverse*layout.mean_weight
    refreshed=t.copy();refreshed[layout.input_mask==0]=invalid_t
    return dict(score_mean=mean,normalized_t=t,refreshed_normalized_t=refreshed,base_exp=base,
                mean_exp=den,inverse=inverse,probabilities=probabilities)

@pytest.mark.parametrize('layer',(0,4,1))
@pytest.mark.parametrize('K',range(1,129))
def test_all_public_keycounts_three_distinct_maps_and_every_physical_slot(layer,K):
    valid=(1,7,16)[K%3];owner,ops,refresh,layout=setup(layer,K,valid)
    data=fixture(owner,layout,K);source=Value(data,21,math.nextafter(DELTA,math.inf));saved=data.copy()
    source._public_borrowed='preserve'
    def inject(original,result):result.data[layout.input_mask==0]=1.75
    refresh.inject=inject
    expected=oracle(data,owner,layout)
    output,stages=owner.evaluate_tile(source,capture_checkpoints=True)
    for name,wanted in expected.items():
        assert np.isfinite(stages[name].data).all()
        np.testing.assert_allclose(stages[name].data,wanted,atol=1e-11,rtol=0,err_msg=name)
    assert output.level==6 and output._ckks_scale==owner.last_plan['tail']['final_scale']
    assert stages['score_mean'].level==19 and stages['normalized_t'].level==18 and stages['refreshed_normalized_t'].level==25
    assert stages['base_exp'].level==21 and stages['mean_exp'].level==19 and stages['inverse'].level==8
    assert np.all(output.data[layout.input_mask==0]==0)
    np.testing.assert_array_equal(source.data,saved);assert source._public_borrowed=='preserve'
    assert len(refresh.calls)==1 and refresh.calls[0][0]==18 and refresh.calls[0][2]==2.
    counts=Counter(ops.calls);assert counts['rotate']==28 and counts['multiply']==18
    assert counts['weighted_sum']==7 and counts['initial_affine']==1 and ops.public_multiplies==15
    assert not owner._active and not owner._sources and not owner._retained and not owner._captured
    assert not owner.last_plan['tail']['recenter_input'] and not owner.last_plan['tail']['extra_pre_power_mask']
    assert not any(name.startswith('_public_') for value in stages.values() for name in vars(value))

@pytest.mark.parametrize('layer',range(12))
def test_exact_centered_domain_endpoints_and_partial_query_layout(layer):
    owner,ops,refresh,layout=setup(layer,3,7);data=np.full(65536,-2048.)
    lo,hi=owner.contract.centered_domain
    for row in range(7):
        keys=np.flatnonzero(layout.key_mask[row])
        for head in range(12):
            endpoint=lo if head%2 else hi
            data[row*4096+keys*16+head]=[endpoint,-endpoint/2,-endpoint/2]
    output,stages=owner.evaluate_tile(Value(data,21),capture_checkpoints=True)
    expected=oracle(data,owner,layout,invalid_t=0.)
    np.testing.assert_allclose(output.data,expected['probabilities'],atol=1e-11,rtol=0)
    assert np.min(stages['normalized_t'].data[layout.input_mask!=0])>=lo/owner.contract.normalized_divisor
    assert np.max(stages['normalized_t'].data[layout.input_mask!=0])<=hi/owner.contract.normalized_divisor

def test_tail_is_an_independent_continuation_without_recentering_or_refresh():
    owner,ops,refresh,layout=setup(1,15,7)
    data=np.full(65536,1.5);data[layout.input_mask!=0]=.4
    source=Value(data,25,math.nextafter(DELTA,-math.inf));source._public_borrowed='keep'
    output,stages=owner.tail(source,capture_checkpoints=True)
    expected_base=np.polynomial.polynomial.polyval(.4,owner.contract.power_coefficients)
    np.testing.assert_allclose(stages['base_exp'].data[layout.input_mask!=0],expected_base,atol=1e-14,rtol=0)
    assert not np.isclose(expected_base,1.) and not refresh.calls
    assert Counter(ops.calls)['rotate']==14 and output.level==6
    assert source._public_borrowed=='keep'

@pytest.mark.parametrize('layer',range(12))
def test_public_contract_and_scale_plans_cover_all_keycounts(layer):
    contract=LayerInitialContract(RAW,layer)
    m=json.loads(RAW)['softmax'][str(layer)]
    assert contract.normalized_divisor==2**(m['squarings']-1)
    for K in range(1,129):
        for scale in (DELTA,math.nextafter(DELTA,math.inf),math.nextafter(DELTA,-math.inf)):
            pre=prefix_scale_plan(Q[:44],input_level=21,canonical_scale=DELTA,input_scale=scale)
            tail=tail_scale_plan(Q[:52],input_level=25,canonical_scale=DELTA,input_scale=scale)
            assert pre['levels']=={'score_mean':19,'normalized_t':18}
            assert tail['levels']=={'base_exp':21,'mean_exp':19,'inverse':8,'base_times_inverse':7,'probabilities':6}
            assert pre['consumed_levels']==3 and tail['consumed_levels']==19
            assert math.isfinite(tail['final_scale']) and tail['initial_updates']==6

@pytest.mark.parametrize('layer',(-1,12,True,False,1.,np.int64(1)))
def test_invalid_layer_rejected(layer):
    with pytest.raises(ValueError):LayerInitialContract(RAW,layer)
