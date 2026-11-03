"""Independent ideal public-layout checks, not encrypted passes."""
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'checkout/cadathlon/common')]
from encrypted_score_reduction import ScoreReductionLayout,ScoreReduction,validate_public_dummy,SLOTS,SHIFTS
from fixtures import public_fixtures,make_client_values,make_client_scores,client_oracles,mean_variation


class Value:
    batch=1;degree=2*SLOTS;num_elements=2;_ckks_nsd=1;_ckks_scale=2.**56
    def __init__(self,data,level=4):
        self.data=np.asarray(data).copy();self.level=level;self._public_row_l2_bound='stale'


class Arithmetic:
    scale=2.**56;refresher=None
    def __init__(self):
        self.ctx=SimpleNamespace(degree=2*SLOTS,num_slots=SLOTS,composite_degree=2,
            secret_key=None,parameters={},_param_cache=SimpleNamespace(secret_key=None,batch=1))
        self.calls=[]
    def _level(self,value):
        if value._ckks_nsd!=1:raise ValueError('invalid NSD')
        return value.level
    def rotate(self,value,shift):
        self.calls.append('rotate');return Value(np.roll(value.data,-shift),value.level)
    def add(self,left,right):
        self.calls.append('add');return Value(left.data+right.data,min(left.level,right.level))
    def weighted_sum(self,values,weights,*,bias=0.):
        self.calls.append('weighted_sum')
        return Value(sum(x.data*w for x,w in zip(values,weights))+bias,min(x.level for x in values)-1)
    def multiply(self,left,right):
        self.calls.append('multiply');return Value(left.data*right.data,min(left.level,right.level)-1)


@pytest.fixture(autouse=True)
def fake_backend(monkeypatch):
    monkeypatch.setitem(sys.modules,'encrypted_packed_qwen',SimpleNamespace(CompiledArithmetic=Arithmetic))
    monkeypatch.setitem(sys.modules,'jaxite_word',SimpleNamespace(Polynomial=Value))
    monkeypatch.setitem(sys.modules,'support',SimpleNamespace(require_secure_execution=lambda *a:None))


def independent_dense(values,keys,valid_rows,*,square,dummy):
    out=np.full(SLOTS,dummy if square else 0.)
    keys=np.broadcast_to(keys,(16,128))
    for row in range(valid_rows):
        selected=np.flatnonzero(keys[row])
        for h in range(12):
            data=values[row*4096+selected*16+h]
            expected=np.dot(data,data) if square else data.sum()/len(selected)
            out[row*4096+np.arange(128)*16+h]=expected
    return out


@pytest.mark.parametrize('valid_rows',(1,7,16))
@pytest.mark.parametrize('pattern',('all','first15','interleaved','last','boundaries','perrow'))
def test_hostile_padding_independent_dense(pattern,valid_rows):
    k=np.arange(128)
    keys={'all':np.ones(128,bool),'first15':k<15,'interleaved':k%3==0,'last':k==127,
          'boundaries':np.isin(k,[0,15,16,63,64,111,112,127]),
          'perrow':np.asarray([k%(r+2)==r%(r+2) for r in range(16)])}[pattern]
    values=np.random.default_rng(1000+valid_rows).uniform(-20.,20.,SLOTS)
    layout=ScoreReductionLayout(keys,valid_rows=valid_rows);arith=Arithmetic();graph=ScoreReduction(arith,layout)
    original=Value(values);mean=graph.mean(original)
    denominator=graph.squared_denominator(original,domain=(.003,2.),inactive_dummy=.271)
    np.testing.assert_allclose(mean.data,independent_dense(values,keys,valid_rows,square=False,dummy=0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(denominator.data,independent_dense(values,keys,valid_rows,square=True,dummy=.271),rtol=1e-14,atol=1e-11)
    assert mean.level==2 and denominator.level==1 and not hasattr(mean,'_public_row_l2_bound')
    assert not hasattr(denominator,'_public_row_l2_bound')
    assert original._public_row_l2_bound=='stale';np.testing.assert_array_equal(original.data,values)
    assert graph.counts==dict(public_multiplications=4,actual_composite_rescales=5,path_levels=5,
        rotations=28,ciphertext_additions=28,ciphertext_multiplications=1,public_bias_additions=1)


@pytest.mark.parametrize('public',public_fixtures(),ids=lambda x:x['name'])
def test_actual_proof_public_fixtures_and_domain(public):
    values=make_client_values(public);scores=make_client_scores(public);reference=client_oracles(scores,values,public)
    keys=np.isin(np.arange(128),public['valid_key_indices'])
    layout=ScoreReductionLayout(keys,valid_rows=public['valid_rows'])
    graph=ScoreReduction(Arithmetic(),layout);ct=Value(values)
    mean=graph.mean(Value(scores));denominator=graph.squared_denominator(ct,domain=public['domain'],inactive_dummy=public['inactive_dummy'])
    np.testing.assert_allclose(mean.data,reference['mean'],rtol=0,atol=1e-15)
    np.testing.assert_allclose(denominator.data,reference['squared_denominator'],rtol=0,atol=1e-15)
    assert not np.array_equal(values[:32768],values[32768:])
    assert np.max(np.abs(values[layout.input_mask==0]))>3.
    assert np.all((denominator.data>=public['domain'][0])&(denominator.data<=public['domain'][1]))
    assert 1.>public['domain'][1]
    variation=mean_variation(reference['mean'],public)
    assert variation['unique_active_means']==12*public['valid_rows']
    assert variation['minimum_within_row_range']>.1 and variation['minimum_across_rows_per_head_range']>.1


@pytest.mark.parametrize('row,key,head',[(0,0,0),(0,127,11),(7,63,5),(8,64,6),(15,127,11)])
def test_boundary_impulses(row,key,head):
    values=np.zeros(SLOTS);values[row*4096+key*16+head]=1.
    graph=ScoreReduction(Arithmetic(),ScoreReductionLayout(np.ones(128,bool)))
    out=graph.mean(Value(values)).data
    expected=np.zeros(SLOTS);expected[row*4096+np.arange(128)*16+head]=1/128
    np.testing.assert_array_equal(out,expected)


@pytest.mark.parametrize('dummy',[True,False,0.,-1.,float('nan'),float('inf'),1.,.0001,.101,complex(.02)])
def test_public_dummy_rejected_before_any_ciphertext(dummy):
    a=Arithmetic();graph=ScoreReduction(a,ScoreReductionLayout(np.ones(128,bool)))
    with pytest.raises(ValueError):graph.squared_denominator(None,domain=(.001,.1),inactive_dummy=dummy)
    assert not a.calls


@pytest.mark.parametrize('domain',[(0.,1.),(-1.,1.),(.1,.1),(1.,.1),(float('nan'),1.),(1.,float('inf')),(True,1.),(.1,),'.1,.2'])
def test_invalid_public_domain(domain):
    with pytest.raises(ValueError):validate_public_dummy(.15,domain)


@pytest.mark.parametrize('mutation',('secret','cache_secret','parameter_secret','refresher','cd','context_degree','context_batch','level','scale','nsd','batch','elements','degree'))
def test_guard_before_arithmetic(mutation):
    a=Arithmetic();g=ScoreReduction(a,ScoreReductionLayout(np.ones(128,bool)));v=Value(np.ones(SLOTS))
    if mutation=='secret':a.ctx.secret_key=object()
    elif mutation=='cache_secret':a.ctx._param_cache.secret_key=object()
    elif mutation=='parameter_secret':a.ctx.parameters['secret_key']=object()
    elif mutation=='refresher':a.refresher=object()
    elif mutation=='cd':a.ctx.composite_degree=1
    elif mutation=='context_degree':a.ctx.degree=16384
    elif mutation=='context_batch':a.ctx._param_cache.batch=2
    elif mutation=='level':v.level=2
    elif mutation=='scale':v._ckks_scale=123.
    elif mutation=='nsd':v._ckks_nsd=2
    elif mutation=='batch':v.batch=2
    elif mutation=='elements':v.num_elements=3
    elif mutation=='degree':v.degree=16384
    with pytest.raises((TypeError,ValueError)):g.squared_denominator(v,domain=(.01,1.),inactive_dummy=.2)
    assert not a.calls


def test_owner_type_and_bound_aware_rejected():
    with pytest.raises(TypeError):ScoreReduction(object(),ScoreReductionLayout(np.ones(128,bool)))
    a=Arithmetic();a.declare_bound=lambda *a:None
    with pytest.raises(TypeError):ScoreReduction(a,ScoreReductionLayout(np.ones(128,bool)))


@pytest.mark.parametrize('mask,rows',[(np.ones(128),16),(np.ones(127,bool),16),(np.zeros(128,bool),16),(np.ones(128,bool),0),(np.ones(128,bool),17),(np.ones(128,bool),True)])
def test_bad_layout(mask,rows):
    with pytest.raises(ValueError):ScoreReductionLayout(mask,valid_rows=rows)


def test_public_weights_immutable_and_rotations_exact():
    layout=ScoreReductionLayout(np.ones(128,bool))
    for field in ('key_mask','counts','input_mask','mean_weight','anchors','dummy_anchors'):
        with pytest.raises(ValueError):getattr(layout,field).reshape(-1)[0]=0
    assert set(layout.rotation_indices())=={s%SLOTS for x in SHIFTS for s in (x,-x)}
    assert len(layout.rotation_indices())==14


def test_missing_noise_degree_is_rejected_explicitly(monkeypatch):
    monkeypatch.delattr(Value,'_ckks_nsd')
    a=Arithmetic();g=ScoreReduction(a,ScoreReductionLayout(np.ones(128,bool)))
    with pytest.raises(ValueError,match='explicit integer noise-scale'):g.mean(Value(np.ones(SLOTS)))
    assert not a.calls


@pytest.mark.parametrize('nsd',[True,1.0,np.int64(1)])
def test_noninteger_noise_degree_one_rejected(nsd):
    a=Arithmetic();g=ScoreReduction(a,ScoreReductionLayout(np.ones(128,bool)));v=Value(np.ones(SLOTS));v._ckks_nsd=nsd
    with pytest.raises(ValueError,match='explicit integer noise-scale'):g.mean(v)
    assert not a.calls
