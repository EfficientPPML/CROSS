"""Independent plaintext/key-plan/lifetime contracts, no encrypted claim."""
from pathlib import Path
from types import SimpleNamespace
import gc
import sys
import weakref
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
from encrypted_streaming_linear import StreamingLinear,RowTileLayout
from encrypted_packed_qwen import MatrixDiagonals
import encrypted_packed_transformer as module
from encrypted_packed_transformer import PackedTransformer,giant_chunks
from encrypted_chained_streaming_linear import (ChainedStreamingLinear,linear_rotation_plan,validate_rotation_strategy,require_chained_rotation_keys)

class Value:
    def __init__(self,data,level=4):
        self.slots=np.asarray(data);self.level=level;self._ckks_scale=1.;self._ckks_nsd=1
        # Simulate public payload metadata only, allowing multi-pending branch.
        self.polynomial=SimpleNamespace(shape=(1,2,8,8,2*(level+1)),dtype='uint32')

class Arithmetic:
    def __init__(self,slots,keys):
        self.events=[];self.refresher=None;self.results=[];self.closed=0
        self.ctx=SimpleNamespace(num_slots=slots,secret_key=None,parameters={},
            _param_cache=SimpleNamespace(secret_key=None,per_level_rotation_keys=False,
                raw_rotation_keys={k:{'a':object(),'b':object()} for k in keys},
                coef_maps={k:object() for k in keys},num_q_at_level=lambda level:2*(level+1)))
        self.ctx.he_add=SimpleNamespace()
        self.ctx.he_add=AddAccessor(self)
    def _level(self,v):return v.level
    def _ensure_capacity(self,values,required):
        self.events.append(('capacity',required));assert all(v.level>=required for v in values)
        return tuple(values)
    def _prepare(self,level):self.events.append(('prepare',level))
    def _finish(self,value):return value
    def _encode_public(self,slots,level,scale):return Value(slots,level)
    def weighted_sum(self,values,weights,bias=0.):
        values,weights=tuple(values),tuple(weights)
        output=Value(sum(v.slots*w for v,w in zip(values,weights))+bias,min(v.level for v in values)-1)
        self.events.append(('weighted',output.slots.tobytes(),output.level))
        self.results.append(weakref.ref(output))
        return output
    def rotate(self,value,index):
        self.events.append(('rotate',index%self.ctx.num_slots,value.level))
        return Value(np.roll(value.slots,-index),value.level)
    def add(self,left,right):
        self.events.append(('add',min(left.level,right.level)))
        return Value(left.slots+right.slots,min(left.level,right.level))
    def _clear_working_caches(self):self.closed+=1

class AddAccessor:
    def __init__(self,arithmetic):self.arithmetic=arithmetic
    def __getitem__(self,level):return SimpleNamespace(add_plain=self.arithmetic.add)

class IdealLinear(ChainedStreamingLinear):
    def __init__(self,layout,weight,strategy='chained'):
        self.layout=layout;self.test_strategy=strategy
        keys=self.required_rotation_indices(weight,layout,rotation_strategy=strategy)
        self.arithmetic=Arithmetic(layout.slots,keys);self.ctx=self.arithmetic.ctx
    @staticmethod
    def required_rotation_indices(weight,layout,*,rotation_strategy='chained'):
        cls=ChainedStreamingLinear if rotation_strategy=='chained' else StreamingLinear
        return cls.required_rotation_indices(weight,layout)
    def _validate_ciphertext(self,value):assert isinstance(value,Value)
    def linear_tile(self,*a,**k):
        if self.test_strategy=='direct':return StreamingLinear.linear_tile(self,*a,**k)
        return super().linear_tile(*a,**k)

@pytest.mark.parametrize('shape,expected,direct_count',[
    ((768,768),{1,64,64064},86),((3072,768),{1,64,61760},122),((768,3072),{1,64,61760},122)])
def test_fullwidth_dense_public_key_inventory(shape,expected,direct_count):
    weight=np.ones(shape);layout=RowTileLayout(65536)
    source=MatrixDiagonals(weight,layout.slots,layout.features)
    assert source.n1==64
    direct=linear_rotation_plan(source)
    chained=linear_rotation_plan(source,strategy='chained')
    assert set(direct.required_keys)==source.rotations and len(direct.required_keys)==direct_count
    assert set(chained.required_keys)==expected
    assert direct.nonzero_rotation_calls==chained.nonzero_rotation_calls==direct_count
    assert StreamingLinear.required_rotation_indices(weight,layout)==direct.required_keys
    assert ChainedStreamingLinear.required_rotation_indices(weight,layout)==chained.required_keys

@pytest.mark.parametrize('shape',[(7,9),(9,7),(5,5)])
@pytest.mark.parametrize('sparse',[False,True])
@pytest.mark.parametrize('limit',[1,3,8])
def test_exact_plain_matrix_oracle_and_unchanged_inputs(shape,sparse,limit,monkeypatch):
    # Full physical rows differ; hostile padding and signed per-source masks.
    rng=np.random.default_rng(771+shape[0]+sparse)
    layout=RowTileLayout(64,features=16);valid=3
    weight=rng.normal(size=shape)
    if sparse:weight[rng.random(shape)<.7]=0
    bias=rng.normal(size=shape[0]);mask=rng.uniform(-1,1,64);mask[::5]=0
    raw=rng.normal(size=(4,16));raw[3]=1e9;raw[:3,shape[1]:]=1e8
    before=[a.copy() for a in (raw,weight,bias,mask)]
    monkeypatch.setattr(module,'pending_limit',lambda *a,**k:limit)
    expected=np.zeros((4,16));expected[:valid,:shape[0]]=(raw[:valid,:shape[1]]*mask.reshape(4,16)[:valid,:shape[1]])@weight.T+bias
    outputs=[];counts=[]
    for strategy in ('direct','chained'):
        worker=IdealLinear(layout,weight,strategy)
        result=worker.linear_tile(Value(raw.ravel()),weight,bias,valid_rows=valid,input_mask=mask)
        np.testing.assert_allclose(result.slots.reshape(4,16),expected,rtol=0,atol=3e-14)
        assert result.level==3
        rotations=[e for e in worker.arithmetic.events if e[0]=='rotate' and e[1]]
        assert {e[1] for e in rotations}==set(worker.required_rotation_indices(weight,layout,rotation_strategy=strategy))
        assert len(rotations)==linear_rotation_plan(MatrixDiagonals(weight,64,16),strategy=strategy).nonzero_rotation_calls
        assert sum(e[0]=='capacity' for e in worker.arithmetic.events)==1
        outputs.append(result.slots);counts.append(len(rotations))
    assert counts[0]==counts[1]
    np.testing.assert_allclose(outputs[0],outputs[1],rtol=0,atol=3e-14)
    for a,b in zip((raw,weight,bias,mask),before):np.testing.assert_array_equal(a,b)

@pytest.mark.parametrize('active,n1',[
    ((3,7,35,43,67,103),8),((33,),8),((3,),8),((0,),8),((),8),
    ((7,31,47,95,119),16),((35,43,99,107),8)])
def test_sparse_actual_gap_schedule_not_assumed_dense(active,n1):
    source=SimpleNamespace(active=active,n1=n1,dimension=128)
    direct=linear_rotation_plan(source);chain=linear_rotation_plan(source,strategy='chained')
    assert direct.nonzero_rotation_calls==chain.nonzero_rotation_calls
    # Independent symbolic rotation of each giant's token recovers its direct power.
    contributions={g:{g:0} for g in chain.giant_indices}
    result={};previous=None
    for g in chain.giant_indices:
        if result:result={origin:(shift+previous-g)%128 for origin,shift in result.items()}
        result.update(contributions[g]);previous=g
    result={origin:(shift+chain.final_giant)%128 for origin,shift in result.items()}
    assert result=={g:g%128 for g in chain.giant_indices}
    position=0
    for index,step in zip(chain.baby_indices,chain.baby_steps):position+=step;assert position==index
    assert set(chain.required_keys)=={x%128 for x in (*chain.baby_steps,*chain.giant_steps,chain.final_giant) if x%128}

@pytest.mark.parametrize('bad',[None,True,1,0,[],{},b'direct','DIRECT',' chained',np.str_('direct')])
def test_strategy_requires_exact_known_string_before_work(bad):
    with pytest.raises(ValueError,match='rotation_strategy'):validate_rotation_strategy(bad)
    with pytest.raises(ValueError,match='rotation_strategy'):
        linear_rotation_plan(MatrixDiagonals(np.eye(3),64,16),strategy=bad)
    with pytest.raises(TypeError):
        ChainedStreamingLinear(SimpleNamespace(),None,rotation_strategy=bad)

@pytest.mark.parametrize('mutation',[
    'missing_key','missing_map','missing_a','missing_b','key_not_mapping','per_level','secret_ctx','secret_cache','secret_parameters','implicit_refresh'])
def test_chained_missing_public_key_or_keyless_context_fails_before_any_arithmetic(mutation):
    weight=np.ones((7,9));worker=IdealLinear(RowTileLayout(64,16),weight)
    cache=worker.ctx._param_cache;key=next(iter(cache.raw_rotation_keys))
    if mutation=='missing_key':cache.raw_rotation_keys.pop(key)
    elif mutation=='missing_map':cache.coef_maps.pop(key)
    elif mutation=='missing_a':cache.raw_rotation_keys[key].pop('a')
    elif mutation=='missing_b':cache.raw_rotation_keys[key].pop('b')
    elif mutation=='key_not_mapping':cache.raw_rotation_keys[key]=object()
    elif mutation=='per_level':cache.per_level_rotation_keys=True
    elif mutation=='secret_ctx':worker.ctx.secret_key=object()
    elif mutation=='secret_cache':cache.secret_key=object()
    elif mutation=='secret_parameters':worker.ctx.parameters['secret_key']=object()
    else:worker.arithmetic.refresher=object()
    with pytest.raises(ValueError):worker.linear_tile(Value(np.ones(64)),weight,valid_rows=3)
    assert worker.arithmetic.events==[]

@pytest.mark.parametrize('failure',['inner','rotate','add'])
def test_failed_chained_queue_releases_all_inner_wrappers_and_keeps_inputs(monkeypatch,failure):
    rng=np.random.default_rng(19);weight=rng.normal(size=(15,15))
    worker=IdealLinear(RowTileLayout(64,16),weight)
    value=Value(rng.normal(size=64));before=value.slots.copy()
    monkeypatch.setattr(module,'pending_limit',lambda *a,**k:8)
    if failure=='inner':
        original=worker.arithmetic.weighted_sum
        def operation(*a,**k):
            if len(worker.arithmetic.results)==2:raise RuntimeError('inner')
            return original(*a,**k)
        worker.arithmetic.weighted_sum=operation
    elif failure=='rotate':
        original=worker.arithmetic.rotate
        def operation(v,index):
            if v.level==3 and index%64:raise RuntimeError('rotate')
            return original(v,index)
        worker.arithmetic.rotate=operation
    else:
        worker.arithmetic.add=lambda *a,**k:(_ for _ in ()).throw(RuntimeError('add'))
    with pytest.raises(RuntimeError,match=failure):worker.linear_tile(value,weight,valid_rows=3)
    gc.collect()
    assert worker.arithmetic.results and all(ref() is None for ref in worker.arithmetic.results)
    np.testing.assert_array_equal(value.slots,before)
    worker.clear();assert worker.arithmetic.closed==1
    assert worker.ctx._param_cache.raw_rotation_keys


@pytest.mark.parametrize('diagonals',[(-11,),(-11,-7), (3,7),(-9,-4,5,11),(0,)])
def test_sparse_weight_routes_with_nonzero_min_giant_and_missing_zero_baby(diagonals):
    weight=np.zeros((15,15))
    for d in diagonals:
        for row in range(15):
            col=row+d
            if 0<=col<15:weight[row,col]=(row+1)/16
    worker=IdealLinear(RowTileLayout(64,16),weight)
    data=np.arange(64,dtype=float).reshape(4,16)/32
    result=worker.linear_tile(Value(data.ravel()),weight,valid_rows=3)
    expected=np.zeros_like(data);expected[:3,:15]=data[:3,:15]@weight.T
    np.testing.assert_allclose(result.slots.reshape(4,16),expected,atol=1e-13,rtol=0)
    assert result.level==3


def rectangular_matrix(outputs,inputs):
    weight=np.zeros((outputs,inputs));row=np.arange(outputs)
    weight[row,row%inputs]=.5;weight[row,(row+1)%inputs]+=-.25
    for shift in (768,1536,2304):
        source=row+shift;valid=source<inputs;weight[row[valid],source[valid]]+=.125
    for target,source in ((0,inputs-1),(outputs-1,0),(min(63,outputs-1),min(64,inputs-1)),(min(64,outputs-1),min(63,inputs-1))):
        weight[target,source]+=.0625
    return weight


@pytest.mark.parametrize('strategy',['direct','chained'])
@pytest.mark.parametrize('slots',[8192,65536])
@pytest.mark.parametrize('outputs,inputs',[(768,768),(3072,768),(768,3072)])
@pytest.mark.parametrize('partial',[False,True])
def test_all_physical_slots_fullwidth_rectangular_and_signed_mask(strategy,slots,outputs,inputs,partial):
    layout=RowTileLayout(slots);valid=layout.rows_per_tile-1 if partial else layout.rows_per_tile
    data=(np.arange(slots,dtype=float).reshape(layout.rows_per_tile,4096)%17-8)*1e6
    active=(np.arange(valid)[:,None]+1)*8+(np.arange(inputs)[None,:]%31)/32
    data[:valid,:inputs]=active;before=data.copy()
    mask=np.where(np.arange(slots)%3,.5,-1.).reshape(data.shape);mask[:,::7]=0.
    weight=rectangular_matrix(outputs,inputs);bias=(np.arange(outputs)%13-6)/64
    worker=IdealLinear(layout,weight,strategy)
    result=worker.linear_tile(Value(data.ravel()),weight,bias,valid_rows=valid,input_mask=mask.ravel())
    expected=np.zeros_like(data);expected[:valid,:outputs]=(active*mask[:valid,:inputs])@weight.T+bias
    np.testing.assert_allclose(result.slots.reshape(data.shape),expected,atol=1e-10,rtol=1e-13)
    np.testing.assert_array_equal(data,before);assert result.level==3


@pytest.mark.parametrize('strategy',['direct','chained'])
def test_zero_matrix_masks_bias_to_requested_rows_without_rotation(strategy):
    layout=RowTileLayout(8192);weight=np.zeros((3,5));bias=np.array([1.,-2.,3.])
    worker=IdealLinear(layout,weight,strategy)
    result=worker.linear_tile(Value(np.full(8192,1e9)),weight,bias,valid_rows=1)
    expected=np.zeros((2,4096));expected[0,:3]=bias
    np.testing.assert_array_equal(result.slots.reshape(expected.shape),expected)
    assert result.level==3 and not any(e[0]=='rotate' and e[1] for e in worker.arithmetic.events)


def test_borrowed_ownership_validation_and_stream_contract_are_inherited_unchanged():
    for name in ('__init__','_validate_ciphertext','iter_evaluate','clear'):
        assert getattr(ChainedStreamingLinear,name) is getattr(StreamingLinear,name)


@pytest.mark.parametrize('bad',[None,True,1.,np.int64(1),0,2])
def test_explicit_exact_integer_nsd1_rejected_before_arithmetic(bad):
    worker=IdealLinear(RowTileLayout(64,16),np.eye(3));value=Value(np.ones(64));value._ckks_nsd=bad
    with pytest.raises(ValueError,match='noise-scale'):worker.linear_tile(value,np.eye(3),valid_rows=1)
    assert worker.arithmetic.events==[]


@pytest.mark.parametrize('bad',[None,False,np.bool_(True),0.,-1.,float('nan'),float('inf'),'1',1+0j,10**500])
def test_scale_must_be_finite_positive_real_before_arithmetic(bad):
    worker=IdealLinear(RowTileLayout(64,16),np.eye(3));value=Value(np.ones(64));value._ckks_scale=bad
    with pytest.raises(ValueError,match='finite positive'):worker.linear_tile(value,np.eye(3),valid_rows=1)
    assert worker.arithmetic.events==[]


@pytest.mark.parametrize('scale',[.5,2.,np.float64(7.),2**56])
def test_noncanonical_positive_actual_scale_is_not_rejected(scale):
    worker=IdealLinear(RowTileLayout(64,16),np.eye(3));value=Value(np.ones(64));value._ckks_scale=scale
    result=worker.linear_tile(value,np.eye(3),valid_rows=1)
    assert result.level==3 and worker.arithmetic.events


def test_new_module_import_is_backend_free():
    import subprocess,os
    code="import sys;sys.path.insert(0,sys.argv[1]);from encrypted_chained_streaming_linear import ChainedStreamingLinear;assert 'jax' not in sys.modules;assert 'jaxite_word' not in sys.modules"
    result=subprocess.run([sys.executable,'-c',code,str(ROOT/'cadathlon/common')],env=dict(os.environ,JAX_PLATFORMS='no_backend_allowed'),text=True,capture_output=True)
    assert result.returncode==0,result.stdout+result.stderr


@pytest.mark.parametrize('text',['1e-4000','1e4000'])
def test_wide_scale_underflow_and_overflow_rejected_without_metadata_mutation(text):
    if np.finfo(np.longdouble).maxexp<=np.finfo(np.float64).maxexp:
        pytest.skip('platform longdouble has no wider exponent range')
    worker=IdealLinear(RowTileLayout(64,16),np.eye(3));value=Value(np.ones(64));scale=np.longdouble(text);value._ckks_scale=scale
    with pytest.raises(ValueError,match='finite positive'):worker.linear_tile(value,np.eye(3),valid_rows=1)
    assert worker.arithmetic.events==[] and value._ckks_scale is scale


@pytest.mark.parametrize('zero',[False,True])
def test_clone_faithful_stale_bound_tags_discarded_on_outputs_only(zero):
    weight=np.zeros((3,5)) if zero else np.arange(15,dtype=float).reshape(3,5)/16
    worker=IdealLinear(RowTileLayout(64,16),weight);value=Value(np.arange(64,dtype=float)/64)
    tags={'_public_absolute_bound':99.,'_public_row_l2':{'bound':13.},'_public_head_energy':17.,'_public_affine':object()}
    for name,tag in tags.items():setattr(value,name,tag)
    before=value.slots.copy()
    # Actual Polynomial facade clones retain arbitrary public relation fields.
    # Reproduce that retention through every primitive, including zero/bias paths.
    for method in ('rotate','add','weighted_sum'):
        original=getattr(worker.arithmetic,method)
        def clone_retaining(*args,_original=original,_name=method,**kwargs):
            source=args[0][0] if _name=='weighted_sum' else args[0]
            result=_original(*args,**kwargs)
            for name,tag in vars(source).items():
                if name.startswith('_public_'):setattr(result,name,tag)
            return result
        setattr(worker.arithmetic,method,clone_retaining)
    result=worker.linear_tile(value,weight,np.array([1.,2.,3.]),valid_rows=2)
    assert result is not value and not any(name.startswith('_public_') for name in vars(result))
    assert all(getattr(value,name) is tag for name,tag in tags.items())
    np.testing.assert_array_equal(value.slots,before)
