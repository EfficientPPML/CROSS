"""Actual chained dense loops on fixed synthetic CLS rows; no full model claim."""
from pathlib import Path
import hashlib,json,math,os
import numpy as np
import pytest
from ideal_refresh import ideal_primitives,primitives,IdentityRefresh
from ideal_primitives import Arithmetic,Value,IdealLinear,DELTA
from encrypted_bert_pooler_head import BertPoolerHead,TENSORS
from test_tanh import RAW,oracle
HERE=Path(__file__).resolve().parent
MODEL=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c/model.safetensors')
MODEL_SHA='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
@pytest.fixture(scope='module')
def arrays():
    assert sha(MODEL)==MODEL_SHA;pins=json.loads((HERE/'head_tensor_pins.json').read_text());out={}
    with MODEL.open('rb') as f:
        n=int.from_bytes(f.read(8),'little');header=json.loads(f.read(n));base=n+8
        for name,(shape,digest) in TENSORS.items():
            pin=pins[name];desc=header[pin['source']];assert desc['dtype']=='F32' and desc['shape']==list(shape)
            a,b=desc['data_offsets'];f.seek(base+a);raw=f.read(b-a);assert hashlib.sha256(raw).hexdigest()==digest==pin['f32le_sha256']
            out[name]=np.frombuffer(raw,dtype='<f4').astype(np.float64).reshape(shape)
    yield out
    assert sha(MODEL)==MODEL_SHA

def setup(arrays,positions=(0,),progress=None):
    a=Arithmetic();r=IdentityRefresh(a);linear=IdealLinear(a);cache=a.ctx._param_cache
    for name in ('pooler','classifier'):
        for key in linear.required_rotation_indices(arrays[name+'.weight'],linear.layout):cache.raw_rotation_keys[key]={'a':object(),'b':object()};cache.coef_maps[key]=object()
    return BertPoolerHead(a,linear,r,arrays,RAW,original_positions=positions,progress=progress),a,r
@pytest.mark.parametrize('positions',((0,),tuple(range(0,32,2))))
@pytest.mark.parametrize('phase',(0.,.713))
@pytest.mark.parametrize('scale',(math.nextafter(DELTA,-math.inf),math.nextafter(DELTA,math.inf)))
def test_full_head_real_dense_loops_all_slots(arrays,positions,phase,scale):
    owner,a,r=setup(arrays,positions);x=np.cos(np.arange(768)*.071+phase)*.35
    data=np.full((16,4096),10000.);data[::2]*=-1.;data[0,:768]=x;source=Value(data.ravel(),12,scale)
    result=owner.evaluate_tile(source,capture_checkpoints=True)
    pooled=x@arrays['pooler.weight'].T+arrays['pooler.bias'];assert np.all(np.abs(pooled)<=34)
    activated,_=oracle(pooled);logits=activated@arrays['classifier.weight'].T+arrays['classifier.bias']
    expected=np.zeros(65536);expected[:2]=logits;error=float(np.max(np.abs(result.output.data-expected)))
    assert error<1e-10 and result.output.level==2 and len(r.calls)==3
    assert [x['level'] for x in result.plan['stages']]==[11,3,2]
    np.testing.assert_allclose(result.checkpoints['pooler_dense'].data[:768],pooled,atol=1e-12,rtol=0)
    np.testing.assert_array_equal(source.data,data.ravel())
    for v in result.checkpoints.values():assert not any(n.startswith('_public_') for n in vars(v))
    dest=os.environ.get('HEAD_CASE_REPORT_DIR')
    if dest:
        p=Path(dest);p.mkdir(exist_ok=True)
        (p/f'{len(positions)}-{phase}-{scale.hex()}.json').write_text(json.dumps(dict(positions=positions,phase=phase,scale=scale,slots=65536,max_abs=error,gate=1e-10,plan=result.plan,rotate_calls=a.calls.count('rotate'),ct_products=a.calls.count('multiply')),indent=2)+'\n')
@pytest.mark.parametrize('kind',('positions','missing_cls','float_position','weight','layout','linear_bound','input_level'))
def test_public_entry_rejections(arrays,kind):
    if kind in ('positions','missing_cls','float_position'):
        positions={'positions':(0,2,1),'missing_cls':(1,),'float_position':(0.,)}[kind]
        with pytest.raises(ValueError):setup(arrays,positions)
        return
    if kind=='weight':
        changed=dict(arrays);changed['pooler.bias']=arrays['pooler.bias'].copy();changed['pooler.bias'][0]+=1.
        with pytest.raises(ValueError):setup(changed)
        return
    owner,a,r=setup(arrays);source=Value(np.zeros(65536),12)
    if kind=='layout':source.degree_layout=(512,256)
    elif kind=='linear_bound':owner.linear.arithmetic.declare_bound=lambda *args:None
    else:source=Value(np.zeros(65536),8)
    with pytest.raises(ValueError):owner.evaluate_tile(source)
    assert not a.calls and not r.calls
@pytest.mark.parametrize('event',('head pooler dense','head round3.output','head classifier dense'))
@pytest.mark.parametrize('kind',('source','current','captured','positions'))
def test_head_callbacks_stop_before_next_operation(arrays,event,kind):
    source=Value(np.zeros(65536),12);seen=[];at=[]
    def progress(name,result,counts):
        if name=='head pooler dense':seen.append(result)
        if name!=event:return
        at.append(len(a.calls))
        if kind=='source':source._ckks_nsd=True
        elif kind=='current':result._ckks_nsd=True
        elif kind=='captured':seen[0]._ckks_nsd=True
        else:owner.original_positions=(0.,)
    owner,a,r=setup(arrays,progress=progress)
    with pytest.raises(ValueError):owner.evaluate_tile(source,capture_checkpoints=True)
    assert at==[len(a.calls)] and not owner._active and owner.last_run is None

def test_classifier_masks_bounded_refresh_noise_in_other_rows_and_features(arrays):
    owner,a,r=setup(arrays,tuple(range(16)));x=np.cos(np.arange(768)*.071)*.35
    data=np.full((16,4096),-9999.);data[0,:768]=x
    def inject(source,result):
        # Deliberate new errors on inactive slots; the actual classifier's
        #source mask, not an ideal refresh silently restoring padding, removes them.
        result.data[768:]=np.where(np.arange(65536-768)%2,1./64.,-1./64.)
    r.inject=inject
    output=owner.evaluate_tile(Value(data.ravel(),12)).output.data
    pooled=x@arrays['pooler.weight'].T+arrays['pooler.bias'];activated,_=oracle(pooled)
    expected=np.zeros(65536);expected[:2]=activated@arrays['classifier.weight'].T+arrays['classifier.bias']
    np.testing.assert_allclose(output,expected,atol=1e-10,rtol=0)
    assert np.count_nonzero(output[2:])==0
