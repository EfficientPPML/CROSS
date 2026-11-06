"""Public real-slot/structural primitive doubles; no backend or ciphertexts."""
from pathlib import Path
from types import SimpleNamespace as NS
from collections import Counter
import sys,json,copy,math,os
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'checkout/cadathlon/common'))
import encrypted_bert_normalization_gain4 as g
from encrypted_bert_normalization_round import NormalizationRoundContract
from encrypted_score_reduction import ScoreReductionLayout
from test_owner_ideal import independent,values_for
PROFILE=next(r for r in json.loads((HERE/'actual_profile_audit.json').read_bytes())['profiles']if r['name']=='depth49_p60_d8')
Q=tuple(PROFILE['q']);P=tuple(PROFILE['p']);D=PROFILE['canonical_scale'];RAW=(HERE/'candidate_manifest.json').read_bytes()

class Barrett:
    computation_format_is_standard=True
    def __init__(self,q):self.moduli=q
class Payload:
    dtype=np.dtype('uint32')
    def __init__(self,data,L):self.data=data;self.shape=(1,2,256,512,2*(L+1));self.ready=0
    def devices(self):return {'ideal-cpu'}
    def block_until_ready(self):self.ready+=1;return self
class Value:
    batch=1;num_elements=2;degree=131072;r=256;c=512;degree_layout=(256,512);precision=32
    def __init__(self,data,L=25,scale=D):
        self.data=np.array(data,dtype=np.float64,copy=True);self.level=L;self._ckks_scale=float(scale);self._ckks_nsd=1
        self.moduli=Q[:2*(L+1)];self.num_moduli=len(self.moduli);self.ntt_ctx=NS(moduli=self.moduli,ff_ctx=Barrett(self.moduli))
        self.polynomial=Payload(self.data,L);self._public_bound='borrowed tag'
class Helper:
    def __init__(self,ctx):self.ctx=ctx
    def clear(self,**kw):pass
class Arithmetic:
    scale=D;refresher=None;progress=None;low_memory=True
    def __init__(self):
        self.calls=[];self.encodings=[]
        keys={k:{n:NS(shape=(8,114,131072),dtype=np.dtype('uint32'))for n in ('a','b')}for k in ScoreReductionLayout.rotation_indices()}
        cache=NS(secret_key=None,batch=1,q_towers=list(Q),p_towers=list(P),dnum=8,degree_layout=(256,512),composite_degree=2,ff_context_cls=Barrett,per_level_rotation_keys=False,raw_rotation_keys=keys,coef_maps={k:NS(shape=(131072,),dtype=np.dtype('int32'))for k in keys},eval_key_a_full=NS(shape=(8,131072,114),dtype=np.dtype('uint32')),eval_key_b_full=NS(shape=(8,131072,114),dtype=np.dtype('uint32')),q_moduli_at_level=lambda L:Q[:2*(L+1)])
        self.ctx=NS(degree=131072,num_slots=65536,composite_degree=2,secret_key=None,parameters={},_param_cache=cache,degree_layout=(256,512),dnum=8,max_level=49,scaling_factor=D,q_towers=list(Q),p_towers=list(P))
        self.rotation=Helper(self.ctx);self.rotation.rotate=self.rotate
        self._plaintext_multiply=Helper(self.ctx);self._plaintext_multiply.multiply=self.pt_multiply
        self._rescale=Helper(self.ctx);self._rescale.rescale=self.rescale
        self._multiplication_compiler=NS(clear_executables=lambda:None)
        self.ctx.he_add={L:NS(add=self.add,add_plain=self.add_plain)for L in range(50)}
        self.ctx.he_sub={L:NS(sub=self.subtract)for L in range(50)}
    def _level(self,v):return v.level
    def _prepare(self,L):pass
    def _finish(self,v):v.polynomial.block_until_ready();return v
    def _clear_working_caches(self):pass
    def _public_slots(self,v):return np.broadcast_to(np.asarray(v,dtype=np.float64),(65536,)).copy()
    def _encode_public(self,slots,L,scale):
        self.encodings.append((slots.copy(),L,scale))
        return NS(slots=slots.copy(),level=L,_ckks_scale=scale,polynomial=np.array([int(np.any(slots))],dtype=np.uint32))
    def pt_multiply(self,v,p,L):self.calls.append('pt');return Value(v.data*p.slots,L,v._ckks_scale*p._ckks_scale)
    def rescale(self,v,L,to):self.calls.append('rescale');return self._finish(Value(v.data,to,v._ckks_scale/(Q[2*L]*Q[2*L+1])))
    def add_plain(self,v,p):self.calls.append('plainadd');return Value(v.data+p.slots,v.level,v._ckks_scale)
    def rotate(self,v,k):self.calls.append('rotate');return self._finish(Value(np.roll(v.data,-k),v.level,v._ckks_scale))
    def add(self,a,b):
        assert a._ckks_scale==b._ckks_scale
        self.calls.append('add');return self._finish(Value(a.data+b.data,min(a.level,b.level),a._ckks_scale))
    def subtract(self,a,b):self.calls.append('sub');return self._finish(Value(a.data-b.data,min(a.level,b.level),a._ckks_scale))
    def multiply(self,a,b):
        self.calls.extend(('multiply','rescale'));L=min(a.level,b.level)
        return self._finish(Value(a.data*b.data,L-1,a._ckks_scale*b._ckks_scale/(Q[2*L]*Q[2*L+1])))

@pytest.fixture(autouse=True)
def primitives(monkeypatch):
    monkeypatch.setitem(sys.modules,'encrypted_packed_qwen',NS(CompiledArithmetic=Arithmetic))
    monkeypatch.setitem(sys.modules,'encrypted_polynomial',NS(EncryptedPolynomialEvaluator=Arithmetic))
    monkeypatch.setitem(sys.modules,'jaxite_word',NS(Polynomial=Value))
    monkeypatch.setitem(sys.modules,'finite_field',NS(BarrettContext=Barrett))
    monkeypatch.setitem(sys.modules,'support',NS(require_secure_execution=lambda *args:None))
    yield
    g._QUARANTINE.clear()

def setup(K=128,rows=16,scale=D,progress=None):
    positions=np.arange(128);layout=ScoreReductionLayout(np.array([((positions+7*r)%128)<K for r in range(16)]),valid_rows=rows)
    rule=NormalizationRoundContract(RAW,K);ops=Arithmetic();owner=g.BertNormalizationRoundGain4(ops,layout,rule,progress=progress)
    return owner,ops,Value(values_for(layout,K),25,scale)

def record_metric(row):
    path=os.environ.get('GAIN4_METRICS')
    if path:
        with open(path,'a')as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')

@pytest.mark.parametrize('K',range(1,129))
@pytest.mark.parametrize('rows',(1,7,16))
def test_fullslot_literal_original_newton(K,rows):
    owner,ops,source=setup(K,rows);before=source.data.copy();out,stages,steps=owner.evaluate_tile(source,capture_checkpoints=True)
    expected,ys=independent(source.data,owner.layout,owner.contract)
    for name,want in expected.items():
        assert np.max(np.abs(stages[name].data-want))<=1e-6
    for i,y,e in steps:
        assert np.max(np.abs(y.data-ys[i-1]))<=1e-6
        if e is not None:assert np.max(np.abs(e.data-(1-expected['denominator']*ys[i-1])))<=1e-6
    assert np.all(out.data[owner.layout.input_mask==0]==0)
    assert Counter(ops.calls)['pt']==3 and Counter(ops.calls)['rotate']==14
    assert Counter(ops.calls)['multiply']==2*owner.contract.updates
    assert Counter(ops.calls)['rescale']==2*owner.contract.updates+3
    assert out.level==25-owner.contract.updates-5 and out._ckks_scale==owner.last_plan['scales']['probabilities']
    assert source._public_bound=='borrowed tag'and np.array_equal(source.data,before)
    assert all(not any(k.startswith('_public_')for k in vars(v))for v in stages.values())
    assert ops.scale==D and ops.ctx.scaling_factor==D
    error=max([float(np.max(np.abs(stages[name].data-want)))for name,want in expected.items()]+[float(np.max(np.abs(y.data-ys[i-1])))for i,y,e in steps]+[float(np.max(np.abs(e.data-(1-expected['denominator']*ys[i-1]))))for i,y,e in steps if e is not None])
    record_metric(dict(kind='all_K_fullslot',key_count=K,valid_rows=rows,slots=65536,max_abs_error=error,fixed_gate=1e-6,passed=True,updates=owner.contract.updates,output_level=out.level,output_scale=out._ckks_scale,primitive_counts=dict(Counter(ops.calls))))

@pytest.mark.parametrize('K',(1,15,128))
@pytest.mark.parametrize('scale',(np.nextafter(D,0.),D,np.nextafter(D,np.inf),D*(1+2e-13),D*(1-2e-13)))
def test_actual_input_scale_without_relabeling(K,scale):
    owner,ops,source=setup(K,7,float(scale));out,stages,_=owner.evaluate_tile(source,capture_checkpoints=True)
    assert owner.last_plan['input_scale']==float(scale)
    for name in owner.last_plan['scales']:assert stages[name]._ckks_scale==owner.last_plan['scales'][name]
    assert math.isclose(out._ckks_scale,D,rel_tol=1e-12)

@pytest.mark.parametrize('mutation',('nsd','layout','ntt','payload','key','callback','method','low_memory','scale','contract'))
def test_callback_mutation_stops_next_work_and_quarantines(mutation):
    box={};at=[]
    def progress(name,value,counts):
        if name!='masked_input':return
        owner,ops,source=box['setup']
        if mutation=='nsd':source._ckks_nsd=True
        if mutation=='layout':source.degree_layout=(256.,512.)
        if mutation=='ntt':source.ntt_ctx.ff_ctx.moduli=Q[:2]
        if mutation=='payload':source.polynomial=Payload(source.data,25)
        if mutation=='key':ops.ctx._param_cache.raw_rotation_keys[16]['a']=object()
        if mutation=='callback':owner.progress=lambda *args:None
        if mutation=='method':ops.rotate=lambda *args:None
        if mutation=='low_memory':ops.low_memory=False
        if mutation=='scale':ops.scale*=2
        if mutation=='contract':object.__setattr__(owner.contract,'updates',1)
        at.append(len(ops.calls))
    owner,ops,source=setup(progress=progress);box['setup']=(owner,ops,source);original_payload=source.polynomial
    with pytest.raises((ValueError,RuntimeError)):owner.evaluate_tile(source)
    assert at==[len(ops.calls)]and not owner._active and owner._failed is not None and g._QUARANTINE
    assert any(row[1]is original_payload for row in g._QUARANTINE[-1][4])
    with pytest.raises(RuntimeError):owner.clear()

@pytest.mark.parametrize('operation',('clear','reentry'))
def test_active_owner_cannot_reenter_or_clear(operation):
    box={}
    def progress(*args):
        owner,ops,source=box['setup']
        if operation=='clear':owner.clear()
        else:owner.evaluate_tile(source)
    owner,ops,source=setup(progress=progress);box['setup']=(owner,ops,source)
    with pytest.raises((RuntimeError,ValueError)):owner.evaluate_tile(source)
    assert not owner._active and owner._failed is not None

@pytest.mark.parametrize('error_type',(RuntimeError,KeyboardInterrupt,SystemExit))
def test_primitive_failure_before_return_retains_original(error_type,monkeypatch):
    owner,ops,source=setup();primary=error_type('primitive failure')
    # Preserve construction-bound function identity; the existing function's
    # input hook fails after entry, before a result can reach owner retention.
    def fail(*args):raise primary
    monkeypatch.setattr(ops,'encodings',NS(append=fail))
    with pytest.raises(error_type)as seen:owner.evaluate_tile(source)
    assert seen.value is primary and owner._failed is primary and g._QUARANTINE[-1][3]is source

@pytest.mark.parametrize('scale',(D*(1+2e-12),D*(1-2e-12),float('nan'),True))
def test_invalid_input_scale_fails_before_primitive(scale):
    owner,ops,source=setup();source._ckks_scale=scale
    with pytest.raises(ValueError):owner.evaluate_tile(source)
    assert not ops.calls and not ops.encodings

@pytest.mark.parametrize('fixture_index',(0,1))
def test_original_two_fixtures_all54_checkpoints_and_derived_gates(fixture_index):
    from fixtures import definitions,make_probabilities,client_oracles,checkpoint_names,checkpoint_error,derived_checks
    item=definitions()[fixture_index];layout=ScoreReductionLayout(item['key_mask'],valid_rows=item['valid_rows'])
    rule=NormalizationRoundContract(RAW,item['key_count']);ops=Arithmetic();owner=g.BertNormalizationRoundGain4(ops,layout,rule)
    source=Value(make_probabilities(item));out,stages,steps=owner.evaluate_tile(source,capture_checkpoints=True)
    references,_=client_oracles(source.data,item,rule)
    measured={name:stages[name].data for name in ('masked_input','squared_numerator','denominator','probabilities')}
    for i,y,e in steps:
        measured[f'y{i:02}']=y.data
        if e is not None:measured[f'e{i:02}']=e.data
    previous=steps[-2][2];measured[f'e{rule.updates:02}']=ops.multiply(previous,previous).data
    assert set(measured)==set(checkpoint_names(rule.updates))
    assert all(checkpoint_error(measured[n],references[n])['passed']for n in measured)
    assert all(row['passed']for row in derived_checks(measured,references,rule.updates).values())
    assert ops.calls.count('multiply')==2*rule.updates+1
    record_metric(dict(kind='original_fixture',name=item['name'],key_count=item['key_count'],valid_rows=item['valid_rows'],slots=65536,checkpoint_count=len(measured),checkpoints={n:checkpoint_error(measured[n],references[n])for n in measured},derived_checks=derived_checks(measured,references,rule.updates),fixed_gate=1e-6,passed=True))

@pytest.mark.parametrize('field',('payload','scale','nsd','ntt'))
def test_final_callback_guards_earlier_captured_arrays(field):
    saved={};ops_box={};at=[]
    def progress(name,value,counts):
        if name=='y01':saved['value']=value
        if name=='probabilities':
            v=saved['value']
            if field=='payload':v.polynomial=Payload(v.data,v.level)
            if field=='scale':v._ckks_scale*=2
            if field=='nsd':v._ckks_nsd=1.
            if field=='ntt':v.ntt_ctx.ff_ctx.computation_format_is_standard=False
            at.append(len(ops_box['ops'].calls))
    owner,ops,source=setup(progress=progress);ops_box['ops']=ops
    with pytest.raises(ValueError):owner.evaluate_tile(source,capture_checkpoints=True)
    assert at==[len(ops.calls)]and owner.last_plan is None

def test_callback_tags_removed_from_earlier_owned_only():
    saved={}
    def progress(name,value,counts):
        if name=='y01':saved['value']=value
        if name=='probabilities':
            saved['value']._public_invented='tag';value._public_invented='tag'
    owner,ops,source=setup(progress=progress);out,stages,_=owner.evaluate_tile(source,capture_checkpoints=True)
    assert source._public_bound=='borrowed tag'
    assert all(not any(n.startswith('_public_')for n in vars(v))for v in stages.values())

@pytest.mark.parametrize('kind',(RuntimeError,KeyboardInterrupt,SystemExit))
def test_failed_result_readiness_quarantines_before_retention(monkeypatch,kind):
    owner,ops,source=setup();primary=kind('result readiness unknown');pending=[]
    def fail(self):pending.append(self);raise primary
    monkeypatch.setattr(Payload,'block_until_ready',fail)
    with pytest.raises(kind)as seen:owner.evaluate_tile(source)
    assert seen.value is primary and pending and owner._failed is primary
    assert g._QUARANTINE[-1][-2]is primary and g._QUARANTINE[-1][3]is source
    with pytest.raises(RuntimeError):g.BertNormalizationRoundGain4(ops,owner.layout,owner.contract)
