from contextlib import contextmanager, nullcontext
from types import SimpleNamespace as NS
import ast,hashlib,json
from pathlib import Path
import pytest
import qwen_backend_composition as b
import qwen_key_contract as k
import qwen_key_adapter as a
from qwen_file_contract import canonical,request,ciphertext_binding,artifact,DELTA
from qwen_public_parameters import ParameterCatalog

class Array:
    def __init__(self,events,name):self.events=events;self.name=name;self.error=None
    def block_until_ready(self):
        self.events.append(('barrier',self.name))
        if self.error:raise self.error
        return self

class CT:
    def __init__(self,profile,events,name,level=49,scale=DELTA):
        self.polynomial=Array(events,name);self.moduli_array=Array(events,name+'.q')
        self.moduli=profile['q_towers'][:2*(level+1)];self.shape=(1,2,256,512,len(self.moduli))
        self.degree_layout=(256,512);self.precision=32;self._ckks_scale=scale;self._ckks_nsd=1
    def get_moduli(self):return self.moduli
    def validate(self):pass

class Files:
    def __init__(self,events,name):self.events=events;self.name=name;self.entered=False;self.retired=False
    def check(self,*,readback=False):self.events.append((self.name,'readback'if readback else'check'))
    def consumer_entered(self):self.entered=True;self.events.append((self.name,'enter'))
    def consumers_retired(self):self.retired=True;self.events.append((self.name,'retire'))
    def readback(self):self.events.append((self.name,'readback'))
    def matrix(self,name):raise AssertionError('recording owner must not request actual parameter bodies')
    def gamma(self,name):raise AssertionError('recording owner must not request actual parameter bodies')

class Helper:
    def __init__(self,events,name):self.events=events;self.name=name;self.error=None
    def clear(self,**kwargs):
        self.events.append(('clear',self.name,kwargs))
        if self.error:raise self.error
    def clear_executables(self):self.clear(executables=True)

class Arithmetic:
    def __init__(self,events,name):
        self.events=events;self.name=name
        for attr in ('rotation','_plaintext_multiply','_rescale','_multiplication_compiler'):
            setattr(self,attr,Helper(events,name+'.'+attr))
    def _clear_working_caches(self):self.events.append(('working_clear',self.name))


def setup(public,length=17,role='server'):
    events=[];p=public.profile();pb=public.check()
    profile=k.PublicProfile.from_bytes(pb['inputs/profile_audit.json'],pb['inputs/actual_key_plan.json'])
    descriptors={s.name:dict(sha256=k.sha(s.name.encode()),dtype='uint32',shape=list(s.shape),bytes=s.raw_bytes+128)for s in profile.arrays}
    body=k.public_body(profile,descriptors,k.KEYGEN);raw=k.canonical(body|dict(context_id=k.sha(k.canonical(body))))
    secret=k.canonical(dict(schema='cross-qwen-full-private-keys-v1',context_id=k.sha(k.canonical(body)),public_manifest_sha256=k.sha(raw),bindings=k.bindings(),secret=dict(sha256='a'*64,dtype='uint32',shape=list(profile.secret.shape),bytes=profile.secret.raw_bytes+128)))
    ids=k.ExpectedIds(k.sha(k.canonical(body)),k.sha(raw),k.KEYGEN,'1'*64,'2'*64,'3'*64,k.sha(secret)if role=='client'else None)
    identity=public.identity(context_id=ids.context_id,model_id=ids.model_id,producer_sources=ids.producer_source_manifest_sha256,consumer_sources=ids.consumer_source_manifest_sha256)
    pub=k.authenticate_public(raw,profile,ids,role=role)
    admission=a.admit(pub,canonical(identity),k.authenticate_private(secret,pub)if role=='client'else None)
    c=request(public,identity,'recording_case',list(range(0,length*2,2)))
    upload=Files(events,'upload');upload.public=public;upload.contract=canonical(c);upload.input_digest='4'*64
    upload.records=tuple(NS(binding=canonical(ciphertext_binding(public,c,'request',i)))for i in range(c['input']['input_tile_count']))
    params=Files(events,'parameters');params.catalog=ParameterCatalog.from_public(public)
    cache=NS(secret_key=None,q_towers=tuple(p['q_towers']),p_towers=tuple(p['p_towers']),degree_layout=(256,512),dnum=8)
    ctx=NS(secret_key=None,_param_cache=cache,parameters={'secret_key':None,'sigma':p['sigma'],'secret_distribution':p['secret_distribution']},**{n:p[n]for n in ('degree','num_slots','max_level','dnum','scaling_factor')},q_towers=tuple(p['q_towers']),p_towers=tuple(p['p_towers']),degree_layout=(256,512))
    state=NS(outputs=[],inputs=[],reduced=[],owner=None,enable_error=None,reduce_error=None,owner_error=None,malformed=None,secure_error_index=None)
    def reduce(value):
        events.append(('reduce',49,25,value))
        if state.reduce_error:raise state.reduce_error
        out=CT(p,events,'reduced'+str(len(state.reduced)),25,value._ckks_scale);state.reduced.append(out);return out
    ctx.he_level_reduce={(49,25):NS(level_reduce=reduce)}
    provider=NS(close=lambda:events.append(('provider','close')))
    def enable(actual):
        assert actual is ctx;events.append(('provider','enable'))
        if state.enable_error:raise state.enable_error
        return provider
    def arithmetic(actual,**kwargs):
        assert actual is ctx and kwargs==dict(low_memory=True,preserve_dispatch_cache=True)
        events.append(('construct','graph'));return Arithmetic(events,'graph')
    def linear(actual,layout,**kwargs):
        assert actual is ctx and layout==65536 and kwargs==dict(low_memory=True,preserve_dispatch_cache=True)
        events.append(('construct','linear'));return NS(arithmetic=Arithmetic(events,'linear'))
    def refresh(actual,**kwargs):
        assert actual is ctx and kwargs==dict(period=65536,level_budget=(4,4),public_bound=2.,meta_precision=10,compiled=True,low_memory=True,preserve_dispatch_cache=True)
        events.append(('construct','refresh'));return NS(compiled_primitives=(Helper(events,'refresh.helper'),),evaluator=Arithmetic(events,'refresh'))
    def contract(*args):
        assert args[:-1]==tuple(pb[n]for n in ('inputs/candidate_manifest.json','inputs/config.json','inputs/site_metadata.json','inputs/certificates.json','inputs/join_plan.json','qkv-export-v1/parameter_manifest.json','inputs/decoder_parameters.json','inputs/actual_key_plan.json'))
        assert args[-1]==tuple(c['input']['original_positions'])
        return NS(vocabulary_blocks=tuple(c['response']['blocks']),positions=args[-1])
    def parameter_source(contract,*,matrix_loader,gamma_loader):
        assert matrix_loader==params.matrix and gamma_loader==params.gamma
        return NS(contract=contract)
    def owner(graph,linear,contract,parameters,*,refresher):
        state.owner=(graph,linear,contract,parameters,refresher)
        def evaluate(values):
            events.append(('owner','evaluate',values));assert values==tuple(state.reduced)
            if state.owner_error:raise state.owner_error
            out=tuple(CT(p,events,'out'+str(i),2,DELTA*(1+i/1000))for i in range(38));state.outputs=list(out)
            rows=tuple(dict(block=i,start=d['start'],stop=d['stop'],level=2,scale=out[i]._ckks_scale,last_position=c['response']['last_original_position'])for i,d in enumerate(c['response']['blocks']))
            result=NS(logit_blocks=out,blocks=contract.vocabulary_blocks,original_positions=contract.positions,last_position=contract.positions[-1],metadata=rows)
            if state.malformed:state.malformed(result)
            return result
        return NS(evaluate=evaluate)
    def secure(actual,value):
        assert actual is ctx;events.append(('secure',value.polynomial.name))
        if value.polynomial.name=='out'+str(state.secure_error_index):raise ValueError('recording actual ring failure')
    def materialize(record):
        events.append(('materialize',len(state.inputs)));out=CT(p,events,'input'+str(len(state.inputs)))
        state.inputs.append(out);return out
    deps=b.Dependencies(enable,arithmetic,linear,lambda n:n,refresh,contract,parameter_source,owner,secure,materialize)
    args=dict(public=public,request=c,admission=admission,ctx=ctx,key_owner=object(),request_files=upload,parameter_files=params,dependencies=deps,cleanup_shield=nullcontext)
    def publish(i,value,binding):
        events.append(('publish',i));assert value is state.outputs[i]
        return artifact(public,c,'response',i,payload_sha256=k.sha(str(i).encode()),actual_scale=value._ckks_scale,input_digest=upload.input_digest)
    return NS(args=args,events=events,state=state,publish=publish,upload=upload,params=params,ctx=ctx,provider=provider,public=public,request=c)

@pytest.mark.parametrize('length',[1,16,17,32])
def test_genuine_prefix_call_and_all38_before_publication_with_actual_scales(public,length):
    case=setup(public,length);engine=b.BorrowedServerEngine(**case.args);out=engine.run(case.publish)
    assert len(case.state.inputs)==(length+15)//16 and len(case.state.reduced)==len(case.state.inputs)
    first=next(i for i,e in enumerate(case.events)if e[0]=='publish')
    assert [e[1]for e in case.events[:first]if e[0]=='secure'and e[1].startswith('out')]==['out'+str(i)for i in range(38)]
    assert [x['binding']['scale']for x in out['artifacts']]==[v._ckks_scale for v in case.state.outputs]
    assert engine.retired and case.upload.retired and case.params.retired
    provider=case.events.index(('provider','close'));assert all(i<provider for i,e in enumerate(case.events)if e[0]in('clear','working_clear'))
    assert case.events.index(('parameters','retire'))>provider and case.events.index(('upload','retire'))>provider
    assert not engine.refs and engine.result is None and not b._QUARANTINE

@pytest.mark.parametrize('field',['context_id','model_id','producer_source_manifest_sha256','consumer_source_manifest_sha256'])
def test_independent_identity_mismatch_before_file_callback(public,field):
    case=setup(public);case.request['public'][field]='f'*64
    with pytest.raises(ValueError):b.BorrowedServerEngine(**case.args)
    assert not case.events

def test_client_role_cannot_enter_server_composition(public):
    case=setup(public,role='client')
    with pytest.raises(ValueError,match='server'):b.BorrowedServerEngine(**case.args)
    assert not case.events

@pytest.mark.parametrize('location',['ctx','cache','parameters'])
def test_all_three_secret_locations_before_backend(public,location):
    case=setup(public)
    if location=='ctx':case.ctx.secret_key=object()
    elif location=='cache':case.ctx._param_cache.secret_key=object()
    else:case.ctx.parameters['secret_key']=object()
    with pytest.raises(ValueError,match='keyless'):b.BorrowedServerEngine(**case.args)
    assert not case.events

@pytest.mark.parametrize('phase',['enable','reduce','owner','publish'])
@pytest.mark.parametrize('error_type',[ValueError,KeyboardInterrupt,SystemExit])
def test_entered_failure_attempts_barriers_quarantines_all_owners_without_clear(public,phase,error_type):
    case=setup(public);error=error_type('recording primary');engine=b.BorrowedServerEngine(**case.args)
    if phase=='enable':case.state.enable_error=error
    elif phase=='reduce':case.state.reduce_error=error
    elif phase=='owner':case.state.owner_error=error
    def publish(*args):
        if phase=='publish':raise error
        return case.publish(*args)
    with pytest.raises(error_type)as caught:engine.run(publish)
    assert caught.value is error and engine.failed is error and b._QUARANTINE[-1][0]is engine
    assert not case.upload.retired and not case.params.retired and not any(e[0]in('clear','working_clear')or e==('provider','close')for e in case.events)
    assert ('parameters','readback')in case.events and ('upload','readback')in case.events
    with pytest.raises((RuntimeError,ValueError)):engine.run(publish)

@pytest.mark.parametrize('mutation',[lambda v:setattr(v,'_ckks_scale',DELTA*2),lambda v:setattr(v,'_ckks_nsd',True),lambda v:setattr(v,'shape',(1,2,256,512,50))])
def test_prefix_metadata_mismatch_not_relabelled(public,mutation):
    case=setup(public);old=case.ctx.he_level_reduce[49,25].level_reduce
    def bad(value):out=old(value);mutation(out);return out
    case.ctx.he_level_reduce[49,25].level_reduce=bad;engine=b.BorrowedServerEngine(**case.args)
    with pytest.raises(ValueError):engine.run(case.publish)
    assert not any(e[0]=='publish'for e in case.events)and engine.failed is not None

@pytest.mark.parametrize('index',[0,19,37])
def test_any_output_security_failure_before_first_publication(public,index):
    case=setup(public);case.state.secure_error_index=index;engine=b.BorrowedServerEngine(**case.args)
    with pytest.raises(ValueError,match='ring'):engine.run(case.publish)
    assert not any(e[0]=='publish'for e in case.events)and len(engine.refs)==42

@pytest.mark.parametrize('mutation',[lambda r:setattr(r,'logit_blocks',r.logit_blocks[:-1]),lambda r:setattr(r,'last_position',True),lambda r:r.metadata[-1].update(level=2.0),lambda r:setattr(r,'original_positions',(999,))])
def test_malformed_result_not_published_and_returned_arrays_retained(public,mutation):
    case=setup(public);case.state.malformed=mutation;engine=b.BorrowedServerEngine(**case.args)
    with pytest.raises(ValueError):engine.run(case.publish)
    assert not any(e[0]=='publish'for e in case.events)and engine.result is not None

@pytest.mark.parametrize('mutation',[lambda e,c:setattr(c.state.inputs[0],'precision',32.0),lambda e,c:e.refs.clear(),lambda e,c:setattr(c.state.outputs[37],'polynomial',Array(c.events,'replacement'))])
def test_callback_cannot_replace_original_arrays_metadata_or_retention(public,mutation):
    case=setup(public);engine=b.BorrowedServerEngine(**case.args)
    def publish(*args):row=case.publish(*args);mutation(engine,case);return row
    with pytest.raises(ValueError):engine.run(publish)
    assert len([e for e in case.events if e[0]=='publish'])==1 and engine._retained

def test_cleanup_failure_attempts_all_helpers_with_cancellation_priority(public):
    case=setup(public);engine=b.BorrowedServerEngine(**case.args);ordinary=OSError('helper clear');cancel=KeyboardInterrupt('later cleanup')
    def publish(*args):
        engine.arithmetic.rotation.error=ordinary;engine.linear.arithmetic._rescale.error=cancel
        return case.publish(*args)
    with pytest.raises(KeyboardInterrupt)as caught:engine.run(publish)
    assert caught.value is cancel and ('provider','close')in case.events
    assert not case.params.retired and not case.upload.retired and engine.failed is cancel

def test_shield_entry_failure_does_not_skip_barriers_or_readback(public):
    case=setup(public);error=RuntimeError('shield entry')
    @contextmanager
    def shield():raise error;yield
    case.args['cleanup_shield']=shield;engine=b.BorrowedServerEngine(**case.args)
    with pytest.raises(RuntimeError)as caught:engine.run(case.publish)
    assert caught.value is error and ('upload','readback')in case.events and ('parameters','readback')in case.events
    assert not case.params.retired and not any(e==('provider','close')for e in case.events)

def test_public_factory_denies_before_argument_or_backend_access():
    class Untouchable:
        def __getattribute__(self,name):raise AssertionError(name)
    with pytest.raises(PermissionError):b.create_reference_session(Untouchable())


def test_source_selection_exact_and_standard_constructor_seams():
    root=Path(__file__).resolve().parent;selection=json.loads((root/'source_selection.json').read_bytes())
    for name,row in selection['selected'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==row['sha256']
    assert selection['ordinary_streaming_linear_sha256']!=selection['rejected_prepared_streaming_linear_sha256']
    tree=ast.parse((root/'qwen_backend_composition.py').read_bytes())
    method=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='_materialize')
    assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='from_array'for n in ast.walk(method))
    assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('encrypt','decrypt','gen_pke_pair','gen_evaluation_key')for n in ast.walk(tree))
