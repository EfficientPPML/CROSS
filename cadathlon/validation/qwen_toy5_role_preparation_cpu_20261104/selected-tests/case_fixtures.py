"""Explicit recording fixtures; no backend imports, RNG or borrowed bodies."""
from contextlib import contextmanager,nullcontext
from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,json,math,os,weakref
ROOT=RUNTIME=OUT=EVENTS=None

def configure(root,runtime,out,events):
    global ROOT,RUNTIME,OUT,EVENTS
    ROOT,RUNTIME,OUT,EVENTS=root,runtime,out,events

def event(label,**fields):EVENTS.append(dict(event=label,**fields))
def labels():return [r['event']for r in EVENTS]
def h(label):return hashlib.sha256(('SYNTHETIC-Q5-TEST:'+label).encode()).hexdigest()
def clone(v):return copy.deepcopy(v)
def compact(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def pretty(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def digest(v):return hashlib.sha256(v).hexdigest()
def expect(kind,call,contains=None,same=None):
    try:call()
    except kind as e:
        if contains is not None:assert contains in str(e),(contains,str(e))
        if same is not None:assert e is same
        return e
    raise AssertionError('expected '+kind.__name__)
@contextmanager
def patch(obj,name,value):
    original=getattr(obj,name);setattr(obj,name,value)
    try:yield original
    finally:setattr(obj,name,original)

def profile():
    from q5_key_contract import PublicProfile
    return PublicProfile.from_bytes((RUNTIME/'public/profile.json').read_bytes(),(RUNTIME/'public/replay-report.json').read_bytes())
def root_meta(name,inode=10):return dict(path=str(OUT/'synthetic-metadata-only'/name),device=1,inode=inode,mode=0o700,uid=os.geteuid())
def state_meta(size,inode=20):return dict(device=1,inode=inode,size=size,mtime_ns=1,ctime_ns=1,mode=0o600,uid=os.geteuid())
def descriptor(spec):return dict(sha256=h(spec.name),dtype='uint32',shape=list(spec.shape),bytes=spec.raw_bytes+128)
def key_bytes(p=None):
    from q5_key_contract import public_body,KEYGEN
    p=p or profile();arrays={s.name:descriptor(s)for s in p.arrays};body=public_body(p,arrays,KEYGEN);public=body|dict(context_id=digest(compact(body)));pub=compact(public)
    priv=compact(dict(schema='cross-qwen-toy5-private-keys-v1',context_id=public['context_id'],public_manifest_sha256=digest(pub),bindings=public['bindings'],secret=descriptor(p.secret)))
    return pub,priv

def checkpoint_fixture():
    """Synthetic complete row/refresh metadata, no graph or inherited proof execution."""
    from q5_checkpoint_contract import numerical_plan
    from q5_actual_metadata_replay import weighted_scale,check_refreshes
    plan=numerical_plan();rows=[]
    for row in plan['checkpoints']:
        t=plan['full_trace'][row['clear_trace_index']]
        v=dict(index=row['index'],trace_index=row['clear_trace_index'],kind=row['kind'],label=row['label'],stack=row['stack'],level=row['planned_level'],scale=t['scale'],nsd=1)
        if row['kind'].startswith('refresh_'):v['public_bound']=row['public_bound']
        rows.append(v)
    befores=[r for r in rows if r['kind']=='refresh_before'];afters=[r for r in rows if r['kind']=='refresh_after'];obs=[]
    for i,(b,a)in enumerate(zip(befores,afters,strict=True)):
        before={k:b[k]for k in('level','scale','nsd')};after={k:a[k]for k in('level','scale','nsd')};bound=b['public_bound'];calls=[]
        if bound!=1.:
            calls.append(dict(before=before,after=dict(level=before['level']-1,scale=weighted_scale(before['scale'],before['level']),nsd=1),weight=1./bound,bias=0.))
        candidates=[plan['profile']['scale']*(2.**j)for j in range(-12,13)]
        candidates=[x for x in candidates if weighted_scale(x,26)==after['scale']]
        assert candidates,'synthetic final-scale fixture must reach exact public nominal row without relaxing arithmetic'
        calls.append(dict(before=dict(level=26,scale=candidates[0],nsd=1),after=after,weight=bound/64,bias=0.))
        obs.append(dict(index=i,public_bound=bound,before=before,weighted_calls=calls,after=after))
    check_refreshes(obs,rows)
    return rows,obs

def setup_bundle():
    from q5_fixture_assets import catalog,model_id
    from q5_key_contract import bindings
    from q5_role_authority import SetupAdmission,ExpectedSetupPins
    p=profile();pub,priv=key_bytes(p);context=json.loads(pub)['context_id'];mid=model_id(p)
    roles={r:h(r+'-role-source')for r in('setup','server','client')};grant=h('setup-grant');archive=h('setup-archive');rows,_=checkpoint_fixture()
    roots={r:root_meta(r,10+i)for i,r in enumerate(('public_keys','private_keys','assets','input','checkpoints'))}
    weights={}
    for i,(name,content)in enumerate(sorted(catalog(p).items())):
        size=math.prod(content['shape'])*8+128
        weights[name]=dict(name=f'weight_{i:02d}.npy',sha256=h(name+'container'),bytes=size,state=state_meta(size,100+i),content=content)
    assets=pretty(dict(schema='qwen-toy5-public-assets-v1',bindings=bindings(),model_id=mid,root=roots['assets'],arrays=weights,terminal_authority=False))
    artifact=dict(name='input.npz',sha256=h('inputbody'),bytes=104857600+1024,state=state_meta(104858624,501),metadata=rows[0])
    original=dict(root=roots['input'],artifact=artifact)
    inp=pretty(dict(schema='qwen-toy5-input-manifest-v1',bindings=dict(context_id=context,model_id=mid,setup_source_manifest_sha256=roles['setup'],setup_grant_sha256=grant),original_input=original))
    manifest=dict(schema='qwen-toy5-setup-manifest-v1',bindings=bindings(),context_id=context,model_id=mid,roles=roles,public_manifest_sha256=digest(pub),private_manifest_sha256=digest(priv),public_root=roots['public_keys'],private_root=roots['private_keys'],asset_root=roots['assets'],assets_manifest_sha256=digest(assets),input_manifest_sha256=digest(inp),original_input=original,public_arrays=203,private_arrays=1,public_weights=35,input_ciphertexts=1,terminal_authority=False)
    raw=compact(manifest)
    report=dict(schema='qwen-toy5-role-report-v1',role='setup',complete=True,passed=True,context_id=context,model_id=mid,source_manifest_sha256=roles['setup'],source_archive_sha256=archive,grant_sha256=grant,manifest_sha256=digest(raw),cleanup_completed=True,audit_denials=[])
    outer=dict(schema='qwen-toy5-role-outer-v1',role='setup',report_sha256=digest(pretty(report)),source_manifest_sha256=roles['setup'],source_archive_sha256=archive,grant_sha256=grant,exit_code=0,normal_return=True,reaped=True,process_group_absent=True,publication_completed=True)
    review=dict(schema='qwen-toy5-setup-root-review-v1',passed=True,manifest_sha256=digest(raw),setup_report_sha256=digest(pretty(report)),setup_outer_sha256=digest(pretty(outer)),source_manifest_sha256=roles['setup'],source_archive_sha256=archive,grant_sha256=grant,original_roots_verified=True,all_key_arrays_verified=True,original_input_verified=True,public_assets_verified=True,cleanup_and_exit_verified=True)
    def rebuild(*,outer_change=None,review_change=None):
        o=clone(outer);rr=clone(review)
        if outer_change:outer_change(o)
        rr['setup_outer_sha256']=digest(pretty(o))
        if review_change:review_change(rr)
        pins=dict(manifest=digest(raw),setup_report=digest(pretty(report)),setup_outer=digest(pretty(o)),setup_root_review=digest(compact(rr)),source_manifest=roles['setup'],source_archive=archive,grant=grant,context_id=context,model_id=mid,public_manifest=digest(pub),private_manifest=digest(priv),assets_manifest=digest(assets),input_manifest=digest(inp),setup_role_source=roles['setup'],server_role_source=roles['server'],client_role_source=roles['client'])
        adm=SetupAdmission(p,ExpectedSetupPins(compact(pins)),raw,pub,assets,inp,tuple(sorted(dict(setup_outer=pretty(o),setup_report=pretty(report),setup_root_review=compact(rr)).items())))
        return adm
    adm=rebuild();adm.check()
    return SimpleNamespace(profile=p,public=pub,private=priv,roots=roots,roles=roles,manifest=manifest,admission=adm,rebuild=rebuild,report=report,outer=outer,model_id=mid,context_id=context)

def server_bundle(setup):
    from q5_checkpoint_roles import authenticate_completed_server
    rows,obs=checkpoint_fixture();s=setup;artifacts=[]
    from q5_checkpoint_contract import numerical_plan
    for i,row in enumerate(rows[1:],1):
        size=numerical_plan()['checkpoints'][i]['raw_payload_bytes']+1024
        artifacts.append(dict(name=f'checkpoint_{i:02d}.npz',sha256=h('body'+str(i)),bytes=size,state=state_meta(size,600+i),metadata=row))
    sp=s.admission.expected.check();grant=h('server-grant');archive=h('server-archive')
    manifest=dict(schema='qwen-toy5-checkpoint-manifest-v1',bindings=dict(setup_report_sha256=sp['setup_report'],setup_outer_sha256=sp['setup_outer'],context_id=s.context_id,model_id=s.model_id,source_manifest_sha256=s.roles['server'],grant_sha256=grant),original_input=s.manifest['original_input'],output_root=s.roots['checkpoints'],artifacts=artifacts,rows=rows,refresh_observations=obs,complete=True,logical_checkpoints=47,new_exports=46,terminal_authority=False)
    raw=pretty(manifest);report=dict(schema='qwen-toy5-role-report-v1',role='server',complete=True,passed=True,context_id=s.context_id,model_id=s.model_id,source_manifest_sha256=s.roles['server'],source_archive_sha256=archive,grant_sha256=grant,manifest_sha256=digest(raw),cleanup_completed=True,audit_denials=[])
    outer=dict(schema='qwen-toy5-role-outer-v1',role='server',report_sha256=digest(pretty(report)),source_manifest_sha256=s.roles['server'],source_archive_sha256=archive,grant_sha256=grant,exit_code=0,normal_return=True,reaped=True,process_group_absent=True,publication_completed=True)
    evidence=dict(s.admission.evidence);evidence.pop('setup_root_review');evidence.update(server_report=pretty(report),server_outer=pretty(outer))
    root=dict(schema='qwen-toy5-server-root-review-v1',passed=True,server_report_sha256=digest(evidence['server_report']),server_outer_sha256=digest(evidence['server_outer']),manifest_sha256=digest(raw),setup_report_sha256=digest(evidence['setup_report']),setup_outer_sha256=digest(evidence['setup_outer']))
    evidence['server_root_review']=pretty(root)
    pins={k:digest(v)for k,v in evidence.items()};pins.update(manifest=digest(raw),context_id=s.context_id,model_id=s.model_id,setup_source=s.roles['setup'],server_source=s.roles['server'],server_grant=grant,output_root=s.roots['checkpoints'],input_root=s.roots['input'],original_input=s.manifest['original_input'])
    public=authenticate_completed_server(raw,evidence,pins)
    return public

def phase_bundle(role,setup=None):
    from q5_role_authority import PhaseAdmission,ExpectedPhasePins
    from q5_key_contract import bindings
    s=setup or setup_bundle();public=server_bundle(s)if role=='client'else None
    deps=compact(dict(schema='SYNTHETIC-dependency-policy'));res=compact(dict(schema='SYNTHETIC-resource-policy'));review=compact(dict(schema='SYNTHETIC-resource-review'))
    pmap={r:s.roles[r]for r in('setup','server','client')}
    borrowed={}if role=='setup'else{k:s.roots[k]for k in('public_keys','assets','input')}
    if role=='client':borrowed.update({k:s.roots[k]for k in('private_keys','checkpoints')})
    outputs={}
    for name in ({'public_keys','private_keys','assets','input','evidence'}if role=='setup'else{'checkpoints','evidence'}if role=='server'else{'evidence'}):
        parent=root_meta('output-parent-'+role+'-'+name,800+len(outputs));outputs[name]=dict(path=parent['path']+'/fresh',parent=parent)
    g=dict(schema='qwen-toy5-role-grant-v1',role=role,nonce='SYNTHETIC',source_manifest_sha256=pmap[role],source_archive_sha256=h(role+'archive'),bindings=bindings(),model_id=s.model_id,roles=pmap,setup_terminal_sha256=None if role=='setup'else digest(s.admission.manifest_bytes),server_terminal_sha256=digest(public.manifest_bytes)if public else None,dependency_policy_sha256=digest(deps),resource_policy_sha256=digest(res),resource_review_sha256=digest(review),borrowed_roots=borrowed,outputs=outputs,actual_execution_enabled=False)
    raw=compact(g);pins={k:g[k]for k in('role','source_manifest_sha256','source_archive_sha256','dependency_policy_sha256','resource_policy_sha256','resource_review_sha256','setup_terminal_sha256','server_terminal_sha256')};pins['grant_sha256']=digest(raw)
    a=PhaseAdmission(s.profile,ExpectedPhasePins(compact(pins)),raw,deps,res,review,None if role=='setup'else s.admission,public);a.check()
    return a,g,pins

class Array:
    """Metadata-only device array; never claims real residues or readiness."""
    def __init__(self,name,shape=(1,),dtype='uint32',fail=None):self.name=name;self.shape=shape;self.dtype=dtype;self.waits=0;self.fail=fail
    def block_until_ready(self):
        self.waits+=1;event('barrier',name=self.name,count=self.waits)
        if self.fail is not None:raise self.fail
        return self
    def __getitem__(self,key):return self
    def __ge__(self,key):return False
class RecordingNP:
    ndarray=Array
    @staticmethod
    def dtype(value):return value
    @staticmethod
    def any(value):return bool(value)

class Value:
    def __init__(self,array,q,ntt,validate_error=None):
        self.batch=1;self.num_elements=2;self.num_moduli=len(q);self.degree=131072;self.precision=32;self.r=256;self.c=512;self.degree_layout=(256,512)
        self.moduli=list(q);self.polynomial=array;self.moduli_array=Array('moduli',(len(q),));self.ntt_ctx=ntt;self._ckks_scale=1.;self._ckks_nsd=1;self.validate_error=validate_error
    def validate(self):
        event('validate')
        if self.validate_error is not None:raise self.validate_error
        return self

def backend_fixture(*,row_index=0,cache=True,validate_error=None,security_error=None,ff_error=None,ntt_error=None):
    from q5_checkpoint_contract import numerical_plan
    rows,_=checkpoint_fixture();row=rows[row_index];q=tuple(numerical_plan()['profile']['q_towers'][:2*(row['level']+1)]);created=[];ntt_objects=[];getter_calls=[]
    class FF:
        def __init__(self,*,moduli):
            event('ff_init',owner=id(self));self.moduli=list(moduli);self.tables={'table':Array('ff_table')};ntt_objects.append(self)
            if ff_error is not None:raise ff_error
    class NTT:
        def __init__(self,*,moduli,parameters):
            event('ntt_init',owner=id(self),ff=id(parameters['finite_field_context']));self.ff_ctx=parameters['finite_field_context'];self.tables=[Array('ntt_table')];self.r=parameters['r'];self.c=parameters['c'];ntt_objects.append(self)
            if ntt_error is not None:raise ntt_error
    original_ntt=SimpleNamespace(ff_ctx=SimpleNamespace(moduli=list(q)))
    class Cache:
        def get_sliced_ntt_q(self,level):getter_calls.append(level);event('cache_get',level=level);return original_ntt
    ctx=SimpleNamespace()
    if cache:ctx._param_cache=Cache()
    class Polynomial:
        @classmethod
        def from_array(cls,array,shapes,parameters):
            event('from_array',shape_keys=sorted(shapes),parameter_keys=sorted(parameters),array=id(array),ntt=id(parameters['ntt_ctx']))
            assert shapes==dict(batch=1,num_elements=2,num_moduli=len(q),degree=131072,precision=32,degree_layout=(256,512))
            assert parameters==dict(moduli=list(q),ntt_ctx=parameters['ntt_ctx'])
            v=Value(array,q,parameters['ntt_ctx'],validate_error);created.append(v);return v
    def asarray(payload,*,dtype):event('transfer');assert dtype=='uint32';return Array('payload',payload.shape)
    def secure(c,values):
        event('security',ctx=id(c),values=[id(v)for v in values]);assert c is ctx
        if security_error is not None:raise security_error
    b=SimpleNamespace(jax=SimpleNamespace(numpy=SimpleNamespace(asarray=asarray,uint32='uint32')),polynomial=SimpleNamespace(Polynomial=Polynomial),finite_field=SimpleNamespace(BarrettContext=FF),ntt_mm=SimpleNamespace(NTTCiphertextBarrettContext=NTT,ntt_ciphertext_context_for=lambda ff:NTT),secure=secure)
    payload=Array('host-payload',(1,2,256,512,len(q)))
    return SimpleNamespace(np=RecordingNP,b=b,ctx=ctx,payload=payload,row=row,q=q,values=created,ntt_objects=ntt_objects,getter_calls=getter_calls,original_ntt=original_ntt)

def generation_fixture(fail_stage=None,partial_store=False,late_source_error=None):
    from q5_key_producer import GenerationIntent,RecordingDependencies,CORE_MANIFEST_SHA256
    from q5_key_contract import bindings,KEYGEN
    p=profile();roots={name:str(OUT/'synthetic-metadata-only'/name)for name in('public','private')}
    parents={name:dict(path=str(Path(path).parent),device=1,inode=99,mode=0o700)for name,path in roots.items()}
    v=dict(schema='qwen-toy5-generation-intent-v1',nonce='SYNTHETIC',model_id=h('model'),generation_source_manifest_sha256=h('generation-source'),generation_source_archive_sha256=h('generation-archive'),core_source_manifest_sha256=CORE_MANIFEST_SHA256,request_producer_source_manifest_sha256=h('setup-role'),response_consumer_source_manifest_sha256=h('client-role'),keygen_source_sha256=KEYGEN,public_root=roots['public'],private_root=roots['private'],parents=parents,bindings=bindings(),parameters=p.parameters,logical_resources=p.resources,setup_preauthorization_sha256=h('preauthorization'),actual_execution_enabled=False)
    intent=GenerationIntent(p,compact(v));stores=[];secret=[[1,-1,0]];calls=[];failure=ValueError('SYNTHETIC '+str(fail_stage)+' denied')
    class Store:
        def __init__(self,path,specs,private):self.path=path;self.specs={s.name:s for s in specs};self.private=private;self.rows={};self.closed=False;self.close_count=0
        def write_array(self,name,value):event('write_array',name=name);self.rows[name]=descriptor(self.specs[name]);return self.rows[name]
        def verify(self):event('verify',private=self.private);return dict(self.rows)
        def write_manifest(self,raw):event('manifest',private=self.private);return raw
        def identity(self):return dict(root=self.path,device=1,inode=200+int(self.private),private=self.private)
        def close(self,primary=None):self.close_count+=1;self.closed=True;event('store_close',private=self.private)
    class Partial(Store):
        @property
        def verify(self):raise failure
    def source(i):
        event('source')
        if fail_stage=='source':raise failure
        if late_source_error is not None and stores and all(s.closed for s in stores):raise late_source_error
        return dict(source_manifest_sha256=v['generation_source_manifest_sha256'],core_source_manifest_sha256=CORE_MANIFEST_SHA256,keygen_source_sha256=KEYGEN,source_unchanged=True,synthetic_only=True)
    def security(params):
        event('generation_security')
        if fail_stage=='security':raise failure
        return dict(qp_secure=True,p_coverage=True,sigma=params['sigma'],secret_distribution=params['secret_distribution'],synthetic_only=True)
    def capacity(value):
        event('capacity')
        if fail_stage=='capacity':raise failure
        return dict(public_root=roots['public'],private_root=roots['private'],fresh_disjoint_roots=True,storage_capacity=True,descriptor_capacity=True,synthetic_only=True)
    def factory(path,specs,*,expected_parent,private=False,shield):
        event('store_acquire',private=private);s=(Partial if partial_store and private else Store)(path,specs,private);stores.append(s);return s
    def pke(q,aux,degree,*,noise_std):event('pke');calls.append(('pke',));return dict(public_key=object(),secret_key=secret)
    def switch():return dict(a=object(),b=object(),modulus={'Q':p.parameters['q_towers'],'P':p.parameters['p_towers']},P=p.parameters['p_towers'],shape=(4,124,131072))
    def evaluation(s,**kwargs):assert s is secret;calls.append(('evaluation',id(s)));event('evaluation');return switch()
    def rotation(s,q,aux,index,**kwargs):assert s is secret;calls.append(('rotation',index,id(s)));event('rotation',index=index);return {index:switch()}
    deps=RecordingDependencies(source,security,capacity,factory,pke,evaluation,rotation,nullcontext)
    return SimpleNamespace(intent=intent,deps=deps,p=p,stores=stores,secret=secret,calls=calls,failure=failure)
