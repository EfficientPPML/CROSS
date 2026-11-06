"""Eight source-authored loader/evaluation boundary cases; no real backend/body.

Executed only by the future reviewed fresh-child harness. Exact runtime403
loader/stage/engine/Finalizer bodies remain selected. All reader, numeric,
public-proof and fake-descriptor seams below are explicit recording operations.
No inherited test module is imported and no numerical source is evaluated.
"""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
import json
import os
import stat

CASES=(
    'loader_registration_before_yield',
    'loader_registration_failure_cleanup',
    'loader_registration_accessor_replacement',
    'loader_recording_none_control',
    'engine_evaluation_boundary_order',
    'engine_evaluation_callback_replacement',
    'engine_evaluation_callback_failure',
    'engine_recording_none_control',
)


class RecordingFiles:
    def __init__(self,env,label):
        self.env=env;self.label=label;self.entered=False;self.retired=False;self.closed=False
    def check(self,*,readback=False):
        assert not self.closed
        self.env.events.append((self.label,'readback'if readback else'check'))
    def consumer_entered(self):
        assert not self.entered;self.entered=True;self.env.events.append((self.label,'entered'))
    def consumers_retired(self):
        assert self.entered and not self.retired;self.retired=True;self.env.events.append((self.label,'retired'))
    def readback(self):self.check(readback=True)
    def matrix(self,name):raise AssertionError('recorded numerical owner cannot request any matrix body')
    def gamma(self,name):raise AssertionError('recorded numerical owner cannot request gamma body')
    def close(self,primary=None):
        assert not self.closed;self.closed=True;self.env.events.append((self.label,'close',primary))


class RecordingCT:
    def __init__(self,env,label,level,scale):
        self.polynomial=env.f.array(label+'.payload');self.moduli_array=env.f.array(label+'.moduli')
        self.moduli=tuple(env.p['q_towers'][:2*(level+1)])
        self.shape=(1,2,256,512,len(self.moduli));self.degree_layout=(256,512)
        self.precision=32;self._ckks_nsd=1;self._ckks_scale=float(scale)
        self.polynomial.shape=self.shape;self.moduli_array.shape=(len(self.moduli),)
    def validate(self):return None
    def get_moduli(self):return self.moduli


class RecordingHelper:
    def __init__(self,env,label):self.env=env;self.label=label
    def clear(self,**kwargs):self.env.events.append(('helper_clear',self.label,kwargs))
    def clear_executables(self):self.clear(executables=True)


class RecordingArithmetic:
    def __init__(self,env,label):
        self.env=env;self.label=label
        for name in ('rotation','_plaintext_multiply','_rescale','_multiplication_compiler'):
            setattr(self,name,RecordingHelper(env,label+'.'+name))
    def _clear_working_caches(self):self.env.events.append(('working_clear',self.label))


def _setup(f,case):
    """Called before exact snapshot capture, with every named seam recorded."""
    f.enable_numpy()
    fc=f.module('qwen_file_contract');kc=f.module('qwen_key_contract')
    ka=f.module('qwen_key_adapter');fa=f.module('qwen_key_files_admission')
    readers=f.module('qwen_key_readers');loader=f.module('qwen_key_loader')
    composition=f.module('qwen_backend_composition');stage=f.module('qwen_stage_composition')
    backend=f.module('qwen_server_backend');parameters=f.module('qwen_public_parameters')
    lifecycle=f.module('qwen_file_lifecycle');publisher=f.module('qwen_publisher')
    native=f.module('qwen_server_native_authority');device_module=f.module('qwen_server_device_owner')
    env=NS(f=f,case=case,fc=fc,kc=kc,ka=ka,fa=fa,readers=readers,loader=loader,
        composition=composition,stage=stage,backend_module=backend,parameters_module=parameters,device_module=device_module,
        events=[],maps=[],transfers=[],directories=[],created_contexts=[],published=[],outputs=[],
        reduced=[],inputs=[],engines=[],numerical_owners=[],registration_seen=[],sentinel_calls=0,
        backend=None,context=None,upload=None,parameters=None,publisher=None,handle=None,
        error=RuntimeError('recorded pre-evaluation synchronization failure'),fake_fd=987654321,
        recording=case in('loader_recording_none_control','engine_recording_none_control'))
    # Only immutable source-owned public profile/rotation JSON is selected.
    # This does not admit a key producer, model proof, or any key array body.
    prefix='key_adapter_source/public_inputs/'
    profile=kc.PublicProfile.from_bytes(f.blobs[prefix+'profile_audit.json'],f.blobs[prefix+'actual_key_plan.json'])
    env.profile=profile;env.p=profile.parameters;env.rotations=profile.rotations
    assert len(profile.arrays)==335 and len(env.rotations)==166
    p=env.p;env.catalog=object()
    metadata_names=('inputs/candidate_manifest.json','inputs/config.json','inputs/site_metadata.json',
        'inputs/certificates.json','inputs/join_plan.json','qkv-export-v1/parameter_manifest.json',
        'inputs/decoder_parameters.json','inputs/actual_key_plan.json')
    public_bytes={name:fc.canonical({'recording_public_metadata':name})for name in metadata_names}
    public_bytes['inputs/actual_key_plan.json']=fc.canonical({'combined_keys':list(env.rotations)})
    def public_check(self):
        assert self is f.public;return dict(public_bytes)
    def public_profile(self):
        assert self is f.public;return dict(p)
    def public_json(self,name):
        assert self is f.public;return json.loads(public_bytes[name])
    f.patch(fc.PublicContract,'check',public_check,'L/E synthetic public source marker check; no proof admission')
    f.patch(fc.PublicContract,'profile',public_profile,'L/E exact public profile geometry copied from pinned JSON')
    f.patch(fc.PublicContract,'json',public_json,'L/E public JSON lookup recording')
    identity=f.public.identity(context_id=f.ids.context_id,model_id=f.ids.model_id,
        producer_sources=f.ids.producer_source_manifest_sha256,consumer_sources=f.ids.consumer_source_manifest_sha256)
    def check_identity(self,value):
        assert self is f.public;fc.same(value,identity,'recording full identity differs');return value
    f.patch(fc.PublicContract,'check_identity',check_identity,'L/E explicitly synthetic identity, not model/key terminal proof')
    env.identity=identity
    request=fc.request(f.public,identity,'device_boundary_recording',[0])
    f.rebuild_request(request);env.request=request
    descriptions={s.name:dict(sha256=fc.sha(s.name.encode()),dtype='uint32',shape=list(s.shape),bytes=s.raw_bytes+128)for s in profile.arrays}
    env.manifest={'arrays':descriptions,'recording_only':True}
    raw=fc.canonical(env.manifest);object.__setattr__(f.ready,'public_manifest_bytes',raw)
    public=kc.PublicKeyBinding(profile,f.ids,raw,'server')
    def public_key_check(self):
        assert self is public and self.profile is profile and self.expected is f.ids and self.raw is raw and self.role=='server'
        return env.manifest
    f.patch(kc.PublicKeyBinding,'check',public_key_check,'L/E synthetic public manifest admission; no actual array hash/producer claim')
    keys=ka.Admission(public,fc.canonical(identity));files=fa.FilesAdmission(f.public,keys)
    # The real Admission/FilesAdmission checks, expected-ID check and all pure
    # request/artifact/scale/shape functions run around these explicit seams.
    f.keys=keys;f.key_public=public;f.files=files;env.files=files
    def catalog(cls,contract):
        assert contract is f.public;return env.catalog
    f.patch(parameters.ParameterCatalog,'from_public',classmethod(catalog),'L/E opaque recorded catalog; matrix/gamma calls forbidden')
    grant=json.loads(f.phase.grant_bytes);store=grant['public_key_store'];env.public_path=store['path']
    env.specs={s.name:(s.shape,s.moduli)for s in profile.arrays}
    def info(device,inode,mode,size=0,mtime=0,ctime=0):
        return NS(st_dev=device,st_ino=inode,st_mode=mode,st_uid=os.geteuid(),st_size=size,st_mtime_ns=mtime,st_ctime_ns=ctime)
    env.directory_info=info(store['device'],store['inode'],stat.S_IFDIR|store['mode'])
    parent=store['parent'];env.parent_info=info(parent['device'],parent['inode'],stat.S_IFDIR|parent['mode'])
    manifest_state=store['manifest']['state']
    env.manifest_info=info(manifest_state['device'],manifest_state['inode'],stat.S_IFREG|manifest_state['mode'],manifest_state['size'],manifest_state['mtime_ns'],manifest_state['ctime_ns'])
    env.key_info={name:info(201,1000+i,stat.S_IFREG|0o600,4,9,10)for i,name in enumerate(env.specs)}
    original_fstat=os.fstat;original_stat=os.stat;original_listdir=os.listdir;original_lstat=Path.lstat
    def fstat(fd):
        if fd==env.fake_fd:return env.directory_info
        return original_fstat(fd)
    def named_stat(path,*args,dir_fd=None,**kwargs):
        if dir_fd==env.fake_fd:
            assert not args and kwargs=={'follow_symlinks':False}
            return env.manifest_info if path==store['manifest']['name']else env.key_info[path]
        return original_stat(path,*args,dir_fd=dir_fd,**kwargs)
    def listdir(path):
        if path==env.fake_fd:return list(env.specs)+['public_manifest.json']
        return original_listdir(path)
    def lstat(path,*args,**kwargs):
        if str(path)==env.public_path:return env.directory_info
        if str(path)==parent['path']:return env.parent_info
        return original_lstat(path,*args,**kwargs)
    for name,fn in (('fstat',fstat),('stat',named_stat),('listdir',listdir)):
        f.patch(os,name,fn,'L/E exact fakeFD metadata only; all other source/temp calls delegated')
    f.patch(Path,'lstat',lstat,'L/E exact synthetic public root/parent observations; no borrowed stat')
    def directory_init(self,path,*,private=False):
        assert str(path)==env.public_path and private is False
        self.path=Path(path);self.fd=env.fake_fd;self.private=False;env.directories.append(self);env.events.append(('directory','open'))
    def directory_check(self):
        assert self.fd==env.fake_fd and self.private is False
        env.events.append(('directory','check'))
    def directory_close(self):
        assert self.fd==env.fake_fd;self.fd=None;env.events.append(('directory','close'))
    f.patch(readers.ReadDirectory,'__init__',directory_init,'L/E real ReadDirectory definition, recorded constructor/no FD')
    f.patch(readers.ReadDirectory,'check',directory_check,'L/E recorded held-directory check')
    f.patch(readers.ReadDirectory,'close',directory_close,'L/E recorded original directory retirement')
    def read_public(owner,binding):
        owner.check();assert binding is public;env.events.append(('manifest','readback'))
        return env.manifest,env.specs,raw
    def map_array(owner,name,description,shape,moduli):
        owner.check();assert description is descriptions[name]and(shape,moduli)==env.specs[name]
        row=NS(name=name,closed=False)
        def close():
            assert not row.closed;row.closed=True;env.events.append(('mapping','close',name))
        row._mmap=NS(close=close);env.maps.append(row);env.events.append(('mapping','admitted',name))
        return row,readers._file_state(env.key_info[name])
    f.patch(readers,'map_array',map_array,'L/E335 descriptor-only map recordings, no NPY/mmap/body')
    f.patch(readers,'read_public',read_public,'L/E synthetic metadata reader')
    f.patch(loader,'read_public',read_public,'L/E original loader imported reader alias recording')
    # Prevent real signal installation while exercising the original attempt-all
    # signal/guard restoration branches in the dedicated future child.
    masks=set();handlers={}
    def mask(how,values):
        previous=set(masks);masks.clear();masks.update(values if how==loader.signal.SIG_SETMASK else previous|set(values));return previous
    def getsignal(sig):return handlers.get(sig,'recorded-original-handler')
    def signal(sig,fn):
        previous=getsignal(sig);handlers[sig]=fn;return previous
    f.patch(loader.signal,'pthread_sigmask',mask,'L/E source-fixed signal-mask recording, no process signal mutation')
    f.patch(loader.signal,'getsignal',getsignal,'L/E source-fixed signal query recording')
    f.patch(loader.signal,'signal',signal,'L/E source-fixed signal setter recording')
    env.masks=masks;env.handlers=handlers
    def forbidden_original(*args,**kwargs):raise AssertionError('original key generation/codec must never be called')
    keymodule=f.core_modules['key_gen']
    for name in loader.KEYGEN_METHODS:setattr(keymodule,name,forbidden_original)
    env.original_generation={name:getattr(keymodule,name)for name in loader.KEYGEN_METHODS}
    def asarray(value,*,dtype):
        assert dtype=='uint32'and value in env.maps
        wrong=case=='loader_registration_failure_cleanup'and not env.transfers
        arr=f.array('key-transfer-'+value.name,device=object()if wrong else None);env.transfers.append(arr)
        def transpose(*axes):
            assert axes==(0,2,1);out=f.array('formatted-'+value.name);env.transfers.append(out);return out
        arr.transpose=transpose;return arr
    # f.start installs recording_jax. These are deferred constructor actions,
    # not genuine backend arrays or a mutation of selected runtime arithmetic.
    class Context:
        def __init__(self,parameters,*,batch,dnum):
            assert batch==1 and dnum==p['dnum']
            self.parameters=parameters;self.secret_key=None;self.public_key=None
            for name in ('degree','num_slots','max_level','dnum','scaling_factor'):setattr(self,name,p[name])
            self.q_towers=tuple(p['q_towers']);self.p_towers=tuple(p['p_towers']);self.degree_layout=tuple(p['degree_layout'])
            self._param_cache=NS(secret_key=None,per_level_rotation_keys=False,raw_rotation_keys={},coef_maps={},
                _formatted_rotation_keys={},q_towers=self.q_towers,p_towers=self.p_towers,degree_layout=self.degree_layout,dnum=dnum)
            self.he_level_reduce={(49,25):NS(level_reduce=lambda value:reduce(value))}
            env.context=self;env.created_contexts.append(self);f.ctx=self;env.events.append(('context','construct'))
            for name in loader.CODEC_METHODS:
                try:getattr(self,name)()
                except RuntimeError:pass
                else:raise AssertionError('all seven loaded-server codecs must be guarded before construction')
        def program_initialization(self,rotations,**kwargs):
            assert tuple(rotations)==env.rotations and kwargs['cache_rotation_keys']is True
            assert set(kwargs['pregenerated_rotation_keys'])==set(env.rotations)
            self._param_cache.raw_rotation_keys=kwargs['pregenerated_rotation_keys']
            self._param_cache.coef_maps={i:object()for i in rotations};env.events.append(('context','initialized'))
            if case=='loader_registration_accessor_replacement':
                f.mutate(env.backend.dependencies,'retain_loaded_device_owners',sentinel)
    for name in loader.CODEC_METHODS:setattr(Context,name,forbidden_original)
    env.original_codecs={name:getattr(Context,name)for name in loader.CODEC_METHODS}
    h=f.core_modules['he_params'];h.DEFAULT_SIGMA=p['sigma'];h.SECRET_DISTRIBUTION=p['secret_distribution']
    h.SecurityTarget=lambda bits,attack:(bits,attack)
    def secure_qp(q,pv,degree,target):
        assert list(q)==p['q_towers']and list(pv)==p['p_towers']and degree==p['degree']and target==(128,'classical')
        env.events.append(('security','recorded_qp'));return True
    def coverage(q,pv,dnum):assert list(q)==p['q_towers']and list(pv)==p['p_towers']and dnum==8
    h.qp_is_secure=secure_qp;h.assert_p_coverage=coverage
    ckks=f.core_modules['ckks_ctx'];ckks.CKKSContext=Context
    ckks.evict_context_caches=lambda ctx:env.events.append(('context','codec_evict'))
    poly=f.core_modules['polynomial'];poly.Polynomial=RecordingCT;poly._POLYNOMIAL_REGISTRY={}
    env.Context=Context;env.asarray=asarray
    def sentinel(*args,**kwargs):env.sentinel_calls+=1;raise AssertionError('replacement callback was invoked')
    env.sentinel=sentinel
    def reduce(value):
        env.events.append(('reduce',49,25));out=RecordingCT(env,'reduced',25,value._ckks_scale);env.reduced.append(out);return out
    def enable(ctx):assert ctx is env.context;return NS(close=lambda:env.events.append(('provider','close')))
    def arithmetic(ctx,**kwargs):assert ctx is env.context;return RecordingArithmetic(env,'graph')
    def linear(ctx,layout,**kwargs):assert ctx is env.context and layout==65536;return NS(arithmetic=RecordingArithmetic(env,'linear'))
    def refresh(ctx,**kwargs):assert ctx is env.context;return NS(compiled_primitives=(RecordingHelper(env,'refresh'),),evaluator=RecordingArithmetic(env,'refresh'))
    def contract(*args):
        assert args[:-1]==tuple(public_bytes[n]for n in metadata_names)and args[-1]==tuple(request['input']['original_positions'])
        return NS(vocabulary_blocks=tuple(request['response']['blocks']),positions=args[-1])
    def parameter_source(contract,*,matrix_loader,gamma_loader):
        assert matrix_loader==env.parameters.matrix and gamma_loader==env.parameters.gamma
        return NS(contract=contract)
    def numerical_owner(graph,linear,contract,parameters,*,refresher):
        def evaluate(prepared):
            env.events.append(('evaluate',));assert prepared==tuple(env.reduced)
            env.outputs=[RecordingCT(env,'output-'+str(i),2,fc.DELTA)for i in range(38)]
            return NS(logit_blocks=tuple(env.outputs),blocks=contract.vocabulary_blocks,
                original_positions=tuple(request['input']['original_positions']),last_position=request['response']['last_original_position'],
                metadata=tuple(dict(block=i,start=b['start'],stop=b['stop'],level=2,scale=fc.DELTA,last_position=request['response']['last_original_position'])for i,b in enumerate(request['response']['blocks'])))
        result=NS(evaluate=evaluate);env.numerical_owners.append(result);env.events.append(('full_owner','constructed'))
        if case=='engine_evaluation_callback_replacement':f.mutate(env.backend.factories,'before_evaluation',sentinel)
        return result
    def secure(ctx,value):assert ctx is env.context;value.validate()
    def materialize(record):
        value=RecordingCT(env,'fresh',49,fc.DELTA);env.inputs.append(value);return value
    env.raw_dependencies=composition.Dependencies(enable,arithmetic,linear,lambda n:n,refresh,contract,parameter_source,numerical_owner,secure,materialize)
    f.patch(composition,'_runtime_dependencies',lambda:env.raw_dependencies,'L/E exact Dependencies with tiny source-owned recording constructors; no numeric imports')
    def request_files(*args,**kwargs):
        if not env.recording:
            row=env.device_module._record(f.device)
            assert row['stage']=='post_keys'and f.runtime is not None
            f.runtime.require_server_body_scope();env.events.append(('post_keys','before_request'))
        upload=RecordingFiles(env,'upload');upload.public=f.public;upload.contract=fc.canonical(request)
        upload.input_digest=f.runtime.input_digest
        upload.records=tuple(NS(binding=fc.canonical(fc.ciphertext_binding(f.public,request,'request',i)))for i in range(request['input']['input_tile_count']))
        env.upload=upload;env.events.append(('request','constructed'));return upload
    def parameter_files(*args,**kwargs):
        params=RecordingFiles(env,'parameters');params.catalog=env.catalog;env.parameters=params;return params
    f.patch(lifecycle,'RequestFiles',request_files,'L/E request owner constructor recording, no ciphertext files')
    f.patch(parameters,'open_identity_bound_parameters',parameter_files,'L/E parameter owner recording; matrix/gamma body access always fails')
    def publisher_init(self,public_arg,request_arg,role,path,*,expected_parent,input_digest,shield):
        assert public_arg is f.public and request_arg==request and role=='response'
        self.public=public_arg;self.request=fc.canonical(request_arg);self.role=role;self.input_digest=input_digest
        self.records=[];self.closed=False;env.publisher=self;env.events.append(('publisher','constructed'))
    def write_ciphertext(self,index,value,binding):
        row=fc.artifact(f.public,request,'response',index,payload_sha256=fc.sha(str(index).encode()),actual_scale=value._ckks_scale,input_digest=self.input_digest)
        fc.same(binding,row['binding'],'recorded publisher binding differs')
        self.records.append(row);env.published.append(index);env.events.append(('publish',index));return row
    def finish(self):
        env.events.append(('publisher','finish'));return fc.canonical(fc.response(f.public,request,self.input_digest,self.records))
    def publisher_close(self,primary=None):
        assert not self.closed;self.closed=True;env.events.append(('publisher','close',primary))
    f.patch(publisher.Publisher,'__init__',publisher_init,'L/E exact Publisher type, recorded constructor/no output directory')
    f.patch(publisher.Publisher,'write_ciphertext',write_ciphertext,'L/E metadata-only38 publisher; no residue/file serialization')
    f.patch(publisher.Publisher,'finish',finish,'L/E original captured finish recording')
    f.patch(publisher.Publisher,'close',publisher_close,'L/E original captured publisher retirement recording')
    original_sync=native.synchronize
    def synchronize(lease,stage_name,context,arrays):
        env.events.append(('device_sync',stage_name))
        if case=='engine_evaluation_callback_failure'and stage_name=='pre_evaluation':raise env.error
        return original_sync(lease,stage_name,context,arrays)
    f.patch(native,'synchronize',synchronize,'L/E original source-fixed native recording, failure only at pre_evaluation')
    env.request_files=request_files;env.parameter_files=parameter_files;env.publisher_class=publisher.Publisher
    f._loader_engine_case=env


def _start(f,case):
    def setup(current):_setup(current,case)
    f.prepare_owner(before_capture=setup);env=f._loader_engine_case
    f.start();f.activate_runtime();f.runtime.allow_public_key_reads()
    # Recording jax is the honestly sourced, already selected temporary module.
    # Its numeric namespace is a recording adapter only, installed before the
    # backend captures this namespace and before any selected loader operation.
    import sys
    jax=sys.modules['jax'];jax.numpy=NS(asarray=env.asarray)
    jax.clear_caches=lambda:env.events.append(('dispatch','clear'))
    env.backend=env.backend_module.ServerBackend(f.runtime)
    return env


def _dependency_for(env):
    if not env.recording:return env.backend.dependencies
    d=env.backend.dependencies
    return env.loader.RecordingDependencies(d.security_check,d.runtime_factory,d.directory_factory,d.map_array)


def _loader_case(f,case):
    env=_start(f,case);yielded=False;error=None
    try:
        with env.loader._load_selected(env.files,env.public_path,None,dependencies=_dependency_for(env))as handle:
            yielded=True;env.handle=handle;env.events.append(('loader','yield'))
            row=env.device_module._record(f.device)
            if env.recording:assert row['handle']is None
            else:
                assert row['handle']is handle and row['context']is handle.ctx
                assert row['transfers']==tuple(env.transfers)and len(row['transfers'])==4
                env.registration_seen.append(row['transfers'])
            assert len(env.maps)==335 and all(not x.closed for x in env.maps)
    except BaseException as caught:error=caught
    assert len(env.maps)==335 and len(env.transfers)==4
    assert env.sentinel_calls==0
    assert all(getattr(env.Context,n)is original for n,original in env.original_codecs.items())
    assert all(getattr(f.core_modules['key_gen'],n)is original for n,original in env.original_generation.items())
    assert not env.loader._ACTIVE.locked()and not env.masks
    assert all(value=='recorded-original-handler'for value in env.handlers.values())
    if case=='loader_registration_failure_cleanup':
        assert not yielded and isinstance(error,(ValueError,PermissionError))
        row=env.device_module._record(f.device)
        assert row['handle']is not None and row['transfers']==tuple(env.transfers)and row['failure']is not None
        assert row in env.device_module._FAILED
        assert all(x.closed for x in env.maps)and all(not vars(ctx)for ctx in env.created_contexts)
        assert not env.loader._FAILURE_QUARANTINE  # known-ready, before yield; owner retains failure
    elif case=='loader_registration_accessor_replacement':
        assert not yielded and isinstance(error,ValueError)and 'registration changed'in str(error)
        assert all(x.closed for x in env.maps)
        assert env.device_module._record(f.device)['handle']is None
    else:
        assert error is None and yielded and env.handle.ctx is None
        assert all(x.closed for x in env.maps)and all(not vars(ctx)for ctx in env.created_contexts)
    assert len([x for x in env.events if x[:2]==('directory','close')])==1
    return dict(passed=True,case=case,checks=dict(metadata_mappings=335,setup_transfer_returns=4,
        yielded=yielded,registration_absent=env.recording,sentinel_calls=0,original_guards_restored=True,
        loader_lock_released=True,known_ready_cleanup=True,complete_device_retirement_claimed=False),
        seams=['synthetic key/public proofs','public profile metadata only','fake directory/stat335 mappings','recorded core/security/context/transfers','recorded signal operations'])


def _engine_case(f,case):
    env=_start(f,case);error=None;raw=None
    factories=env.backend.factories;dependencies=_dependency_for(env)
    if env.recording:
        def request_files(handle):env.handle=handle;return env.request_files()
        def parameters(handle):return env.parameter_files()
        def publisher(handle,digest):
            return env.publisher_class(f.public,env.request,'response','recording-output',expected_parent=b'{}',input_digest=digest,shield=nullcontext)
        factories=env.stage.RecordingStageFactories(request_files,parameters,publisher,lambda ctx:env.raw_dependencies,nullcontext)
    try:
        raw=env.stage._server_selected(env.files,env.public_path,env.request,key_dependencies=dependencies,factories=factories)
    except BaseException as caught:error=caught
    failure=case in('engine_evaluation_callback_replacement','engine_evaluation_callback_failure')
    assert env.sentinel_calls==0 and len(env.maps)==335 and len(env.transfers)==4
    assert not env.loader._ACTIVE.locked()and not env.composition._ACTIVE.locked()
    assert env.upload.closed and env.parameters.closed and env.publisher.closed
    if failure:
        assert raw is None and not env.published and not any(x[0]=='evaluate'for x in env.events)
        if case=='engine_evaluation_callback_failure':assert error is env.error
        else:assert isinstance(error,ValueError)and 'callback changed'in str(error)
        assert env.composition._QUARANTINE and env.stage._QUARANTINE and env.loader._FAILURE_QUARANTINE
        assert env.loader._FAILURE_QUARANTINE[-1]['context']is env.context
        assert env.loader._FAILURE_QUARANTINE[-1]['public_mappings']==tuple(env.maps)
        assert not any(x.closed for x in env.maps)
        assert not env.upload.retired and not env.parameters.retired
    else:
        assert error is None and type(raw)is bytes and env.published==list(range(38))
        assert env.upload.retired and env.parameters.retired and all(x.closed for x in env.maps)
        assert any(x[:2]==('provider','close')for x in env.events)
        assert not env.composition._QUARANTINE and not env.stage._QUARANTINE and not env.loader._FAILURE_QUARANTINE
        if not env.recording:
            row=env.device_module._record(f.device)
            assert row['evaluation'][2]is env.numerical_owners[0]and row['evaluation'][3]==tuple(env.reduced)
            assert row['handle'].ctx is None
            assert env.events.index(('post_keys','before_request'))<env.events.index(('request','constructed'))
            assert env.events.index(('full_owner','constructed'))<env.events.index(('device_sync','pre_evaluation'))<env.events.index(('evaluate',))
            assert env.events.index(('publisher','finish'))<next(i for i,x in enumerate(env.events)if x[:2]==('directory','close'))
            f.handle=row['handle'];f.ctx=row['context']
            f.finish_runtime_owner()
            assert [json.loads(raw)['stage']for raw in row['observations']]==['post_init','post_keys','pre_evaluation','terminal']
            assert row['mode']=='retired'and row['default_exit_attempted']is True
        else:assert env.device_module._record(f.device)['handle']is None
    return dict(passed=True,case=case,checks=dict(evaluations=sum(x[0]=='evaluate'for x in env.events),
        publications=len(env.published),sentinel_calls=0,original_file_close_attempts=True,
        entered_failure_quarantine=failure,none_recording_control=env.recording,
        numerical_or_backend_execution=False,complete_process_retirement_claimed=False),
        seams=['synthetic public/key identities','recorded335 mapping/context setup','real stage/engine/Finalizer bodies',
            'tiny source-owned numeric constructors/arrays','recorded request/parameter/publisher owners','recorded native synchronization'])


def run(f,case):
    if case not in CASES:raise ValueError('unselected L/E boundary case')
    return _loader_case(f,case)if case.startswith('loader_')else _engine_case(f,case)
