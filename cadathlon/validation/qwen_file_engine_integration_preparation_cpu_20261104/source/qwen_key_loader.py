"""Disabled real entry plus an explicit recording bridge to reviewed loader bodies.

No real backend factory is provided. _load_for_recording is a trusted internal
synthetic seam, not a capability or role grant. Real open_* always deny first.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
import gc,os,signal,sys,threading
from cleanup_origin import mark,select,protect_body
from qwen_key_files_admission import FilesAdmission
from qwen_key_readers import ReadDirectory,map_array,read_public,read_private,_file_state
KEYGEN_METHODS=('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen')
CODEC_METHODS=('encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch','decode')
_ACTIVE=threading.Lock()
_FAILURE_QUARANTINE=[]

@dataclass(frozen=True)
class RecordingDependencies:
    security_check:object
    runtime_factory:object
    directory_factory:object=ReadDirectory
    map_array:object=map_array
    synthetic_only:bool=True

def _admit_recording(admission,client_private,dependencies):
    if type(admission)is not FilesAdmission:raise TypeError('full public file admission required before readers')
    admission.check()
    if type(dependencies)is not RecordingDependencies or dependencies.synthetic_only is not True:
        raise PermissionError('only explicit recording dependencies are available')
    if any(not callable(getattr(dependencies,n))for n in ('security_check','runtime_factory','directory_factory','map_array')):
        raise TypeError('exact recording dependency callables required')
    expected='server'if client_private is None else'client'
    if admission.keys.public.role!=expected:raise ValueError('admitted key role differs from loader role')

def _parameters(p):
    return {name:p[name] for name in ('degree','num_slots','q_towers','p_towers','composite_degree','scaling_factor','sigma','secret_distribution')}|dict(
        degree_layout=tuple(p['degree_layout']),output_scale=p['scaling_factor'],p=56,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)

def _keyless(ctx,p,rotations):
    cache=getattr(ctx,'_param_cache',None)
    if (cache is None or ctx.secret_key is not None or cache.secret_key is not None or ctx.parameters.get('secret_key')is not None
            or ctx.public_key is not None or ctx.parameters.get('public_key')is not None):raise ValueError('server must contain no client secret or PKE key')
    if getattr(cache,'per_level_rotation_keys',None)is not False:raise ValueError('server reusable public rotation keys required')
    if set(cache.raw_rotation_keys)!=set(rotations) or set(cache.coef_maps)!=set(rotations):raise ValueError('initialized server rotation inventory differs')
    if list(ctx.q_towers)!=p['q_towers'] or list(ctx.p_towers)!=p['p_towers'] or ctx.degree_layout!=tuple(p['degree_layout']):raise ValueError('initialized server public profile differs')

def _dispose_context(runtime,ctx,attempt):
    if ctx is None:return
    # Stage owner must already have dropped its compiled bindings. Setup
    #transfers are synchronized separately even when construction failed.
    attempt('codec cache eviction',lambda:runtime.ckks.evict_context_caches(ctx))
    cache=getattr(ctx,'_param_cache',None)
    if cache is not None:
        module=sys.modules.get(type(getattr(cache,'ntt_q_max',None)).__module__)
        table_cache=getattr(module,'_barrett_ntt_context_cache',None)
        if table_cache is not None:attempt('public NTT cache clear',table_cache.clear)
        # Clear references from any accessor retained by the now-finished
        #handler, then invalidate this loader-owned context and its cache.
        for name,value in tuple(vars(ctx).items()):
            instances=getattr(value,'_instances',None)
            if instances is not None:attempt('operator cache clear',instances.clear)
        attempt('parameter cache invalidate',cache.__dict__.clear)
    parameters=getattr(ctx,'parameters',None)
    if isinstance(parameters,dict):attempt('parameter key references clear',parameters.clear)
    attempt('context invalidate',ctx.__dict__.clear)

@contextmanager
def _load_for_recording(admission,public_keys,client_private,*,dependencies):
    _admit_recording(admission,client_private,dependencies)
    if not _ACTIVE.acquire(blocking=False):raise RuntimeError('key loaders require a dedicated serial process')
    server=client_private is None;owners=[];maps=[];states=[];patches=[];transfers=[];runtime=ctx=handle=None;primary=None;cleanup_error=None;stage_entered=False
    def attempt(label,fn):
        nonlocal cleanup_error
        try:return fn()
        except BaseException as error:
            cleanup_error=select(cleanup_error,[(label,mark(error))])
    def forbidden(*args,**kwargs):raise RuntimeError('key generation or client codec forbidden in loaded keyless server')
    def retire_mapping(mapped):
        # Retain an uncertain close and never retry it in this process. This
        # covers both early client authentication drops and final retirement.
        try:mapped._mmap.close()
        except BaseException as error:
            _FAILURE_QUARANTINE.append(dict(public_mappings=(mapped,),transfer_owners=(),
                context=None,runtime=None,handle=None,reason='uncertain key mapping close'))
            raise mark(error)
    try:
        if _FAILURE_QUARANTINE:raise RuntimeError('unresolved key transfers quarantined; this process must exit')
        admission.check();p=admission.keys.public.profile.parameters;rotations=admission.keys.public.profile.rotations;dependencies.security_check(p)
        if os.environ.get('CROSS_SKIP_TOPLEVEL_ROTKEYS')=='1':raise ValueError('per-level runtime key generation mode is forbidden')
        public_owner=dependencies.directory_factory(public_keys);owners.append(public_owner)
        public,specs,raw_public=read_public(public_owner,admission.keys.public);arrays={}
        for name,(shape,moduli) in specs.items():
            mapped,state=dependencies.map_array(public_owner,name,public['arrays'][name],shape,moduli)
            states.append((public_owner,name,state))
            # The client needs only its PKE pair. Authenticate all others, then
            #drop each mapping immediately; no whole-store duplicate is created.
            maps.append(mapped)
            if server or name=='public_key.npy':arrays[name]=mapped
            else:
                maps.pop();retire_mapping(mapped)
        private=None
        if not server:
            private_owner=dependencies.directory_factory(client_private,private=True);owners.append(private_owner)
            private,raw_private=read_private(private_owner,admission.keys.private,raw_public)
            secret,state=dependencies.map_array(private_owner,'secret.npy',private['secret'],(len(p['q_towers']),p['degree']),p['q_towers'])
            maps.append(secret);states.append((private_owner,'secret.npy',state))
        runtime=dependencies.runtime_factory()
        # Guard every alias before context construction/initialization. Partial
        #installation is still restored by the attempt-all finally path.
        if set(runtime.key_modules)!={'key_gen','jaxite_word.key_gen'}:raise RuntimeError('both keygen aliases required')
        for module in runtime.key_modules.values():
            for name in KEYGEN_METHODS:
                original=getattr(module,name);patches.append((module,name,original));setattr(module,name,forbidden)
        if server:
            for name in CODEC_METHODS:
                original=getattr(runtime.Context,name);patches.append((runtime.Context,name,original));setattr(runtime.Context,name,forbidden)
            evaluation=[]
            for part in ('a','b'):
                transferred=runtime.jax.numpy.asarray(arrays[f'relin_{part}.npy'],dtype='uint32');transfers.append(transferred)
                formatted=transferred.transpose(0,2,1);transfers.append(formatted);evaluation.append(formatted)
            ctx=runtime.Context(_parameters(p)|dict(evaluation_key=evaluation),batch=1,dnum=p['dnum'])
            rotation_keys={index:{part:arrays[f'rot_{index}_{part}.npy'] for part in ('a','b')} for index in rotations}
            ctx.program_initialization(list(rotations),dnum=p['dnum'],r=p['degree_layout'][0],c=p['degree_layout'][1],batch=1,pregenerated_rotation_keys=rotation_keys,cache_rotation_keys=True)
            _keyless(ctx,p,rotations)
        else:
            ctx=runtime.Context(_parameters(p)|dict(public_key=arrays['public_key.npy'],secret_key=secret),batch=1,dnum=p['dnum'])
        for owner in owners:owner.check()
        handle=SimpleNamespace(ctx=ctx,public_manifest=public,public=admission.keys.public,admission=admission)
        if not server:handle.private_manifest=private
        stage_entered=True
        yield handle
        if server:_keyless(ctx,p,rotations)
        for owner,name,state in states:
            owner.check()
            if _file_state(os.stat(name,dir_fd=owner.fd,follow_symlinks=False))!=state:raise ValueError('key file changed during loaded stage')
        if (read_public(public_owner,admission.keys.public)[2]!=raw_public
                or set(os.listdir(public_owner.fd))!=set(specs)|{'public_manifest.json'}):raise ValueError('public manifest or file inventory changed during stage')
        if not server:
            checked,raw_checked=read_private(private_owner,admission.keys.private,raw_public)
            if raw_checked!=raw_private:raise ValueError('private manifest changed during stage')
    except BaseException as error:primary=protect_body(error);raise
    finally:
        shield=[];deferred=[]
        cancellation_signals=(signal.SIGINT,signal.SIGTERM,signal.SIGHUP)
        if threading.current_thread() is threading.main_thread():
            def remember_signal(signum,frame):deferred.append(signum)
            # Query without mutation first: SIG_BLOCK may change the mask and
            #then raise before returning its previous value.
            previous_mask=attempt('cleanup original signal mask',
                lambda:signal.pthread_sigmask(signal.SIG_BLOCK,()))
            if previous_mask is not None:
                try:
                    attempt('cleanup signal block',
                        lambda:signal.pthread_sigmask(signal.SIG_BLOCK,cancellation_signals))
                    for signum in cancellation_signals:
                        def install(sig=signum):
                            original=signal.getsignal(sig)
                            # Track before mutation: a setter can change the
                            #process handler and then raise or deliver a signal.
                            shield.append((sig,original))
                            signal.signal(sig,remember_signal)
                        attempt('cleanup signal shield',install)
                finally:
                    attempt('cleanup setup mask restoration',
                        lambda:signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask))
        readiness_known=True
        # Lazy rotation-key conversions can be created after setup. Keep all
        #still-reachable formatted device arrays until their barriers run.
        cache=getattr(ctx,'_param_cache',None) if ctx is not None else None
        formatted=getattr(cache,'_formatted_rotation_keys',{}) if cache is not None else {}
        def retain(value):
            if isinstance(value,dict):
                for child in value.values():retain(child)
            elif isinstance(value,(tuple,list)):
                for child in value:retain(child)
            else:transfers.append(value)
        attempt('retain lazy key transfers',lambda:retain(formatted))
        stage_failed=stage_entered and primary is not None
        if runtime is not None:
            # Constructor/transpose failures can leave pending transfers even
            #when no CKKSContext was assigned. Block each unique array before
            #releasing its host backing; never infer completion from ctx.
            seen=set()
            for transferred in transfers:
                if id(transferred) in seen:continue
                seen.add(id(transferred));wait=getattr(transferred,'block_until_ready',None)
                if callable(wait):
                    def synchronize(fn=wait):fn();return True
                    if attempt('key transfer synchronization',synchronize) is not True:readiness_known=False
            if not stage_failed and readiness_known:
                attempt('context disposal',lambda:_dispose_context(runtime,ctx,attempt))
                if handle is not None:handle.ctx=None
                attempt('global dispatch cache clear',runtime.jax.clear_caches)
                attempt('Polynomial registry clear',runtime.polynomial._POLYNOMIAL_REGISTRY.clear)
                attempt('garbage collection',gc.collect)
        if not readiness_known or stage_failed:
            # A failed engine may already have dropped a pending lazy key
            #conversion from its cache. Its readiness is no longer observable;
            #retain ALL public backing even if remaining barriers succeeded.
            reason='failed entered '+('server'if server else'client')+' stage' if stage_failed else 'unresolved key transfer readiness'
            _FAILURE_QUARANTINE.append(dict(public_mappings=tuple(maps),transfer_owners=tuple(transfers),context=ctx,runtime=runtime,handle=handle,reason=reason))
            note='cleanup incomplete: '+reason+'; key mappings and context quarantined until process exit' 
            if primary is not None:primary.add_note(note)
            elif cleanup_error is not None:cleanup_error.add_note(note)
            else:cleanup_error=mark(RuntimeError(note))
        else:
            for mapped in reversed(maps):attempt('key mapping close',lambda m=mapped:retire_mapping(m))
        maps.clear()
        for owner in reversed(owners):attempt('key directory close',owner.close)
        for module,name,original in reversed(patches):attempt('guard restoration',lambda m=module,n=name,o=original:setattr(m,n,o))
        _ACTIVE.release()
        if shield:
            previous_mask=attempt('cleanup original restoration mask',
                lambda:signal.pthread_sigmask(signal.SIG_BLOCK,()))
            try:
                if previous_mask is not None:
                    attempt('cleanup restoration signal block',
                        lambda:signal.pthread_sigmask(signal.SIG_BLOCK,cancellation_signals))
                for signum,original in reversed(shield):
                    attempt('cleanup signal restoration',lambda s=signum,o=original:signal.signal(s,o))
            finally:
                if previous_mask is not None:
                    # Pending cancellation can invoke the caller's restored
                    #handler here; keep body-primary/cleanup-only precedence.
                    attempt('cleanup final mask restoration',
                        lambda:signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask))
        if deferred:
            def cancellation():raise KeyboardInterrupt('signals deferred until key cleanup completed: '+repr(deferred))
            attempt('deferred cleanup cancellation',cancellation)
        chosen=select(primary,[('loader cleanup',cleanup_error)])
        if chosen is not None and chosen is not primary:raise chosen

def open_client(admission,public_keys,client_private):
    raise PermissionError('actual Qwen client key loading remains disabled pending independent role grant and proof')

def open_server(admission,public_keys):
    raise PermissionError('actual Qwen server key loading remains disabled pending independent role grant and proof')
