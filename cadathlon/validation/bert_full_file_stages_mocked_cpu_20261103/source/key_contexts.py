"""Authenticated client/server key contexts for a dedicated serial stage process.

Rotation arrays remain read-only memory maps; they are never concatenated/copied
as a complete store. The server formats only the relin key during setup, and
core accessors format individual rotations when requested. The handler owns its
arithmetic/compiled owners and must close them before exiting this loader.
No context or key array is valid after exit. Loaders never generate keys.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import gc,hashlib,importlib,math,os,signal,stat,sys,threading
import numpy as np
import client_keys as keys
KEYGEN_METHODS=('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen')
CODEC_METHODS=('encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch','decode')
PRIVATE_OPTIONAL=frozenset(('requests.json','completed.json'))
_ACTIVE=threading.Lock()
# Failed readiness is not a reusable cache: preserve public host backing until
#dedicated-process exit, and fail every subsequent loader attempt.
_FAILURE_QUARANTINE=[]


class _ReadDirectory:
    def __init__(self,path,*,private=False):
        self.path=Path(path);self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            info=os.fstat(self.fd);self.identity=(info.st_dev,info.st_ino);self.private=private;self.check()
            if private and stat.S_IMODE(info.st_mode)!=0o700:raise ValueError('private key directory must be0700')
        except BaseException as primary:
            try:self.close()
            except BaseException as error:primary.add_note('directory close: '+repr(error))
            raise
    def check(self):
        info=self.path.lstat();held=os.fstat(self.fd)
        if (not stat.S_ISDIR(info.st_mode) or (info.st_dev,info.st_ino)!=self.identity
                or (held.st_dev,held.st_ino)!=self.identity):raise ValueError('key directory or its parent was replaced')
    def close(self):
        fd=self.fd;self.fd=None
        if fd is not None:os.close(fd)


def _file_state(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)

def _map_array(owner,name,description,shape,moduli):
    """Authenticate and map the SAME bounded regular file descriptor."""
    if (type(description)is not dict or set(description)!={'sha256','dtype','shape','bytes'}
            or type(description['sha256'])is not str or not keys.HEX.fullmatch(description['sha256'])
            or description['dtype']!='uint32' or type(description['bytes'])is not int
            or type(description['shape'])is not list or any(type(x)is not int for x in description['shape']) or tuple(description['shape'])!=shape):
        raise ValueError('strict key array descriptor differs')
    owner.check();fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=owner.fd);mapped=None
    try:
        info=os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_size!=description['bytes'] or info.st_size>math.prod(shape)*4+4096
                or (owner.private and stat.S_IMODE(info.st_mode)!=0o600)):
            raise ValueError('bounded regular key array or private permissions differ')
        with os.fdopen(os.dup(fd),'rb') as f:
            version=np.lib.format.read_magic(f)
            if version not in ((1,0),(2,0)):raise ValueError('unsupported NPY key version')
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            stored,fortran,dtype=reader(f,max_header_size=512);offset=f.tell()
        if (type(stored)is not tuple or any(type(x)is not int for x in stored) or stored!=shape or fortran
                or dtype!=np.dtype('uint32') or offset+math.prod(shape)*4!=info.st_size):raise ValueError('key NPY header differs before mapping')
        if keys._sha_fd(fd)!=description['sha256']:raise ValueError('key array byte identity differs')
        with os.fdopen(os.dup(fd),'rb') as f:mapped=np.memmap(f,mode='r',dtype=np.uint32,shape=shape,offset=offset)
        for limb,modulus in enumerate(moduli):
            if np.any(mapped[...,limb,:]>=modulus):raise ValueError('key residue outside declared modulus')
        owner.check()
        if _file_state(os.fstat(fd))!=_file_state(info):raise ValueError('key file changed while mapping')
        result=(mapped,_file_state(info))
    finally:
        active=sys.exception();failure=None
        try:os.close(fd)
        except BaseException as error:
            if active is not None:active.add_note('key descriptor close: '+repr(error))
            else:failure=error
        if mapped is not None and (active is not None or failure is not None):
            try:mapped._mmap.close()
            except BaseException as error:
                if active is not None:active.add_note('key mapping cleanup: '+repr(error))
                elif failure is None:failure=error
                else:failure.add_note('key mapping cleanup: '+repr(error))
        if active is None and failure is not None:raise failure
    return result



def _read_public(owner,p,rotations):
    raw=keys._read(owner.fd,'public_manifest.json',1<<20);manifest=keys._json(raw);specs=keys.inventory(p,rotations)
    if type(manifest)is not dict or set(manifest)!={'schema','bindings','parameters','rotations','arrays','representation','keygen_source_sha256','security_target','reusable_rotation_keys','context_id'}:
        raise ValueError('exact public key schema required')
    if type(manifest['keygen_source_sha256'])is not str or not keys.HEX.fullmatch(manifest['keygen_source_sha256']):raise ValueError('key generator source identity required')
    if type(manifest['arrays'])is not dict or set(manifest['arrays'])!=set(specs):raise ValueError('complete public key inventory required')
    body=keys._body(p,rotations,manifest['arrays'],manifest['keygen_source_sha256'])
    if raw!=keys.canonical(body|dict(context_id=keys.digest(body))):raise ValueError('public context identity differs')
    if set(os.listdir(owner.fd))!=set(specs)|{'public_manifest.json'}:raise ValueError('unexpected public key files')
    owner.check();return manifest,specs,raw


def _read_private(owner,public,raw_public):
    names=set(os.listdir(owner.fd));required={'secret.npy','private_manifest.json'}
    if not required<=names or names-required-PRIVATE_OPTIONAL:raise ValueError('unexpected private key files')
    for name in names:
        info=os.stat(name,dir_fd=owner.fd,follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o600:raise ValueError('private files must be regular0600')
    raw=keys._read(owner.fd,'private_manifest.json',65536);manifest=keys._json(raw)
    if type(manifest)is not dict or set(manifest)!={'schema','context_id','public_manifest_sha256','bindings','secret'}:raise ValueError('exact private key schema required')
    expected=dict(schema='cross-bert-full-private-keys-v1',context_id=public['context_id'],public_manifest_sha256=hashlib.sha256(raw_public).hexdigest(),bindings=dict(keys.BINDINGS),secret=manifest['secret'])
    if raw!=keys.canonical(expected):raise ValueError('private/public key identity differs')
    owner.check();return manifest,raw


def _crypto():
    import jax
    from jaxite_word import CKKSContext,ckks_ctx,polynomial
    modules={name:importlib.import_module(name) for name in ('key_gen','jaxite_word.key_gen')}
    if {m.__name__ for m in modules.values()}!={'key_gen','jaxite_word.key_gen'}:raise RuntimeError('both actual keygen module spellings required')
    jax.config.update('jax_enable_x64',True)
    return SimpleNamespace(Context=CKKSContext,jax=jax,ckks=ckks_ctx,polynomial=polynomial,key_modules=modules)


def _parameters(p):
    return {name:p[name] for name in ('degree','num_slots','q_towers','p_towers','composite_degree','scaling_factor','sigma','secret_distribution')}|dict(
        degree_layout=tuple(p['degree_layout']),output_scale=p['scaling_factor'],p=60,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)


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
def _load(public_keys,client_private,*,profile_bytes,plan_bytes):
    if not _ACTIVE.acquire(blocking=False):raise RuntimeError('key loaders require a dedicated serial process')
    server=client_private is None;owners=[];maps=[];states=[];patches=[];transfers=[];runtime=ctx=handle=None;primary=None;cleanup_error=None;stage_entered=False
    def attempt(label,fn):
        nonlocal cleanup_error
        try:return fn()
        except BaseException as error:
            if primary is not None:primary.add_note(label+': '+repr(error))
            elif cleanup_error is None:cleanup_error=error
            elif not isinstance(error,Exception) and isinstance(cleanup_error,Exception):
                error.add_note('earlier cleanup failure: '+repr(cleanup_error));cleanup_error=error
            else:cleanup_error.add_note(label+': '+repr(error))
    def forbidden(*args,**kwargs):raise RuntimeError('key generation or client codec forbidden in loaded keyless server')
    try:
        if _FAILURE_QUARANTINE:raise RuntimeError('unresolved key transfers quarantined; this process must exit')
        p,rotations=keys.public_contract(profile_bytes,plan_bytes);keys._security_check(p)
        if os.environ.get('CROSS_SKIP_TOPLEVEL_ROTKEYS')=='1':raise ValueError('per-level runtime key generation mode is forbidden')
        public_owner=_ReadDirectory(public_keys);owners.append(public_owner)
        public,specs,raw_public=_read_public(public_owner,p,rotations);arrays={}
        for name,(shape,moduli) in specs.items():
            mapped,state=_map_array(public_owner,name,public['arrays'][name],shape,moduli)
            states.append((public_owner,name,state))
            # The client needs only its PKE pair. Authenticate all others, then
            #drop each mapping immediately; no116GiB duplicate is created.
            maps.append(mapped)
            if server or name=='public_key.npy':arrays[name]=mapped
            else:
                mapped._mmap.close();maps.pop()
        private=None
        if not server:
            private_owner=_ReadDirectory(client_private,private=True);owners.append(private_owner)
            private,raw_private=_read_private(private_owner,public,raw_public)
            secret,state=_map_array(private_owner,'secret.npy',private['secret'],(len(p['q_towers']),p['degree']),p['q_towers'])
            maps.append(secret);states.append((private_owner,'secret.npy',state))
        runtime=_crypto()
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
        handle=SimpleNamespace(ctx=ctx,public_manifest=public)
        if not server:handle.private_manifest=private
        stage_entered=True
        yield handle
        if server:_keyless(ctx,p,rotations)
        for owner,name,state in states:
            owner.check()
            if _file_state(os.stat(name,dir_fd=owner.fd,follow_symlinks=False))!=state:raise ValueError('key file changed during loaded stage')
        if (keys._read(public_owner.fd,'public_manifest.json',1<<20)!=raw_public
                or set(os.listdir(public_owner.fd))!=set(specs)|{'public_manifest.json'}):raise ValueError('public manifest or file inventory changed during stage')
        if not server:
            checked,raw_checked=_read_private(private_owner,public,raw_public)
            if raw_checked!=raw_private:raise ValueError('private manifest changed during stage')
    except BaseException as error:primary=error;raise
    finally:
        shield=[];deferred=[]
        if threading.current_thread() is threading.main_thread():
            def remember_signal(signum,frame):deferred.append(signum)
            for signum in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
                def install(sig=signum):
                    original=signal.getsignal(sig);signal.signal(sig,remember_signal);shield.append((sig,original))
                attempt('cleanup signal shield',install)
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
            attempt('context disposal',lambda:_dispose_context(runtime,ctx,attempt))
            if handle is not None:handle.ctx=None
            attempt('global dispatch cache clear',runtime.jax.clear_caches)
            attempt('Polynomial registry clear',runtime.polynomial._POLYNOMIAL_REGISTRY.clear)
            attempt('garbage collection',gc.collect)
        ctx=None
        stage_failed=server and stage_entered and primary is not None
        if not readiness_known or stage_failed:
            # A failed engine may already have dropped a pending lazy key
            #conversion from its cache. Its readiness is no longer observable;
            #retain ALL public backing even if remaining barriers succeeded.
            reason='failed entered server stage' if stage_failed else 'unresolved key transfer readiness'
            _FAILURE_QUARANTINE.append(dict(public_mappings=tuple(maps),transfer_owners=tuple(transfers),reason=reason))
            note='cleanup incomplete: '+reason+'; public mappings quarantined until process exit' 
            if primary is not None:primary.add_note(note)
            elif cleanup_error is not None:cleanup_error.add_note(note)
            else:cleanup_error=RuntimeError(note)
        else:
            for mapped in reversed(maps):attempt('key mapping close',mapped._mmap.close)
        maps.clear()
        for owner in reversed(owners):attempt('key directory close',owner.close)
        for module,name,original in reversed(patches):attempt('guard restoration',lambda m=module,n=name,o=original:setattr(m,n,o))
        _ACTIVE.release()
        for signum,original in reversed(shield):attempt('cleanup signal restoration',lambda s=signum,o=original:signal.signal(s,o))
        if deferred:
            def cancellation():raise KeyboardInterrupt('signals deferred until key cleanup completed: '+repr(deferred))
            attempt('deferred cleanup cancellation',cancellation)
        if primary is None and cleanup_error is not None:raise cleanup_error


def load_client(public_keys,client_private,*,profile_bytes,plan_bytes):
    """Reopen authenticated client keys; known request/completed metadata allowed."""
    return _load(public_keys,client_private,profile_bytes=profile_bytes,plan_bytes=plan_bytes)

def load_server(public_keys,*,profile_bytes,plan_bytes):
    """Yield public initialized context while all client codecs/keygen are blocked."""
    return _load(public_keys,None,profile_bytes=profile_bytes,plan_bytes=plan_bytes)
