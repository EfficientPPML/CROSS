"""Actual client-only fixed-profile key generation and authenticated storage.

No input/model loading, context construction, codec, or inference is performed.
The external stage owns when to call generate. Tests may replace _security_check,
_crypto and storage seams, and must never present mocked results as real keys.
"""
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,math,os,re,shutil,stat,sys
import numpy as np
from cleanup_origin import mark,is_cleanup,protect_body
PROFILE_SHA256='f86b3b56bed72aeabec50b81dababeb62ba1627cf6764efcec189c930f708f18'
PLAN_SHA256='bb0d15f96fc2b01dd611a526f9aa9274965fdefe9b350e2663593a57283268c7'
BINDINGS=dict(original_source_archive_sha256='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b',
 profile_sha256=PROFILE_SHA256,diagnostic_plan_sha256=PLAN_SHA256)
HEX=re.compile(r'[0-9a-f]{64}\Z')

def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def _json(raw):
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    def invalid(value):raise ValueError('nonfinite JSON constant')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)

def public_contract(profile_bytes,plan_bytes):
    if type(profile_bytes)is not bytes or hashlib.sha256(profile_bytes).hexdigest()!=PROFILE_SHA256:raise ValueError('exact original toy4 p56 profile required')
    if type(plan_bytes)is not bytes or hashlib.sha256(plan_bytes).hexdigest()!=PLAN_SHA256:raise ValueError('exact reviewed toy4 diagnostic plan required')
    p=_json(profile_bytes);plan=_json(plan_bytes);rotations=plan['required_rotation_indices']
    if p!=plan['profile'] or len(rotations)!=199 or any(type(i)is not int for i in rotations)or rotations!=sorted(set(rotations)):raise ValueError('complete original199-key inventory required')
    params=dict(degree=p['degree'],num_slots=p['slots'],q_towers=p['q_towers'],p_towers=p['p_towers'],composite_degree=2,
        degree_layout=[256,512],precision=32,max_level=p['chain_depth'],scaling_factor=p['scale'],dnum=p['dnum'],batch=1,
        sigma=3.190000057220458984375,secret_distribution='uniform_ternary')
    return params,tuple(rotations)

def _security_check(p):
    # Core imports may initialize public backend constants, but construct no
    #CKKS context or keys. Both real predicates precede _crypto/key generation.
    from jaxite_word import he_params
    if type(p['sigma'])is not float or p['sigma']!=he_params.DEFAULT_SIGMA or p['secret_distribution']!=he_params.SECRET_DISTRIBUTION:
        raise ValueError('fixed security distributions differ')
    if not he_params.qp_is_secure(p['q_towers'],p['p_towers'],p['degree'],he_params.SecurityTarget(128,'classical')):
        raise ValueError('actual QP modulus is insecure')
    he_params.assert_p_coverage(p['q_towers'],p['p_towers'],p['dnum'])
    return dict(target_bits=128,attack='classical',actual_log2_qp=sum(math.log2(x) for x in p['q_towers']+p['p_towers']),P_coverage=True)

def _crypto():
    from jaxite_word import key_gen
    return key_gen

def inventory(p,rotations):
    q=len(p['q_towers']);n=p['degree'];alpha=(q+p['dnum']-1)//p['dnum'];parts=(q+alpha-1)//alpha
    switched=(parts,q+len(p['p_towers']),n)
    result={'public_key.npy':((2,q,n),tuple(p['q_towers']))}
    for name in ('relin_a.npy','relin_b.npy'):result[name]=(switched,tuple(p['q_towers']+p['p_towers']))
    for index in rotations:
        for part in ('a','b'):result[f'rot_{index}_{part}.npy']=(switched,tuple(p['q_towers']+p['p_towers']))
    return result

def resources(p,rotations):
    specs=inventory(p,rotations);public=sum(math.prod(shape)*4 for shape,_ in specs.values());private=len(p['q_towers'])*p['degree']*4
    return dict(public_array_bytes=public,private_array_bytes=private,public_array_count=len(specs),rotation_count=len(rotations),
        public_required_bytes=public+len(specs)*4096+(1<<30),private_required_bytes=private+4096+(1<<20),
        exclusions='working key-generation buffers, server contexts/compiled controls, model weights and ciphertexts; not a peak-memory estimate')

def preflight(public_keys,client_private,*,profile_bytes,plan_bytes):
    p,rotations=public_contract(profile_bytes,plan_bytes);security=_security_check(p);public_keys=Path(public_keys);client_private=Path(client_private)
    if public_keys==client_private or public_keys in client_private.parents or client_private in public_keys.parents:raise ValueError('public/private destinations must be separate siblings or separate trees')
    parents=[]
    for path in (public_keys,client_private):
        if path.exists() or path.is_symlink():raise FileExistsError('key destinations must be fresh')
        info=path.parent.lstat()
        if not stat.S_ISDIR(info.st_mode):raise ValueError('existing regular parent directory required')
        parents.append((info.st_dev,shutil.disk_usage(path.parent)))
    r=resources(p,rotations);requirements=[r['public_required_bytes'],r['private_required_bytes']]
    if parents[0][0]==parents[1][0]:
        if min(x[1].free for x in parents)<sum(requirements):raise OSError('insufficient shared-filesystem free space for complete key inventory')
    elif any(parent[1].free<need for parent,need in zip(parents,requirements)):raise OSError('insufficient public/private filesystem free space')
    r['filesystems']=[dict(device=dev,total_bytes=usage.total,free_bytes=usage.free) for dev,usage in parents]
    return p,rotations,dict(security=security,storage=r)

class _OwnedDirectory:
    def __init__(self,path):
        self.path=Path(path);self.parent_fd=None;self.fd=None;self.identity=None;created=False
        try:
            self.parent_fd=os.open(self.path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            os.mkdir(self.path.name,0o700,dir_fd=self.parent_fd);created=True
            info=os.stat(self.path.name,dir_fd=self.parent_fd,follow_symlinks=False);self.identity=(info.st_dev,info.st_ino)
            self.fd=os.open(self.path.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=self.parent_fd);self.check()
        except BaseException as primary:
            if created and self.identity is not None:
                try:self.remove()
                except BaseException as error:primary.add_note('new directory cleanup: '+repr(error))
            for descriptor in (self.fd,self.parent_fd):
                if descriptor is not None:
                    try:os.close(descriptor)
                    except BaseException as error:primary.add_note('new directory close: '+repr(error))
            raise
    def _check_held(self):
        current=os.stat(self.path.name,dir_fd=self.parent_fd,follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino)!=self.identity:raise ValueError('owned directory path replaced; replacement retained')
        if self.fd is not None:
            held=os.fstat(self.fd)
            if (held.st_dev,held.st_ino)!=self.identity:raise ValueError('owned directory descriptor changed')
    def check(self):
        self._check_held()
        visible=self.path.lstat()
        if not stat.S_ISDIR(visible.st_mode) or (visible.st_dev,visible.st_ino)!=self.identity:
            raise ValueError('owned directory parent/path replaced; replacement retained')
    def remove(self):
        # A renamed parent can still contain our original child. Delete only
        #that inode through the retained parent descriptor, never its replacement.
        self._check_held();shutil.rmtree(self.path.name,dir_fd=self.parent_fd)
    def close(self,*,_close_state=None):
        state=_CloseState()if _close_state is None else _close_state
        fd=self.fd;parent=self.parent_fd;self.fd=self.parent_fd=None
        try:
            if fd is not None:state.close(fd)
        finally:
            if parent is not None:state.close(parent)

def _array(value,shape,moduli):
    a=np.asarray(value)
    if a.dtype.kind not in 'ui' or a.shape!=shape:raise ValueError('keygen must return exact integral public storage shape')
    for limb,modulus in enumerate(moduli):
        x=a[...,limb,:]
        if np.any(x<0) or np.any(x>=modulus):raise ValueError('key residue outside declared modulus before uint32 conversion')
    return a.astype(np.uint32,copy=False)

_IO_FAILURE_QUARANTINE=[]


class _CloseState:
    """Share cleanup-origin state across nested metadata handles.

    Genuine body failures remain primary. Cleanup-only cancellation outranks
    ordinary cleanup errors. Uncertain resources are retained, never retried.
    This is dedicated-process failure handling, not process-wide atomicity.
    """
    def __init__(self):self.errors=[]
    def cleanup(self,label,action,owner=None):
        primary=protect_body(sys.exception());cleanup_only=is_cleanup(primary)
        try:action()
        except BaseException as error:
            mark(error);self.errors.append(error)
            if owner is not None:_IO_FAILURE_QUARANTINE.append((owner,error))
            if primary is None:raise
            if cleanup_only and isinstance(primary,Exception)and not isinstance(error,Exception):
                error.add_note('earlier metadata cleanup: '+repr(primary));raise
            if error is not primary:primary.add_note(label+': '+repr(error))
    def close(self,fd):
        self.cleanup('metadata descriptor close',lambda:os.close(fd),('descriptor',fd))


@contextmanager
def _stream(fd,mode,state):
    # The caller retains the descriptor even if stream construction fails.
    stream=os.fdopen(fd,mode,closefd=False)
    try:yield stream
    finally:state.cleanup('metadata stream close',stream.close,stream)


def _sha_fd(fd):
    # Borrow the original descriptor; no transfer/duplication is required.
    os.lseek(fd,0,os.SEEK_SET);digest=hashlib.sha256()
    while True:
        block=os.read(fd,1<<20)
        if not block:break
        digest.update(block)
    return digest.hexdigest()

def _store(owner,name,value,shape,moduli,*,_close_state=None):
    owner.check();a=_array(value,shape,moduli);state=_CloseState()if _close_state is None else _close_state
    descriptor=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=owner.fd)
    try:
        with _stream(descriptor,'wb',state)as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(descriptor)
    finally:state.close(descriptor)
    owner.check();fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=owner.fd)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode)or info.st_size>math.prod(shape)*4+4096:raise ValueError('new key array file changed')
        result=dict(sha256=_sha_fd(fd),dtype='uint32',shape=list(shape),bytes=info.st_size)
        after=os.fstat(fd)
        if any(getattr(info,n)!=getattr(after,n)for n in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('new key bytes changed during readback')
        owner.check();return result
    finally:state.close(fd)

def _write(owner,name,value,*,_close_state=None):
    owner.check();raw=canonical(value);state=_CloseState()if _close_state is None else _close_state
    fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=owner.fd)
    try:
        with _stream(fd,'wb',state)as f:f.write(raw);f.flush();os.fsync(fd)
        owner.check();os.fsync(owner.fd)
    finally:state.close(fd)

def _read(directory_fd,name,maximum,*,_close_state=None):
    state=_CloseState()if _close_state is None else _close_state
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory_fd)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size>maximum:raise ValueError('regular bounded key file required')
        chunks=[];remaining=maximum+1
        while remaining:
            block=os.read(fd,min(remaining,1<<20))
            if not block:break
            chunks.append(block);remaining-=len(block)
        raw=b''.join(chunks)
        if len(raw)>maximum:raise ValueError('key metadata exceeds limit')
        fields=('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns');after=os.fstat(fd)
        if len(raw)!=info.st_size or any(getattr(after,n)!=getattr(info,n)for n in fields):raise ValueError('key metadata changed during descriptor read')
        return raw
    finally:state.close(fd)

def _authenticate_array(directory_fd,name,description,shape,moduli,*,_close_state=None):
    if (type(description)is not dict or set(description)!={'sha256','dtype','shape','bytes'}
            or type(description['sha256'])is not str or not HEX.fullmatch(description['sha256'])
            or description['dtype']!='uint32' or type(description['bytes'])is not int
            or type(description['shape'])is not list or any(type(x)is not int for x in description['shape']) or tuple(description['shape'])!=shape):
        raise ValueError('strict key array descriptor differs')
    mapped=None;state=_CloseState()if _close_state is None else _close_state
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory_fd)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size!=description['bytes'] or info.st_size>math.prod(shape)*4+4096:raise ValueError('bounded key array file differs')
        with _stream(fd,'rb',state) as f:
            version=np.lib.format.read_magic(f)
            if version not in ((1,0),(2,0)):raise ValueError('unsupported NPY key version')
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            stored,fortran,dtype=reader(f,max_header_size=512);offset=f.tell()
        if (type(stored)is not tuple or any(type(x)is not int for x in stored) or stored!=shape or fortran
                or dtype!=np.dtype('uint32') or offset+math.prod(shape)*4!=info.st_size):raise ValueError('key NPY header differs before mapping')
        if _sha_fd(fd)!=description['sha256']:raise ValueError('key array byte identity differs')
        with _stream(fd,'rb',state) as f:mapped=np.memmap(f,mode='r',dtype=np.uint32,shape=shape,offset=offset)
        for limb,modulus in enumerate(moduli):
            if np.any(mapped[...,limb,:]>=modulus):raise ValueError('serialized key residue outside declared modulus')
        after=os.fstat(fd)
        if any(getattr(info,n)!=getattr(after,n)for n in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')):raise ValueError('authenticated key changed during readback')
    finally:
        try:
            if mapped is not None:state.cleanup('key mapping cleanup',lambda:mapped._mmap.close(),mapped)
        finally:state.close(fd)

def _body(p,rotations,arrays,source_hash):
    return dict(schema='cross-bert-toy4-public-keys-v1',bindings=dict(BINDINGS),parameters=p,rotations=list(rotations),
        arrays=arrays,representation='standard-ntt-residues/digit-modulus-degree',keygen_source_sha256=source_hash,
        security_target=dict(bits=128,attack='classical'),reusable_rotation_keys=True)

def authenticate_public(public_keys,*,profile_bytes,plan_bytes,_close_state=None):
    p,rotations=public_contract(profile_bytes,plan_bytes);_security_check(p);specs=inventory(p,rotations)
    state=_CloseState()if _close_state is None else _close_state
    path=Path(public_keys);fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        identity=os.fstat(fd);raw=_read(fd,'public_manifest.json',1<<20,_close_state=state);m=_json(raw)
        if type(m)is not dict or set(m)!={'schema','bindings','parameters','rotations','arrays','representation','keygen_source_sha256','security_target','reusable_rotation_keys','context_id'}:raise ValueError('public key manifest inventory differs')
        if type(m['keygen_source_sha256'])is not str or not HEX.fullmatch(m['keygen_source_sha256']):raise ValueError('keygen source identity required')
        if type(m['arrays'])is not dict or set(m['arrays'])!=set(specs):raise ValueError('complete public array inventory required')
        body=_body(p,rotations,m['arrays'],m['keygen_source_sha256']);expected=body|dict(context_id=digest(body))
        if raw!=canonical(expected):raise ValueError('public context identity or canonical manifest differs')
        if set(os.listdir(fd))!=set(specs)|{'public_manifest.json'}:raise ValueError('unexpected public key files')
        for name,(shape,moduli) in specs.items():_authenticate_array(fd,name,m['arrays'][name],shape,moduli,_close_state=state)
        current=path.lstat()
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino)!=(identity.st_dev,identity.st_ino):raise ValueError('public key directory changed during authentication')
        return m
    finally:state.close(fd)

def _switch_metadata(value,p,shape):
    if (type(value)is not dict or set(value)!={'a','b','modulus','P','shape'} or value['modulus']!={'Q':p['q_towers'],'P':p['p_towers']}
            or type(value['P'])is not list or any(type(x)is not int for x in value['P'])
            or type(value['modulus'])is not dict or set(value['modulus'])!={'Q','P'}
            or any(type(value['modulus'][n])is not list or any(type(x)is not int for x in value['modulus'][n]) for n in ('Q','P'))
            or value['P']!=p['p_towers'] or type(value['shape'])is not tuple or any(type(x)is not int for x in value['shape']) or value['shape']!=shape):
        raise ValueError('key-switch generator metadata differs')

def generate(public_keys,client_private,*,profile_bytes,plan_bytes):
    """Generate actual keys only after security/capacity checks; never overwrite."""
    p,rotations,preflight_report=preflight(public_keys,client_private,profile_bytes=profile_bytes,plan_bytes=plan_bytes)
    public=None;private=None;primary=None;result=None;state=_CloseState()
    try:
        public=_OwnedDirectory(public_keys);private=_OwnedDirectory(client_private)
        kg=_crypto();source_hash=hashlib.sha256(Path(kg.__file__).read_bytes()).hexdigest();specs=inventory(p,rotations)
        pair=kg.gen_pke_pair(p['q_towers'],[],p['degree'],noise_std=p['sigma'])
        if type(pair)is not dict or set(pair)!={'public_key','secret_key'}:raise ValueError('exact PKE pair required')
        sk_shape=(len(p['q_towers']),p['degree']);secret=pair['secret_key']
        # Preserve the producer's Python-integer secret for core keygen: feeding
        #uint32 would overflow its secret-square arithmetic before modular reduction.
        secret_record=_store(private,'secret.npy',secret,sk_shape,p['q_towers'],_close_state=state)
        arrays={'public_key.npy':_store(public,'public_key.npy',pair['public_key'],*specs['public_key.npy'],_close_state=state)}
        del pair
        ek=kg.gen_evaluation_key(secret,q=p['q_towers'],P=p['p_towers'],dnum=p['dnum'],noise_std=p['sigma'])
        _switch_metadata(ek,p,specs['relin_a.npy'][0])
        for part in ('a','b'):
            name=f'relin_{part}.npy';arrays[name]=_store(public,name,ek[part],*specs[name],_close_state=state)
        del ek
        for index in rotations:
            generated=kg.gen_rotation_key(secret,p['q_towers'],p['p_towers'],index,dnum=p['dnum'],noise_std=p['sigma'])
            if type(generated)is not dict or set(generated)!={index}:raise ValueError('exact one-index rotation result required')
            entry=generated[index];_switch_metadata(entry,p,specs[f'rot_{index}_a.npy'][0])
            for part in ('a','b'):
                name=f'rot_{index}_{part}.npy';arrays[name]=_store(public,name,entry[part],*specs[name],_close_state=state)
            del entry,generated
        del secret
        body=_body(p,rotations,arrays,source_hash);manifest=body|dict(context_id=digest(body))
        private_manifest=dict(schema='cross-bert-toy4-private-keys-v1',context_id=manifest['context_id'],
            public_manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),bindings=dict(BINDINGS),secret=secret_record)
        _write(private,'private_manifest.json',private_manifest,_close_state=state);_write(public,'public_manifest.json',manifest,_close_state=state)
        authenticated=authenticate_public(public_keys,profile_bytes=profile_bytes,plan_bytes=plan_bytes,_close_state=state)
        if authenticated!=manifest:raise ValueError('stored public key authentication differs')
        _authenticate_array(private.fd,'secret.npy',secret_record,sk_shape,p['q_towers'],_close_state=state)
        if _read(private.fd,'private_manifest.json',65536,_close_state=state)!=canonical(private_manifest):raise ValueError('private manifest readback differs')
        if set(os.listdir(private.fd))!={'secret.npy','private_manifest.json'}:raise ValueError('unexpected private key files')
        for name in ('secret.npy','private_manifest.json'):
            if stat.S_IMODE(os.stat(name,dir_fd=private.fd,follow_symlinks=False).st_mode)!=0o600:raise ValueError('private file permission differs')
        public.check();private.check();result=dict(public_manifest=manifest,private_manifest=private_manifest,preflight=preflight_report)
    except BaseException as error:
        primary=protect_body(error)
        def remove(index):
            owners=(public,private)
            if index==len(owners):return
            try:
                if owners[index]is not None:state.cleanup('owned key rollback',owners[index].remove,owners[index])
            finally:remove(index+1)
        remove(0)
        raise
    finally:
        def close(index):
            owners=(public,private)
            if index==len(owners):return
            try:
                if owners[index]is not None:owners[index].close(_close_state=state)
            finally:close(index+1)
        close(0)
    return result
