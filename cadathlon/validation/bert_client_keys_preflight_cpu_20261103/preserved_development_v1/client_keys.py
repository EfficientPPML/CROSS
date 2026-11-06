"""Actual client-only fixed-profile key generation and authenticated storage.

No input/model loading, context construction, codec, or inference is performed.
The external stage owns when to call generate. Tests may replace _security_check,
_crypto and storage seams, and must never present mocked results as real keys.
"""
from pathlib import Path
import hashlib,json,math,os,re,shutil,stat,sys
import numpy as np
PROFILE_SHA256='fe2408637f9d6de19ac6aef9f6f749aaef9dd56ccbb37b5334293cd223ede035'
PLAN_SHA256='bdc527179b9d1082d3dc0d90f1203523089aa8a24831941ad42a2513d20f7e04'
BINDINGS=dict(checkpoint_sha256='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034',
 config_sha256='ec2b6cd6ce741dda6c972aea96f9d1a449cbb4b5349e688c5370a9b2f6872d30',
 candidate_sha256='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9',
 profile_sha256=PROFILE_SHA256,rotation_plan_sha256=PLAN_SHA256)
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
    if type(profile_bytes)is not bytes or hashlib.sha256(profile_bytes).hexdigest()!=PROFILE_SHA256:raise ValueError('exact common p60 profile required')
    if type(plan_bytes)is not bytes or hashlib.sha256(plan_bytes).hexdigest()!=PLAN_SHA256:raise ValueError('exact frozen twelve-layer rotation plan required')
    p=_json(profile_bytes)['profile'];plan=_json(plan_bytes)
    keys=plan['union_keys']
    if len(keys)!=130 or any(type(i)is not int for i in keys) or keys!=sorted(set(keys)):raise ValueError('complete ordered130-key inventory required')
    params=dict(degree=p['degree'],num_slots=p['slots'],q_towers=p['q'],p_towers=p['p'],composite_degree=2,
        degree_layout=[256,512],precision=32,max_level=p['depth'],scaling_factor=p['canonical_scale'],dnum=p['dnum'],batch=1,
        sigma=3.190000057220458984375,secret_distribution='uniform_ternary')
    return params,tuple(keys)

def _security_check(p):
    # Importing these public arithmetic predicates does not initialize a CKKS
    #context or backend. Both predicates execute before _crypto/key generation.
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
    def check(self):
        current=os.stat(self.path.name,dir_fd=self.parent_fd,follow_symlinks=False)
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino)!=self.identity:raise ValueError('owned directory path replaced; replacement retained')
        if self.fd is not None:
            held=os.fstat(self.fd)
            if (held.st_dev,held.st_ino)!=self.identity:raise ValueError('owned directory descriptor changed')
    def remove(self):
        self.check();shutil.rmtree(self.path.name,dir_fd=self.parent_fd)
    def close(self):
        primary=None
        for name in ('fd','parent_fd'):
            descriptor=getattr(self,name);setattr(self,name,None)
            if descriptor is not None:
                try:os.close(descriptor)
                except BaseException as error:
                    if primary is None:primary=error
                    else:primary.add_note('additional directory close: '+repr(error))
        if primary is not None:raise primary

def _array(value,shape,moduli):
    a=np.asarray(value)
    if a.dtype.kind not in 'ui' or a.shape!=shape:raise ValueError('keygen must return exact integral public storage shape')
    for limb,modulus in enumerate(moduli):
        x=a[...,limb,:]
        if np.any(x<0) or np.any(x>=modulus):raise ValueError('key residue outside declared modulus before uint32 conversion')
    return a.astype(np.uint32,copy=False)

def _sha_fd(fd):
    os.lseek(fd,0,os.SEEK_SET)
    with os.fdopen(os.dup(fd),'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def _store(owner,name,value,shape,moduli):
    owner.check();a=_array(value,shape,moduli)
    descriptor=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=owner.fd)
    with os.fdopen(descriptor,'wb') as f:np.save(f,a,allow_pickle=False);f.flush();os.fsync(f.fileno())
    owner.check()
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=owner.fd)
    try:result=dict(sha256=_sha_fd(fd),dtype='uint32',shape=list(shape),bytes=os.fstat(fd).st_size)
    finally:os.close(fd)
    return result

def _write(owner,name,value):
    owner.check();fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=owner.fd)
    with os.fdopen(fd,'wb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
    owner.check();os.fsync(owner.fd)

def _read(directory_fd,name,maximum):
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory_fd)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size>maximum:raise ValueError('regular bounded key file required')
        with os.fdopen(os.dup(fd),'rb') as f:raw=f.read(maximum+1)
        if len(raw)>maximum:raise ValueError('key metadata exceeds limit')
        return raw
    finally:os.close(fd)

def _authenticate_array(directory_fd,name,description,shape,moduli):
    if (type(description)is not dict or set(description)!={'sha256','dtype','shape','bytes'}
            or type(description['sha256'])is not str or not HEX.fullmatch(description['sha256'])
            or description['dtype']!='uint32' or type(description['bytes'])is not int
            or type(description['shape'])is not list or any(type(x)is not int for x in description['shape']) or tuple(description['shape'])!=shape):
        raise ValueError('strict key array descriptor differs')
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory_fd)
    mapped=None
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size!=description['bytes'] or info.st_size>math.prod(shape)*4+4096:raise ValueError('bounded key array file differs')
        with os.fdopen(os.dup(fd),'rb') as f:
            version=np.lib.format.read_magic(f)
            if version not in ((1,0),(2,0)):raise ValueError('unsupported NPY key version')
            reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
            stored,fortran,dtype=reader(f,max_header_size=512);offset=f.tell()
        if (type(stored)is not tuple or any(type(x)is not int for x in stored) or stored!=shape or fortran
                or dtype!=np.dtype('uint32') or offset+math.prod(shape)*4!=info.st_size):raise ValueError('key NPY header differs before mapping')
        if _sha_fd(fd)!=description['sha256']:raise ValueError('key array byte identity differs')
        with os.fdopen(os.dup(fd),'rb') as f:mapped=np.memmap(f,mode='r',dtype=np.uint32,shape=shape,offset=offset)
        for limb,modulus in enumerate(moduli):
            if np.any(mapped[...,limb,:]>=modulus):raise ValueError('serialized key residue outside declared modulus')
    finally:
        active=sys.exception();failure=None
        for cleanup in ((mapped._mmap.close,) if mapped is not None else ())+(lambda:os.close(fd),):
            try:cleanup()
            except BaseException as error:
                if active is not None:active.add_note('key read cleanup: '+repr(error))
                elif failure is None:failure=error
                else:failure.add_note('additional key read cleanup: '+repr(error))
        if active is None and failure is not None:raise failure

def _body(p,rotations,arrays,source_hash):
    return dict(schema='cross-bert-full-public-keys-v1',bindings=dict(BINDINGS),parameters=p,rotations=list(rotations),
        arrays=arrays,representation='standard-ntt-residues/digit-modulus-degree',keygen_source_sha256=source_hash,
        security_target=dict(bits=128,attack='classical'),reusable_rotation_keys=True)

def authenticate_public(public_keys,*,profile_bytes,plan_bytes):
    p,rotations=public_contract(profile_bytes,plan_bytes);_security_check(p);specs=inventory(p,rotations)
    path=Path(public_keys);fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        identity=os.fstat(fd);raw=_read(fd,'public_manifest.json',1<<20);m=_json(raw)
        if type(m)is not dict or set(m)!={'schema','bindings','parameters','rotations','arrays','representation','keygen_source_sha256','security_target','reusable_rotation_keys','context_id'}:raise ValueError('public key manifest inventory differs')
        if type(m['keygen_source_sha256'])is not str or not HEX.fullmatch(m['keygen_source_sha256']):raise ValueError('keygen source identity required')
        if type(m['arrays'])is not dict or set(m['arrays'])!=set(specs):raise ValueError('complete public array inventory required')
        body=_body(p,rotations,m['arrays'],m['keygen_source_sha256']);expected=body|dict(context_id=digest(body))
        if raw!=canonical(expected):raise ValueError('public context identity or canonical manifest differs')
        if set(os.listdir(fd))!=set(specs)|{'public_manifest.json'}:raise ValueError('unexpected public key files')
        for name,(shape,moduli) in specs.items():_authenticate_array(fd,name,m['arrays'][name],shape,moduli)
        current=path.lstat()
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino)!=(identity.st_dev,identity.st_ino):raise ValueError('public key directory changed during authentication')
        return m
    finally:
        active=sys.exception()
        try:os.close(fd)
        except BaseException as error:
            if active is None:raise
            active.add_note('public key directory close: '+repr(error))

def _switch_metadata(value,p,shape):
    if (type(value)is not dict or set(value)!={'a','b','modulus','P','shape'} or value['modulus']!={'Q':p['q_towers'],'P':p['p_towers']}
            or value['P']!=p['p_towers'] or type(value['shape'])is not tuple or any(type(x)is not int for x in value['shape']) or value['shape']!=shape):
        raise ValueError('key-switch generator metadata differs')

def generate(public_keys,client_private,*,profile_bytes,plan_bytes):
    """Generate actual keys only after security/capacity checks; never overwrite."""
    p,rotations,preflight_report=preflight(public_keys,client_private,profile_bytes=profile_bytes,plan_bytes=plan_bytes)
    public=None;private=None;primary=None;close_error=None;result=None
    try:
        public=_OwnedDirectory(public_keys);private=_OwnedDirectory(client_private)
        kg=_crypto();source_hash=hashlib.sha256(Path(kg.__file__).read_bytes()).hexdigest();specs=inventory(p,rotations)
        pair=kg.gen_pke_pair(p['q_towers'],[],p['degree'],noise_std=p['sigma'])
        if type(pair)is not dict or set(pair)!={'public_key','secret_key'}:raise ValueError('exact PKE pair required')
        sk_shape=(len(p['q_towers']),p['degree']);secret=pair['secret_key']
        # Preserve the producer's Python-integer secret for core keygen: feeding
        #uint32 would overflow its secret-square arithmetic before modular reduction.
        secret_record=_store(private,'secret.npy',secret,sk_shape,p['q_towers'])
        arrays={'public_key.npy':_store(public,'public_key.npy',pair['public_key'],*specs['public_key.npy'])}
        del pair
        ek=kg.gen_evaluation_key(secret,q=p['q_towers'],P=p['p_towers'],dnum=p['dnum'],noise_std=p['sigma'])
        _switch_metadata(ek,p,specs['relin_a.npy'][0])
        for part in ('a','b'):
            name=f'relin_{part}.npy';arrays[name]=_store(public,name,ek[part],*specs[name])
        del ek
        for index in rotations:
            generated=kg.gen_rotation_key(secret,p['q_towers'],p['p_towers'],index,dnum=p['dnum'],noise_std=p['sigma'])
            if type(generated)is not dict or set(generated)!={index}:raise ValueError('exact one-index rotation result required')
            entry=generated[index];_switch_metadata(entry,p,specs[f'rot_{index}_a.npy'][0])
            for part in ('a','b'):
                name=f'rot_{index}_{part}.npy';arrays[name]=_store(public,name,entry[part],*specs[name])
            del entry,generated
        del secret
        body=_body(p,rotations,arrays,source_hash);manifest=body|dict(context_id=digest(body))
        private_manifest=dict(schema='cross-bert-full-private-keys-v1',context_id=manifest['context_id'],
            public_manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),bindings=dict(BINDINGS),secret=secret_record)
        _write(private,'private_manifest.json',private_manifest);_write(public,'public_manifest.json',manifest)
        authenticated=authenticate_public(public_keys,profile_bytes=profile_bytes,plan_bytes=plan_bytes)
        if authenticated!=manifest:raise ValueError('stored public key authentication differs')
        _authenticate_array(private.fd,'secret.npy',secret_record,sk_shape,p['q_towers'])
        if _read(private.fd,'private_manifest.json',65536)!=canonical(private_manifest):raise ValueError('private manifest readback differs')
        if set(os.listdir(private.fd))!={'secret.npy','private_manifest.json'}:raise ValueError('unexpected private key files')
        for name in ('secret.npy','private_manifest.json'):
            if stat.S_IMODE(os.stat(name,dir_fd=private.fd,follow_symlinks=False).st_mode)!=0o600:raise ValueError('private file permission differs')
        public.check();private.check();result=dict(public_manifest=manifest,private_manifest=private_manifest,preflight=preflight_report)
    except BaseException as error:
        primary=error
        for owner in (public,private):
            if owner is not None:
                try:owner.remove()
                except BaseException as cleanup:primary.add_note('owned key cleanup: '+repr(cleanup))
        raise
    finally:
        for owner in (public,private):
            if owner is not None:
                try:owner.close()
                except BaseException as error:
                    if primary is not None:primary.add_note('owned key handles: '+repr(error))
                    elif close_error is None:close_error=error
                    else:close_error.add_note('additional key handle close: '+repr(error))
        if primary is None and close_error is not None:raise close_error
    return result
