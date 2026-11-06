"""Held public parameter files for the Qwen owner; no model/backend construction.

These ordinary trusted-Python components do not grant permission to read files.
A future actual role must admit source/freeze/asset/role proofs before opening
them. This preparation uses only independently constructed synthetic files.
"""
from dataclasses import dataclass
import hashlib,io,math,mmap,os,stat,zipfile
import numpy as np
from reused_client_io import HeldDirectory,state
from qwen_public_parse import parse
from qwen_finalizer import Finalizer
from qwen_file_contract import PublicContract,canonical,sha,hash_value,same
from model_admission import ExpectedStoreIdentity

_QUARANTINE=[]
HEADER_LIMIT=16384
GAMMA_CONTAINER_LIMIT=4<<20

def flat_name(value):
    if type(value)is not str or not value or value in ('.','..')or '/'in value or '\\'in value or '\x00'in value:
        raise ValueError('flat public filename required')
    return value

@dataclass(frozen=True)
class ParameterCatalog:
    """Canonical descriptor bytes preserve every original provenance field."""
    matrices:bytes
    gammas:bytes
    gamma_container_sha256:str
    provenance:bytes
    @classmethod
    def from_public(cls,public):
        if type(public)is not PublicContract:raise TypeError('exact source-bound PublicContract required')
        public.check()
        q=public.json('qkv-export-v1/parameter_manifest.json');d=public.json('inputs/decoder_parameters.json')
        g=public.json('inputs/gamma_preparation_report.json')
        expected={}
        for layer in range(28):
            for role,width in (('q',2048),('k',1024),('v',1024)):
                expected[f'model.layers.{layer}.self_attn.{role}_proj.weight']=('qkv',[width,1024])
            for name,shape in (('self_attn.o_proj',[1024,2048]),('mlp.gate_proj',[3072,1024]),('mlp.up_proj',[3072,1024]),('mlp.down_proj',[1024,3072])):
                expected[f'model.layers.{layer}.{name}.weight']=('decoder',shape)
        for name in ('lm_head.weight','model.embed_tokens.weight'):expected[name]=('decoder',[151936,1024])
        combined={}
        for family,manifest in (('qkv',q),('decoder',d)):
            for name,description in manifest['matrices'].items():
                if name in combined or name not in expected or expected[name][0]!=family:raise ValueError('public parameter role inventory differs')
                same(description['shape'],expected[name][1],'public parameter shape differs')
                if description['name']!=name or description['dtype']not in ('float64','<f8')or description['source_dtype']!='BF16'or description['promotion_exact']is not True:
                    raise ValueError('exact promoted parameter provenance required')
                flat_name(description['file'])
                for field in ('sha256','file_sha256','source_raw_sha256'):hash_value(description[field])
                combined[name]=dict(family=family,descriptor=description)
        if set(combined)!=set(expected):raise ValueError('complete84+114 parameter inventory required')
        head=combined['lm_head.weight']['descriptor'];embedding=combined['model.embed_tokens.weight']['descriptor']
        if head['storage_alias_of']is not None or embedding['storage_alias_of']!='lm_head.weight'or embedding['exact_bf16_alias_verified']is not True:
            raise ValueError('exact tied embedding/head provenance required')
        for key in ('file','shape','dtype','sha256','file_sha256','source_raw_sha256'):
            same(embedding[key],head[key],'tied embedding/head descriptor differs')
        gamma_names={f'{i}.{role}.inverse_sqrt'for i in range(28)for role in ('input','q','k','post')}|{'final.inverse_sqrt'}
        if set(g['arrays'])!=gamma_names:raise ValueError('all113 gamma descriptors required')
        for name,row in g['arrays'].items():
            same(row['shape'],[128 if name.split('.')[1]in ('q','k')else 1024],'gamma width differs')
            if row['dtype']!='float64':raise ValueError('gamma dtype differs')
            hash_value(row['sha256'])
        return cls(canonical(combined),canonical(g['arrays']),hash_value(g['fixture_npz_sha256']),
            canonical(dict(qkv_metadata_sha256=sha(public.check()['qkv-export-v1/parameter_manifest.json']),
                decoder_metadata_sha256=sha(public.check()['inputs/decoder_parameters.json']),
                gamma_metadata_sha256=sha(public.check()['inputs/gamma_preparation_report.json']),
                historical_qkv_identity={k:q[k]for k in ('candidate_sha256','checkpoint_sha256','config_sha256')},
                historical_decoder_identity={k:d[k]for k in ('candidate_sha256','checkpoint_sha256','config_sha256')},
                gamma_asset_sha256=g['asset_sha256_before'])))
    def check(self):
        for raw in (self.matrices,self.gammas,self.provenance):
            if type(raw)is not bytes or canonical(parse(raw))!=raw:raise ValueError('immutable canonical catalog bytes required')
        hash_value(self.gamma_container_sha256)
        return self

class _FdReader:
    """Positioned reads borrow a held fd; no dup or ownership handoff."""
    def __init__(self,fd,size):self.fd=fd;self.size=size;self.position=0
    def tell(self):return self.position
    def seekable(self):return True
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.position+offset if whence==1 else self.size+offset if whence==2 else -1
        if target<0:raise ValueError('negative file position')
        self.position=target;return target
    def read(self,count=-1):
        if count<0:count=max(0,self.size-self.position)
        raw=os.pread(self.fd,min(count,max(0,self.size-self.position)),self.position)
        self.position+=len(raw);return raw

def _hash_fd(fd,start=0,stop=None):
    if stop is None:stop=os.fstat(fd).st_size
    out=hashlib.sha256();position=start
    while position<stop:
        raw=os.pread(fd,min(1<<20,stop-position),position)
        if not raw:raise ValueError('truncated held public file')
        out.update(raw);position+=len(raw)
    return out.hexdigest()

class _HeldFile:
    def __init__(self,directory,name,maximum,expected_hash,expected_file=None):
        self.directory=directory;self.name=flat_name(name);self.fd=None;self.failed=None;self.expected=hash_value(expected_hash)
        self.expected_file=None if expected_file is None else canonical(expected_file)
        try:
            directory.check()
            self.fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=directory.fd)
            info=os.fstat(self.fd)
            if not stat.S_ISREG(info.st_mode)or not 0<info.st_size<=maximum:raise ValueError('bounded regular public file required')
            self.identity=state(info);self.size=info.st_size;self.check()
            if _hash_fd(self.fd)!=self.expected:raise ValueError('public file hash differs')
            self.check()
        except BaseException as primary:
            try:self.close()
            except BaseException as secondary:
                if secondary is not primary:primary.add_note('public file setup cleanup: '+repr(secondary))
            raise
    def check(self):
        if self.failed is not None:raise self.failed
        if self.fd is None:raise ValueError('public file is closed')
        self.directory.check();visible=os.stat(self.name,dir_fd=self.directory.fd,follow_symlinks=False)
        if not stat.S_ISREG(visible.st_mode)or state(visible)!=self.identity or state(os.fstat(self.fd))!=self.identity:
            raise ValueError('held public file inode/extent changed')
        if self.expected_file is not None:
            expected=parse(self.expected_file)['state'];held=os.fstat(self.fd)
            for info in (held,visible):
                actual=dict(device=info.st_dev,inode=info.st_ino,size=info.st_size,mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns,mode=stat.S_IMODE(info.st_mode))
                same(actual,expected,'externally expected held file identity differs before body read')
    def readback(self):
        self.check()
        if _hash_fd(self.fd)!=self.expected:raise ValueError('held public bytes changed')
        self.check()
    def close(self):
        if self.failed is not None:raise self.failed
        fd=self.fd;self.fd=None
        if fd is not None:
            try:os.close(fd)
            except BaseException as error:self.failed=error;_QUARANTINE.append((self,fd,error));raise

def _header(stream,size,shape):
    version=np.lib.format.read_magic(stream)
    if version not in ((1,0),(2,0)):raise ValueError('unsupported NPY version')
    prefix=stream.read(2 if version==(1,0)else 4)
    if len(prefix)!=(2 if version==(1,0)else 4):raise ValueError('truncated NPY header length')
    length=int.from_bytes(prefix,'little')
    if not 0<length<=HEADER_LIMIT or stream.tell()+length>min(size,HEADER_LIMIT):raise ValueError('NPY header exceeds pre-read bound')
    header=stream.read(length)
    if len(header)!=length:raise ValueError('truncated NPY header')
    bounded=io.BytesIO(prefix+header)
    reader=np.lib.format.read_array_header_1_0 if version==(1,0)else np.lib.format.read_array_header_2_0
    stored,fortran,dtype=reader(bounded,max_header_size=HEADER_LIMIT)
    if type(stored)is not tuple or any(type(x)is not int for x in stored)or stored!=tuple(shape)or fortran or dtype.str!='<f8':
        raise ValueError('promoted little-endian float64 C header differs')
    offset=stream.tell()
    if offset>HEADER_LIMIT or offset+math.prod(shape)*8!=size:raise ValueError('exact NPY extent differs')
    return offset

class _MatrixLease:
    def __init__(self,directory,description,expected_file=None):
        self.file=None;self.mapping=None;self.value=None;self.failed=None
        try:
            shape=description['shape']
            if type(shape)is not list or not shape or any(type(x)is not int or x<=0 for x in shape):raise ValueError('exact positive descriptor shape required')
            self.file=_HeldFile(directory,description['file'],math.prod(shape)*8+HEADER_LIMIT,description['file_sha256'],expected_file)
            offset=_header(_FdReader(self.file.fd,self.file.size),self.file.size,shape)
            if _hash_fd(self.file.fd,offset,self.file.size)!=description['sha256']:raise ValueError('promoted raw payload hash differs')
            self.mapping=mmap.mmap(self.file.fd,0,access=mmap.ACCESS_READ)
            self.value=np.ndarray(tuple(shape),dtype='<f8',buffer=self.mapping,offset=offset,order='C')
            flat=self.value.reshape(-1)
            for start in range(0,flat.size,1<<17):
                if not np.isfinite(flat[start:start+(1<<17)]).all():raise ValueError('nonfinite public matrix')
            self.file.check()
        except BaseException as primary:
            try:self.close(primary=primary)
            except BaseException as secondary:
                if secondary is not primary:primary.add_note('matrix setup cleanup: '+repr(secondary))
            raise
    def close(self,primary=None):
        if self.failed is not None:raise self.failed
        final=Finalizer({},primary);self.value=None
        mapping=self.mapping;self.mapping=None
        if mapping is not None:
            def close_map():
                try:mapping.close()
                except BaseException as error:self.failed=error;_QUARANTINE.append((self,mapping,error));raise
            final.attempt('public mapping close',close_map)
        if self.file is not None:final.attempt('public file close',self.file.close)
        final.raise_if_failed()

class PublicParameterFiles:
    """Own public backing files; caller owns the consumer/context barriers."""
    def __init__(self,catalog,*,qkv_root,decoder_root,gamma_file,expected_store=None):
        from pathlib import Path
        if type(catalog)is not ParameterCatalog:raise TypeError('exact parameter catalog required')
        catalog.check()
        self.expected_store=expected_store;self.expected_store_state=None
        if expected_store is not None:
            if type(expected_store)is not ExpectedStoreIdentity:raise TypeError('exact immutable external store identity required')
            expected_store.check();self.expected_store_state=(expected_store,expected_store.raw,expected_store.aggregate)
            actual_catalog=dict(matrices=parse(catalog.matrices),gammas=parse(catalog.gammas),gamma_container_sha256=catalog.gamma_container_sha256,provenance=parse(catalog.provenance))
            same(actual_catalog,expected_store.aggregate.catalog,'externally expected parameter catalog differs before open')
            paths=dict(qkv=str(qkv_root),decoder=str(decoder_root),gamma=str(Path(gamma_file).parent))
            same(paths,{name:row['path']for name,row in expected_store.check()['roots'].items()},'external parameter root paths differ before open')
            if Path(gamma_file).name!=expected_store.gamma['name']:raise ValueError('external gamma filename differs before open')
        self.catalog=catalog;self.catalog_state=(catalog.matrices,catalog.gammas,catalog.gamma_container_sha256,catalog.provenance)
        self.roots={};self.matrices={};self.gamma_lease=None;self.gamma_values={};self.entered=False;self.retired=False;self.closed=False;self.failed=None
        self.gamma_path=Path(gamma_file)
        try:
            for family,path in (('qkv',qkv_root),('decoder',decoder_root),('gamma',self.gamma_path.parent)):
                self.roots[family]=HeldDirectory(path)
                if expected_store is not None:self._check_expected_root(family)
        except BaseException as primary:
            try:self.close(primary=primary)
            except BaseException as secondary:
                if secondary is not primary:primary.add_note('parameter setup cleanup: '+repr(secondary))
            raise
    def _check_expected_root(self,family):
        owner=self.roots[family];expected=self.expected_store.root(family)
        if str(owner.path)!=expected['path']or owner.path.resolve(strict=True)!=owner.path:raise ValueError('externally expected canonical root differs')
        owner.check()
        for info in (os.fstat(owner.fd),owner.path.lstat()):
            if not stat.S_ISDIR(info.st_mode):raise ValueError('externally expected root must be directory')
            same(dict(path=str(owner.path),device=info.st_dev,inode=info.st_ino,mode=stat.S_IMODE(info.st_mode)),expected,'externally expected held root differs before file open')
    def check(self):
        if self.failed is not None:raise self.failed
        if self.closed:raise ValueError('public parameter owner is closed')
        self.catalog.check()
        if (self.catalog.matrices,self.catalog.gammas,self.catalog.gamma_container_sha256,self.catalog.provenance)!=self.catalog_state:raise ValueError('public catalog changed')
        if self.expected_store_state is not None:
            expected,raw,aggregate=self.expected_store_state
            if self.expected_store is not expected or expected.raw is not raw or expected.aggregate is not aggregate:raise ValueError('original expected-store binding changed')
            expected.check()
            for family in self.roots:self._check_expected_root(family)
        for owner in self.roots.values():owner.check()
    def matrix(self,name):
        self.check();rows=parse(self.catalog.matrices)
        if type(name)is not str or name not in rows:raise ValueError('unknown public matrix name')
        row=rows[name];description=row['descriptor'];key=(row['family'],description['file'])
        expected=None if self.expected_store is None else self.expected_store.file(row['family'],description['file'])
        if key not in self.matrices:self.matrices[key]=_MatrixLease(self.roots[row['family']],description,expected)
        lease=self.matrices[key];lease.file.check();return lease.value
    def gamma(self,site):
        self.check();rows=parse(self.catalog.gammas)
        if type(site)is not str or site not in rows:raise ValueError('unknown public gamma site')
        if self.gamma_lease is None:
            expected=None if self.expected_store is None else self.expected_store.file('gamma',self.gamma_path.name)
            lease=_HeldFile(self.roots['gamma'],self.gamma_path.name,GAMMA_CONTAINER_LIMIT,self.catalog.gamma_container_sha256,expected)
            self.gamma_lease=lease
            with zipfile.ZipFile(_FdReader(lease.fd,lease.size))as archive:
                entries=archive.infolist();expected={name+'.npy'for name in rows}
                if len(entries)!=len(expected)or {e.filename for e in entries}!=expected or any(e.flag_bits&1 or e.compress_type not in (zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED)for e in entries)or sum(e.file_size for e in entries)>GAMMA_CONTAINER_LIMIT:
                    raise ValueError('exact bounded gamma NPZ inventory required')
                checked={}
                for entry in entries:
                    row=rows[entry.filename[:-4]];limit=math.prod(row['shape'])*8+HEADER_LIMIT
                    if not 0<entry.file_size<=limit:raise ValueError('gamma member exceeds declared bound')
                    raw=archive.read(entry);stream=io.BytesIO(raw);offset=_header(stream,len(raw),row['shape'])
                    payload=raw[offset:]
                    if sha(payload)!=row['sha256']:raise ValueError('gamma raw payload hash differs')
                    value=np.frombuffer(payload,dtype='<f8').reshape(tuple(row['shape']))
                    if not np.isfinite(value).all():raise ValueError('nonfinite public gamma')
                    checked[entry.filename[:-4]]=value
                lease.check();self.gamma_values=checked
        self.gamma_lease.check();return self.gamma_values[site]
    def consumer_entered(self):
        self.check()
        if self.entered or self.retired:raise ValueError('one consumer lifetime required')
        self.entered=True
    def consumers_retired(self):
        self.check()
        if not self.entered or self.retired:raise ValueError('live consumer lifetime required')
        self.retired=True
    def readback(self):
        self.check()
        for lease in self.matrices.values():lease.file.readback()
        if self.gamma_lease is not None:self.gamma_lease.readback()
    def quarantine(self,error):
        if not isinstance(error,BaseException):raise TypeError('primary failure required')
        if self.failed is None:self.failed=error;_QUARANTINE.append((self,error))
    def close(self,primary=None):
        if self.failed is not None:
            if primary is not None and primary is not self.failed:primary.add_note('public backing already quarantined')
            raise primary if primary is not None else self.failed
        if self.closed:return
        if self.entered and not self.retired:
            error=primary if primary is not None else RuntimeError('unretired consumer requires dedicated-process quarantine')
            self.quarantine(error);raise error
        final=Finalizer({},primary)
        final.attempt('public parameter final readback',self.readback)
        for lease in reversed(tuple(self.matrices.values())):final.attempt('public matrix retirement',lease.close)
        if self.gamma_lease is not None:final.attempt('public gamma retirement',self.gamma_lease.close)
        for owner in reversed(tuple(self.roots.values())):final.attempt('public directory retirement',owner.close)
        self.closed=True;self.gamma_values={}
        try:final.raise_if_failed()
        except BaseException as error:self.quarantine(error);raise

def open_source_bound_parameters(public,*,qkv_root,decoder_root,gamma_file):
    """Rederive the fixed catalog BEFORE opening any parameter directory.

    This component is still not an execution grant: an actual role's external
    admission must precede this call. Direct custom catalogs are a trusted
    synthetic/component API and are not accepted through this factory.
    """
    catalog=ParameterCatalog.from_public(public)
    return PublicParameterFiles(catalog,qkv_root=qkv_root,decoder_root=decoder_root,gamma_file=gamma_file)

def open_identity_bound_parameters(public,expected_store):
    """Trusted component only; external source/role/read grant must precede it."""
    if type(public)is not PublicContract or type(expected_store)is not ExpectedStoreIdentity:raise TypeError('exact public contract/external store identity required')
    public.check();expected_store.check()
    if public.blobs!=expected_store.aggregate.public.blobs:raise ValueError('store/public metadata differs before open')
    catalog=ParameterCatalog.from_public(public)
    roots=expected_store.check()['roots'];gamma=expected_store.gamma
    from pathlib import Path
    return PublicParameterFiles(catalog,qkv_root=roots['qkv']['path'],decoder_root=roots['decoder']['path'],gamma_file=Path(roots['gamma']['path'])/gamma['name'],expected_store=expected_store)

def open_actual_admitted_parameters(*args,**kwargs):
    raise PermissionError('actual model parameter role admission is unavailable')
