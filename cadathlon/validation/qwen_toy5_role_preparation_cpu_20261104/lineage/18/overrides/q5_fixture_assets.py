"""Fixed public toy weights, owned NPY files and independent manifest binding.

No learned-model or existing body is read by source preparation. Setup generation
is deferred behind the actual role's still-denied admission. Server only uses
PublicAssets; it never calls the toy generator or derives clear embeddings.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib,io,math,os,stat
from cleanup_origin import protect_body
from qwen_finalizer import Finalizer
from q5_key_contract import PublicProfile,parse as compact_parse,canonical as compact,sha,bindings
from q5_checkpoint_contract import canonical,parse,numerical_plan
from q5_checkpoint_transport import HeldRoot,Exporter,descriptor,state,_hash,finish
_QUARANTINE=[]
_OWNERS={}

def catalog(profile):
    if type(profile)is not PublicProfile:raise TypeError('exact source-bound Q5 profile required')
    profile.check();report=compact_parse(profile.plan_bytes,1<<20)['diagnostic'];rows=report['public_weight_hashes']
    if type(rows)is not dict or len(rows)!=35:raise ValueError('exact original35 public weights required')
    for name,row in rows.items():
        if type(name)is not str or type(row)is not dict or set(row)!={'shape','dtype','sha256'}or row['dtype']!='float64':raise ValueError('fixed public weight metadata differs')
        if type(row['shape'])is not list or any(type(n)is not int or n<=0 for n in row['shape']):raise ValueError('typed public weight shape required')
    return rows

def model_id(profile):return sha(compact(dict(schema='qwen-toy5-fixed-weight-content-v1',bindings=bindings(),arrays=catalog(profile))))

def _check_array(np,array,row):
    if type(array)is not np.ndarray or array.dtype!=np.dtype('float64')or list(array.shape)!=row['shape']or not np.isfinite(array).all():
        raise ValueError('exact finite original public weight array required')
    if sha(array.tobytes(order='C'))!=row['sha256']:raise ValueError('original seeded public weight content differs')

def generate_setup_assets(np,profile,*,check):
    """Deferred original RNG call, with source/grant/resource guard before it."""
    check()
    from q5_toy_weights import toy_weights
    case=numerical_plan()['case'];weights=toy_weights(case['config'],case['weight_seed']);check()
    rows=catalog(profile)
    if set(weights)!=set(rows):raise ValueError('complete fixed35 public weight inventory required')
    owned={}
    for name in sorted(rows):
        _check_array(np,weights[name],rows[name]);owned[name]=np.frombuffer(weights[name].tobytes(order='C'),dtype=np.float64).reshape(tuple(rows[name]['shape']))
    embeddings=owned['model.embed_tokens.weight'][np.asarray(case['token_ids'],dtype=np.int64)]
    expected=compact_parse(profile.plan_bytes,1<<20)['diagnostic']['public_embedding_sha256']
    if embeddings.shape!=(2,4,16)or sha(embeddings.tobytes(order='C'))!=expected:raise ValueError('original fixed embedding content differs')
    embeddings=np.frombuffer(embeddings.tobytes(order='C'),dtype=np.float64).reshape((2,4,16));check()
    return owned,embeddings

@dataclass(frozen=True)
class AssetBinding:
    profile:PublicProfile
    raw:bytes
    expected_sha256:str
    expected_root:bytes
    def check(self):
        if type(self)is not AssetBinding or type(self.raw)is not bytes or not 0<len(self.raw)<=1<<20 or sha(self.raw)!=self.expected_sha256:
            raise ValueError('independently pinned public asset manifest required')
        rows=catalog(self.profile);v=parse(self.raw)
        if type(v)is not dict or set(v)!={'schema','bindings','model_id','root','arrays','terminal_authority'}or canonical(v)!=self.raw:
            raise ValueError('closed original public asset manifest required')
        if v['schema']!='qwen-toy5-public-assets-v1'or v['bindings']!=bindings()or v['model_id']!=model_id(self.profile)or v['terminal_authority']is not False:
            raise ValueError('fixed Q5 public asset identity differs')
        if canonical(v['root'])!=self.expected_root:raise ValueError('external original asset root differs')
        descriptor(self.expected_root)
        if type(v['arrays'])is not dict or set(v['arrays'])!=set(rows):raise ValueError('complete35 original asset files required')
        for i,name in enumerate(sorted(rows)):
            record=v['arrays'][name]
            if type(record)is not dict or set(record)!={'name','sha256','bytes','state','content'}or record['name']!=f'weight_{i:02d}.npy'or record['content']!=rows[name]:raise ValueError('ordered exact public asset descriptors required')
            if type(record['bytes'])is not int or not math.prod(rows[name]['shape'])*8<record['bytes']<=math.prod(rows[name]['shape'])*8+524:raise ValueError('bounded NPY asset extent required')
        return v

class AssetWriter(Exporter):
    def __init__(self,np,profile,output,expected_parent):
        self.profile=profile;self.profile_bytes=(profile.profile_bytes,profile.plan_bytes)
        super().__init__(np,output,expected_parent,{},bindings())
    def write_assets(self,weights):
        self.check();rows=catalog(self.profile)
        if self.records or self.finished or type(weights)is not dict or set(weights)!=set(rows):raise ValueError('one complete original weight publication required')
        result={}
        for i,name in enumerate(sorted(rows)):
            _check_array(self.np,weights[name],rows[name]);value=weights[name]
            record=self._write(f'weight_{i:02d}.npy',lambda stream,value=value:self.np.save(stream,value,allow_pickle=False),value.nbytes+524)
            record['content']=rows[name];result[name]=record;self.records.append(record)
            _check_array(self.np,value,rows[name]);self.check()
        raw=canonical(dict(schema='qwen-toy5-public-assets-v1',bindings=bindings(),model_id=model_id(self.profile),root=parse(self.root.raw),arrays=result,terminal_authority=False))
        self._write('assets_manifest.json',lambda stream:stream.write(raw),1<<20);self.check();self.finished=True
        owner=PublicAssets(self.np,AssetBinding(self.profile,raw,sha(raw),self.root.raw));primary=None
        try:owner.load();owner.check()
        except BaseException as error:primary=protect_body(error);raise
        finally:finish(primary,[('original owned asset readback close',lambda:owner.close(primary),owner)])
        return raw

class PublicAssets:
    """Hold original root and35 FDs; immutable arrays survive through graph use."""
    def __init__(self,np,binding):
        if type(binding)is not AssetBinding:raise TypeError('exact public asset binding required')
        binding.check();self.np=np;self.binding=binding;self.root=None;self.weights=None;self.closed=False
        r=dict(owner=self,np=np,binding=binding,root=None,root_close=None,root_check=None,files=[],weights=None,entered=False,closed=False,failure=None)
        _OWNERS[id(self)]=r
        try:
            root=HeldRoot(binding.expected_root);r['root']=root;self.root=root;r['root_close']=root.close;r['root_check']=root.check
        except BaseException as error:r['failure']=protect_body(error);_QUARANTINE.append((self,r,error));raise
    def _record(self):
        r=_OWNERS.get(id(self))
        if r is None or r['owner']is not self:raise ValueError('original public asset owner missing')
        return r
    def _binding(self):
        r=self._record()
        if self.np is not r['np']or self.binding is not r['binding']or self.root is not r['root']or self.weights is not r['weights']or self.closed is not r['closed']:
            raise ValueError('original public asset binding changed')
        if r['closed']:raise ValueError('public assets closed')
        if r['failure']is not None:raise r['failure']
        r['binding'].check();r['root_check']();return r
    def load(self):
        r=self._binding()
        if r['entered']:raise ValueError('public asset load one-use')
        r['entered']=True;v=r['binding'].check();np=r['np'];weights={}
        try:
            for name,record in sorted(v['arrays'].items()):
                fd=os.open(record['name'],os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=r['root'].fd)
                held=[fd,record];r['files'].append(held);before=os.fstat(fd)
                if not stat.S_ISREG(before.st_mode)or before.st_uid!=os.geteuid()or stat.S_IMODE(before.st_mode)!=0o600 or state(before)!=record['state']or before.st_size!=record['bytes']:
                    raise ValueError('original owned public weight identity differs')
                if _hash(fd,before.st_size)!=record['sha256']:raise ValueError('original public weight file digest differs')
                raw=os.pread(fd,before.st_size,0)
                if len(raw)!=before.st_size or raw[:6]!=b'\x93NUMPY'or raw[6:8]not in(b'\x01\x00',b'\x02\x00'):raise ValueError('complete NPY1/2 asset required')
                width=2 if raw[6]==1 else 4;length=int.from_bytes(raw[8:8+width],'little')
                if not 0<length<=512 or 8+width+length+math.prod(record['content']['shape'])*8!=len(raw):raise ValueError('asset header pre-read extent differs')
                stream=io.BytesIO(raw)
                try:array=np.load(stream,allow_pickle=False,max_header_size=512)
                finally:stream.close()
                _check_array(np,array,record['content']);weights[name]=np.frombuffer(array.tobytes(order='C'),dtype=np.float64).reshape(array.shape)
            r['weights']=weights;self.weights=weights;self.check();return weights
        except BaseException as error:r['failure']=protect_body(error);_QUARANTINE.append((self,r,error));raise
    def check(self):
        r=self._binding();v=r['binding'].check()
        if r['weights']is None or set(r['weights'])!=set(v['arrays']):raise ValueError('complete35 held public weights required')
        for fd,record in r['files']:
            held=os.fstat(fd);named=os.stat(record['name'],dir_fd=r['root'].fd,follow_symlinks=False)
            if state(held)!=record['state']or state(named)!=record['state']or _hash(fd,held.st_size)!=record['sha256']:raise ValueError('original public weight changed during use')
        for name,array in r['weights'].items():_check_array(r['np'],array,v['arrays'][name]['content'])
        return r['weights']
    def close(self,primary=None):
        r=self._record()
        if r['closed']:
            if primary is not None:raise primary
            if r['failure']is not None:raise r['failure']
            return
        original_files=tuple(r['files']);final=Finalizer({},primary)
        final.attempt('original public assets before close',self._binding)
        for held in reversed(original_files):
            fd=held[0];held[0]=None
            if fd is not None:final.attempt('sole original weight FD close',lambda fd=fd:os.close(fd))
        if r['root_close']is not None:final.attempt('original asset root close',lambda:r['root_close'](primary))
        r['closed']=True;self.closed=True;error=final.selected()
        if error is not None or r['failure']is not None:
            error=error if error is not None else r['failure'];r['failure']=error;_QUARANTINE.append((self,r,original_files,error));raise error
        r['weights']=None;self.weights=None;r['files'].clear()
