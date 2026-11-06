"""Owned opaque six-field NPZ exports and held-root public readback.

No key/model/private codec is imported. NumPy is supplied by an explicitly
admitted caller. Actual role factories remain unavailable in this source draft.
"""
import hashlib,io,math,os,re,stat,struct,sys,zipfile
from pathlib import Path
from cleanup_origin import protect_body,select
from q5_checkpoint_contract import canonical,parse,numerical_plan,checkpoint_row,all_checkpoints
from q5_actual_metadata_replay import check_refreshes
_QUARANTINE=[]
_PUBLISH_OWNERS={}
def _same_method(a,b):
    return (getattr(a,'__self__',None)is getattr(b,'__self__',None)and
            getattr(a,'__func__',a)is getattr(b,'__func__',b))
def _publisher_owners(owner):
    record=_PUBLISH_OWNERS.get(id(owner))
    if record is None or record['owner']is not owner:raise ValueError('original publisher ownership record missing')
    return record
def _publisher_bindings(owner):
    record=_publisher_owners(owner)
    for name,obj,check,close in record['roots']:
        if getattr(owner,name)is not obj or not _same_method(obj.check,check)or not _same_method(obj.close,close):
            raise ValueError('original publisher root or callback changed')
    return record

_FIELDS=('payload','moduli','degree_layout','precision','scale','nsd')

def state(info):
    return dict(device=info.st_dev,inode=info.st_ino,size=info.st_size,mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns,mode=stat.S_IMODE(info.st_mode),uid=info.st_uid)

def directory(info,path):
    return dict(path=str(path),device=info.st_dev,inode=info.st_ino,mode=stat.S_IMODE(info.st_mode),uid=info.st_uid)

def path(value):
    if type(value)is not str or not value.startswith('/')or value.startswith('//')or '\\'in value or '\x00'in value:raise ValueError('canonical absolute path required')
    p=Path(value)
    if str(p)!=value or '..'in p.parts or p==p.parent or p.resolve()!=p:raise ValueError('resolved ordinary non-root path required')
    return p

def finish(primary,actions):
    protect_body(primary);errors=[]
    for label,fn,owner in actions:
        try:fn()
        except BaseException as error:
            _QUARANTINE.append((label,owner,error));errors.append((label,error))
    chosen=select(primary,errors)
    if chosen is not None and chosen is not primary:raise chosen

def descriptor(raw):
    if type(raw)is not bytes or not 0<len(raw)<=8192:raise ValueError('immutable external root descriptor required')
    row=parse(raw)
    if type(row)is not dict or set(row)!={'path','device','inode','mode','uid'}or canonical(row)!=raw:raise ValueError('closed canonical root descriptor required')
    path(row['path'])
    for k in ('device','inode','mode','uid'):
        if type(row[k])is not int or row[k]<(1 if k=='inode'else 0):raise ValueError('typed root descriptor required')
    if row['mode']>4095:raise ValueError('root mode out of range')
    return row

class HeldRoot:
    def __init__(self,expected,*,parent=False):
        row=descriptor(expected);self.raw=expected;self.path=path(row['path']);self.expected=row;self.parent=parent
        self.euid=os.geteuid();self.fd=None;self._original=None;self.failure=None
        self._fixed=(self.raw,self.path,self.euid,self.parent)
        self._policy(row);self._named()  # BEFORE any body or directory open.
        primary=None
        try:
            self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC);self._original=self.fd;self.check()
        except BaseException as error:
            primary=protect_body(error);self.close(primary);raise
    def _policy(self,row):
        if self.parent:
            if row['uid']not in(0,self.euid)or(row['mode']&0o022 and not row['mode']&stat.S_ISVTX):raise ValueError('trusted owned/sticky original parent required')
        elif row['uid']!=self.euid or row['mode']!=0o700:raise ValueError('original effective-UID0700 root required')
    def _named(self):
        info=self.path.lstat()
        if not stat.S_ISDIR(info.st_mode)or directory(info,self.path)!=self.expected:raise ValueError('named original root differs')
        return info
    def check(self):
        if self.failure is not None:raise self.failure
        if (self.raw,self.path,self.euid,self.parent)!=self._fixed or descriptor(self.raw)!=self.expected or os.geteuid()!=self.euid or self.path.resolve()!=self.path:raise ValueError('immutable root binding differs')
        self._policy(self.expected);self._named()
        if self.fd is None or self.fd!=self._original:raise ValueError('original directory FD differs or closed')
        info=os.fstat(self.fd)
        if not stat.S_ISDIR(info.st_mode)or directory(info,self.path)!=self.expected:raise ValueError('held original root differs')
    def close(self,primary=None):
        original=self._original;self._original=None;self.fd=None
        if original is None:return
        def retire():
            try:os.close(original)
            except BaseException as error:self.failure=error;raise
        finish(primary,[('original root FD close',retire,(self,original))])


def _name(name):
    if type(name)is not str or re.fullmatch(r'[a-zA-Z0-9_-]{1,80}\.npz',name)is None:raise ValueError('one exact relative NPZ name required')
    return name

def _hash(fd,size):
    h=hashlib.sha256();offset=0
    while offset<size:
        block=os.pread(fd,min(1<<20,size-offset),offset)
        if not block:raise ValueError('truncated original artifact')
        h.update(block);offset+=len(block)
    return h.hexdigest()

def _array_spec(np,meta):
    n=2*(meta['level']+1)
    return dict(payload=((1,2,256,512,n),np.dtype('uint32')),moduli=((n,),np.dtype('int64')),
                degree_layout=((2,),np.dtype('int64')),precision=((),np.dtype('int64')),scale=((),np.dtype('float64')),nsd=((),np.dtype('int64')))

def _read_npz(np,stream,meta):
    """Read only stored six-array NPZ after every declared header extent is bounded."""
    specs=_array_spec(np,meta);archive=None;primary=None
    try:
        archive=zipfile.ZipFile(stream)
        infos=archive.infolist()
        if len(infos)!=6 or {r.filename for r in infos}!={n+'.npy'for n in _FIELDS}:raise ValueError('exact six NPZ fields required')
        for info in infos:
            name=info.filename[:-4];shape,dtype=specs[name];extent=math.prod(shape)*dtype.itemsize
            if info.compress_type!=zipfile.ZIP_STORED or info.flag_bits&1 or info.compress_size!=info.file_size or not extent+10<=info.file_size<=extent+524:raise ValueError('bounded uncompressed NPY extent required')
            member=None;inner=None
            try:
                member=archive.open(info);prefix=member.read(8)
                if len(prefix)!=8 or prefix[:6]!=b'\x93NUMPY' or prefix[6:]not in(b'\x01\x00',b'\x02\x00'):raise ValueError('fixed NPY1/2 header required')
                width=2 if prefix[6]==1 else 4;length_raw=member.read(width)
                if len(length_raw)!=width:raise ValueError('truncated NPY length')
                length=int.from_bytes(length_raw,'little')
                if not 0<length<=512 or 8+width+length+extent!=info.file_size:raise ValueError('header cap checked before header read')
                raw=member.read(length)
                if len(raw)!=length:raise ValueError('truncated bounded header')
                header=io.BytesIO(length_raw+raw)
                reader=np.lib.format.read_array_header_1_0 if width==2 else np.lib.format.read_array_header_2_0
                observed,fortran,actual_dtype=reader(header,max_header_size=512)
                if observed!=shape or fortran is not False or actual_dtype!=dtype or actual_dtype.hasobject:raise ValueError('exact typed NPY descriptor required')
            except BaseException as error:inner=protect_body(error);raise
            finally:
                if member is not None:finish(inner,[('zip member close',member.close,member)])
    except BaseException as error:primary=protect_body(error);raise
    finally:
        if archive is not None:finish(primary,[('zip directory close',archive.close,archive)])
    stream.seek(0);loaded=None;primary=None
    try:
        loaded=np.load(stream,allow_pickle=False,max_header_size=512)
        arrays={n:loaded[n]for n in _FIELDS}
        for name,array in arrays.items():
            shape,dtype=specs[name]
            if type(array)is not np.ndarray or array.shape!=shape or array.dtype!=dtype:raise ValueError('exact full array required')
        expected_q=numerical_plan()['profile']['q_towers'][:2*(meta['level']+1)]
        if not np.array_equal(arrays['moduli'],np.array(expected_q,dtype=np.int64))or arrays['degree_layout'].tolist()!=[256,512]or int(arrays['precision'])!=32 or float(arrays['scale'])!=meta['scale']or int(arrays['nsd'])!=1:raise ValueError('actual six-field metadata differs')
        for j,q in enumerate(expected_q):
            if np.any(arrays['payload'][...,j]>=q):raise ValueError('RNS residue outside exact Q prefix')
        for array in arrays.values():array.flags.writeable=False
        return arrays
    except BaseException as error:primary=protect_body(error);raise
    finally:
        if loaded is not None:finish(primary,[('NPZ reader close',loaded.close,loaded)])

class ArtifactReader:
    """Borrow one externally pinned root; retain its original FD across all47 admission."""
    def __init__(self,np,root_descriptor):self.np=np;self.root=HeldRoot(root_descriptor);self.closed=False
    def read(self,artifact,*,retain_payload=False):
        if type(artifact)is not dict or set(artifact)!={'name','sha256','bytes','state','metadata'}:raise ValueError('closed externally pinned artifact required')
        name=_name(artifact['name']);row=artifact['metadata'];checkpoint_row(row,row['index'])
        if type(artifact['sha256'])is not str or len(artifact['sha256'])!=64 or any(c not in '0123456789abcdef'for c in artifact['sha256']):raise ValueError('exact artifact digest required')
        observed=artifact['state']
        if type(observed)is not dict or set(observed)!={'device','inode','size','mtime_ns','ctime_ns','mode','uid'}or any(type(v)is not int or v<0 for v in observed.values())or observed['inode']<=0:raise ValueError('exact artifact original-state fields required')
        cap=numerical_plan()['checkpoints'][row['index']]['raw_payload_bytes']+128*1024
        if type(artifact['bytes'])is not int or not 0<artifact['bytes']<=cap:raise ValueError('bounded exact artifact extent required')
        self.root.check();fd=None;stream=None;arrays=None;primary=None
        try:
            fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC,dir_fd=self.root.fd)
            before=os.fstat(fd)
            if not stat.S_ISREG(before.st_mode)or before.st_uid!=self.root.euid or stat.S_IMODE(before.st_mode)!=0o600 or state(before)!=artifact['state']or before.st_size!=artifact['bytes']:raise ValueError('original0600 artifact identity differs')
            if _hash(fd,before.st_size)!=artifact['sha256']:raise ValueError('artifact body hash differs')
            stream=os.fdopen(fd,'rb',closefd=False);arrays=_read_npz(self.np,stream,row)
            named=os.stat(name,dir_fd=self.root.fd,follow_symlinks=False)
            if state(os.fstat(fd))!=state(before)or state(named)!=state(before)or not stat.S_ISREG(named.st_mode):raise ValueError('artifact changed during readback')
            self.root.check()
            return arrays['payload'] if retain_payload else dict(artifact)
        except BaseException as error:primary=protect_body(error);raise
        finally:
            actions=[]
            if stream is not None:actions.append(('artifact stream close',stream.close,stream))
            if fd is not None:actions.append(('original artifact FD close',lambda original=fd:os.close(original),(fd,arrays)))
            finish(primary,actions)
    def check(self):self.root.check()
    def close(self,primary=None):
        if self.closed:return
        self.closed=True;self.root.close(primary)

class Exporter:
    """Fresh output,46 ordered writes; input is an external original reference."""
    def __init__(self,np,output_path,expected_parent,original_input,bindings):
        self.np=np;self.path=path(output_path);self.parent=None;self.root=None;self.records=[];self.finished=False;self.failure=None;self.closed=False
        if descriptor(expected_parent)['path']!=str(self.path.parent):raise ValueError('granted output parent mismatch')
        self.input_bytes=canonical(original_input);self.bindings_bytes=canonical(bindings)
        primary=None;record=dict(owner=self,roots=[],closed=False);_PUBLISH_OWNERS[id(self)]=record
        try:
            self.parent=HeldRoot(expected_parent,parent=True)
            record['roots'].append(('parent',self.parent,self.parent.check,self.parent.close));self.parent.check()
            os.mkdir(self.path.name,0o700,dir_fd=self.parent.fd)
            observed=os.stat(self.path.name,dir_fd=self.parent.fd,follow_symlinks=False)
            self.root=HeldRoot(canonical(directory(observed,self.path)))
            record['roots'].append(('root',self.root,self.root.check,self.root.close))
            self._fixed=(self.np,self.path,self.parent,self.root,self.input_bytes,self.bindings_bytes,self.records);self.check()
        except BaseException as error:primary=protect_body(error);self.close(primary);raise
    def check(self):
        if self.closed:raise ValueError('exporter closed')
        if any(a is not b for a,b in zip((self.np,self.path,self.parent,self.root,self.input_bytes,self.bindings_bytes,self.records),self._fixed)):raise ValueError('original exporter owners/bindings changed')
        if self.failure is not None:raise self.failure
        for name,owner,check,close in _publisher_bindings(self)['roots']:check()
    def _write(self,name,write,cap):
        from q5_owned_io import publish
        return publish(self,name,write,cap)
    def checkpoint(self,index,value,row):
        from q5_checkpoint_arithmetic import state as capture,same_state,metadata
        self.check();checkpoint_row(row,index)
        if index!=len(self.records):raise ValueError('strict ordered47 callbacks required')
        held=capture(value,numerical_plan()['profile']);barrier=held[1].block_until_ready;barrier()
        if index==0:
            original=parse(self.input_bytes)
            if original['artifact']['metadata']!=row:raise ValueError('admitted original input metadata differs')
            self.records.append(original['artifact']);same_state(held,numerical_plan()['profile']);return
        if self.finished:raise ValueError('checkpoint after finish')
        np=self.np;payload=np.array(held[1],copy=True)
        if payload.dtype!=np.dtype('uint32')or payload.shape!=tuple(numerical_plan()['checkpoints'][index]['planned_shape']):raise ValueError('original opaque payload shape/dtype required')
        payload.flags.writeable=False;meta=metadata(value,numerical_plan()['profile'])
        fields=dict(payload=payload,moduli=np.asarray(value.moduli,dtype=np.int64),degree_layout=np.asarray((256,512),dtype=np.int64),
                    precision=np.asarray(32,dtype=np.int64),scale=np.asarray(meta['scale'],dtype=np.float64),nsd=np.asarray(meta['nsd'],dtype=np.int64))
        name=f'checkpoint_{index:02d}.npz';cap=payload.nbytes+128*1024
        result=self._write(name,lambda stream:np.savez(stream,**fields),cap);result['metadata']=parse(canonical(row))
        reader=ArtifactReader(np,self.root.raw);primary=None
        try:reader.read(result)
        except BaseException as error:primary=protect_body(error);raise
        finally:finish(primary,[('owned readback root close',lambda:reader.close(primary),reader)])
        same_state(held,numerical_plan()['profile']);self.check();self.records.append(result)
    def finish(self,result):
        self.check()
        if self.finished or result.get('complete')is not True or len(self.records)!=47:raise ValueError('complete47 required; partial server stays failed')
        rows=[r['metadata']for r in self.records];all_checkpoints(rows);check_refreshes(result['refresh_observations'],rows)
        if canonical(rows)!=canonical(result['checkpoints']):raise ValueError('producer/export trace differs')
        manifest=dict(schema='qwen-toy5-checkpoint-manifest-v1',bindings=parse(self.bindings_bytes),original_input=parse(self.input_bytes),
                      output_root=parse(self.root.raw),artifacts=self.records[1:],rows=rows,refresh_observations=result['refresh_observations'],
                      complete=True,logical_checkpoints=47,new_exports=46,terminal_authority=False)
        raw=canonical(manifest)
        if len(raw)>1<<20:raise ValueError('bounded manifest required')
        self._write('manifest.json',lambda stream:stream.write(raw),1<<20);self.check();self.finished=True
        return raw
    def close(self,primary=None):
        record=_publisher_owners(self)
        if record['closed']:
            if primary is not None:raise primary
            if self.failure is not None:raise self.failure
            return
        record['closed']=True;self.closed=True
        originals=tuple(record['roots']);actions=[('original exporter root bindings',lambda:_publisher_bindings(self),(self,record))]
        for name,owner,check,close in reversed(originals):
            actions.append(('original export '+name+' close',lambda close=close:close(primary),owner))
            actions.append(('post-close original export bindings',lambda:_publisher_bindings(self),(self,record)))
        try:finish(primary,actions)
        except BaseException as error:self.failure=error;_QUARANTINE.append((self,record,originals,error));raise


class InputWriter(Exporter):
    """One original L49 setup artifact, same opaque six fields and ownership."""
    def __init__(self,np,output_path,expected_parent,bindings):
        super().__init__(np,output_path,expected_parent,{},bindings)
    def write_input(self,value,row):
        from q5_checkpoint_arithmetic import state as capture,same_state,metadata
        self.check();checkpoint_row(row,0)
        if self.records or self.finished:raise ValueError('exact one original input required')
        held=capture(value,numerical_plan()['profile']);held[1].block_until_ready()
        np=self.np;payload=np.array(held[1],copy=True)
        if payload.shape!=(1,2,256,512,100)or payload.dtype!=np.dtype('uint32'):
            raise ValueError('fresh original L49 input shape/dtype required')
        actual=metadata(value,numerical_plan()['profile'])
        if actual!=dict(level=49,scale=numerical_plan()['profile']['scale'],nsd=1)or any(row[k]!=actual[k]for k in actual):
            raise ValueError('fresh canonical-scale input required')
        fields=dict(payload=payload,moduli=np.asarray(value.moduli,dtype=np.int64),degree_layout=np.asarray((256,512),dtype=np.int64),
                    precision=np.asarray(32,dtype=np.int64),scale=np.asarray(actual['scale'],dtype=np.float64),nsd=np.asarray(1,dtype=np.int64))
        record=self._write('input.npz',lambda stream:np.savez(stream,**fields),payload.nbytes+128*1024)
        record['metadata']=parse(canonical(row));reader=ArtifactReader(np,self.root.raw);primary=None
        try:reader.read(record)
        except BaseException as error:primary=protect_body(error);raise
        finally:finish(primary,[('original input readback close',lambda:reader.close(primary),reader)])
        same_state(held,numerical_plan()['profile']);self.check();self.records.append(record)
        return dict(root=parse(self.root.raw),artifact=record)
    def finish(self,result):
        self.check()
        if self.finished or len(self.records)!=1:raise ValueError('one fully written original input required')
        original=dict(root=parse(self.root.raw),artifact=self.records[0])
        if canonical(result)!=canonical(original):raise ValueError('original input return differs')
        raw=canonical(dict(schema='qwen-toy5-input-manifest-v1',bindings=parse(self.bindings_bytes),original_input=original))
        self._write('input_manifest.json',lambda stream:stream.write(raw),1<<20)
        self.check();self.finished=True;return raw
