"""Retained key-file writer for owned synthetic preparation; no actual role.

Uses reviewed non-owning fdopen(closefd=False), never an owning handoff/dup.
Failed directories stay in place. Manifest existence does not certify readiness.
"""
from pathlib import Path
import hashlib,json,math,os,stat
import numpy as np
from cleanup import cleanup_signals
from cleanup_origin import protect_body,mark
from key_descriptor_io import _CloseState,_stream,_sha_fd
from qwen_finalizer import Finalizer
from qwen_key_contract import ArraySpec,canonical
from qwen_key_readers import _header

_QUARANTINE=[]
CHUNK=1<<20

def create_actual_key_store(*args,**kwargs):
    raise PermissionError('Qwen key generation actual admission is unavailable')

def _state(info):return(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
def _identity(info):return(info.st_dev,info.st_ino)
def _flat(name):
    if type(name)is not str or not name or name in ('.','..')or any(c in name for c in ('/','\\','\x00')):
        raise ValueError('flat key filename required')
    return name

def _array(value,shape,moduli):
    a=np.asarray(value)
    if a.dtype.kind not in 'ui' or a.shape!=shape:raise ValueError('keygen must return exact integral public storage shape')
    for limb,modulus in enumerate(moduli):
        x=a[...,limb,:]
        if np.any(x<0) or np.any(x>=modulus):raise ValueError('key residue outside declared modulus before uint32 conversion')
    return a.astype(np.uint32,copy=False)

class _Descriptor:
    def __init__(self,label):self.label=label;self.fd=None;self.failed=None
    def close(self):
        fd=self.fd;self.fd=None
        if fd is not None:
            try:os.close(fd)
            except BaseException as error:
                self.failed=mark(error);_QUARANTINE.append((self,fd,error));raise
        elif self.failed is not None:raise self.failed

class _StoredFile(_Descriptor):
    def __init__(self,owner,name,spec=None):
        super().__init__(name);self.owner=owner;self.name=name;self.spec=spec
        self.inode=None;self.state=None;self.digest=None;self.description=None
    def check(self):
        self.owner._directory_check()
        if self.fd is None:raise ValueError('owned file descriptor unavailable')
        held=os.fstat(self.fd);named=os.stat(self.name,dir_fd=self.owner.directory.fd,follow_symlinks=False)
        if (not stat.S_ISREG(held.st_mode)or not stat.S_ISREG(named.st_mode)or
                _identity(held)!=self.inode or _identity(named)!=self.inode or
                stat.S_IMODE(held.st_mode)!=0o600 or stat.S_IMODE(named.st_mode)!=0o600):
            raise ValueError('owned key file inode or mode differs')
        if self.state is not None and (_state(held)!=self.state or _state(named)!=self.state):
            raise ValueError('owned key file metadata changed')
    def readback(self):
        self.check()
        if self.state is None or self.digest is None:raise ValueError('incomplete owned key file')
        maximum=self.spec.raw_bytes+4096 if self.spec is not None else(1<<16 if self.owner.private else 1<<20)
        if not 0<self.state[2]<=maximum:raise ValueError('owned key file extent exceeds bound')
        if self.spec is not None:
            spec=self.spec;close_state=_CloseState();os.lseek(self.fd,0,os.SEEK_SET)
            with _stream(self.fd,'rb',close_state)as stream:offset=_header(stream,self.state[2],spec.shape)
            # Read one bounded limb chunk at a time, from the original fd.
            # No full key mapping, duplicate fd or full335-array RAM copy.
            degree=spec.shape[-1];limbs=len(spec.moduli)
            for row in range(math.prod(spec.shape[:-2])):
                for limb,modulus in enumerate(spec.moduli):
                    start=offset+(row*limbs+limb)*degree*4;remaining=degree*4
                    while remaining:
                        count=min(CHUNK,remaining);raw=os.pread(self.fd,count,start)
                        if len(raw)!=count:raise ValueError('truncated owned key residues')
                        if np.any(np.frombuffer(raw,dtype=np.uint32)>=modulus):raise ValueError('owned key residue outside modulus')
                        start+=count;remaining-=count
        if _sha_fd(self.fd)!=self.digest:raise ValueError('owned key file hash differs')
        self.check()

class KeyStore:
    """Trusted component API; custom small specs are synthetic, not admission.

    A generation role must supply the exact PublicProfile specs after its full
    independent source/grant gate. This class has no authority to activate it.
    """
    def __init__(self,root,specs,*,private=False,shield=cleanup_signals):
        if type(private)is not bool or type(specs)is not tuple or not specs:raise TypeError('exact private flag/spec tuple required')
        names=[]
        for spec in specs:
            if (type(spec)is not ArraySpec or type(spec.shape)is not tuple or len(spec.shape)<2 or
                    any(type(x)is not int or x<=0 for x in spec.shape)or type(spec.moduli)is not tuple or
                    len(spec.moduli)!=spec.shape[-2]or any(type(x)is not int or not 1<x<2**32 for x in spec.moduli)):
                raise ValueError('typed key shape/moduli required')
            names.append(_flat(spec.name))
        if len(set(names))!=len(names):raise ValueError('unique key filenames required')
        manifest='private_manifest.json'if private else'public_manifest.json'
        if manifest in names:raise ValueError('manifest is not an array')
        self.root=Path(root).absolute();self.parent_path=self.root.parent;self.specs=specs;self.private=private;self.shield=shield
        self.manifest_name=manifest;self.parent=_Descriptor('key parent');self.directory=_Descriptor('key directory')
        self.parent_identity=None;self.directory_identity=None;self.files=[];self.records=[];self.retained=[]
        self.closed=False;self.failed=None;self.completed=False;self.manifest_bytes=None
        self._fixed=(self.root,self.parent_path,specs,private,shield,self.parent,self.directory,manifest)
        if self.root.name in('','.','..')or self.root==self.parent_path:raise ValueError('fresh key child required')
        try:
            if self.parent_path.resolve(strict=True)!=self.parent_path:raise ValueError('key parent aliases another path')
            self.parent.fd=os.open(self.parent_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            self.parent_identity=_identity(os.fstat(self.parent.fd))
            if _identity(self.parent_path.lstat())!=self.parent_identity:raise ValueError('key parent changed before creation')
            os.mkdir(self.root.name,0o700,dir_fd=self.parent.fd)
            self.directory.fd=os.open(self.root.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=self.parent.fd)
            self.directory_identity=_identity(os.fstat(self.directory.fd));self._directory_check()
            os.fsync(self.parent.fd);self.check()
        except BaseException as primary:
            self.close(primary=primary);raise
    def _directory_check(self):
        if self.parent.fd is None or self.directory.fd is None:raise ValueError('owned key directory descriptor unavailable')
        parent=self.parent_path.lstat();held_parent=os.fstat(self.parent.fd)
        named=os.stat(self.root.name,dir_fd=self.parent.fd,follow_symlinks=False);held=os.fstat(self.directory.fd);direct=self.root.lstat()
        if (not stat.S_ISDIR(parent.st_mode)or _identity(parent)!=self.parent_identity or _identity(held_parent)!=self.parent_identity or
                not stat.S_ISDIR(named.st_mode)or not stat.S_ISDIR(held.st_mode)or not stat.S_ISDIR(direct.st_mode)or
                _identity(named)!=self.directory_identity or _identity(held)!=self.directory_identity or _identity(direct)!=self.directory_identity or
                any(stat.S_IMODE(x.st_mode)!=0o700 for x in (named,held,direct))):
            raise ValueError('owned key parent/root inode or mode changed')
    def check(self):
        if self.closed:raise ValueError('key store is closed')
        if self.failed is not None:raise self.failed
        current=(self.root,self.parent_path,self.specs,self.private,self.shield,self.parent,self.directory,self.manifest_name)
        if any(a is not b for a,b in zip(current,self._fixed)):raise ValueError('key store binding changed')
        self._directory_check()
        if set(os.listdir(self.directory.fd))!={f.name for f in self.files}:raise ValueError('owned key inventory differs')
        for file in self.files:file.check()
    def _open(self,name,spec=None):
        self.check();file=_StoredFile(self,name,spec);self.files.append(file)
        file.fd=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_NONBLOCK,0o600,dir_fd=self.directory.fd)
        file.inode=_identity(os.fstat(file.fd));file.check();return file
    def _seal(self,file):
        os.fsync(file.fd);file.state=_state(os.fstat(file.fd))
        maximum=file.spec.raw_bytes+4096 if file.spec is not None else(1<<16 if self.private else 1<<20)
        if not 0<file.state[2]<=maximum:raise ValueError('new key file extent exceeds bound')
        file.digest=_sha_fd(file.fd)
        file.readback();os.fsync(self.directory.fd);file.check()
    def write_array(self,name,value):
        try:
            self.check()
            if self.completed or type(name)is not str or len(self.records)>=len(self.specs)or name!=self.specs[len(self.records)].name:
                raise ValueError('complete ordered key arrays required')
            spec=self.specs[len(self.records)];self.retained.append(value);a=_array(value,spec.shape,spec.moduli);self.retained.append(a)
            file=self._open(name,spec);state=_CloseState()
            with _stream(file.fd,'wb',state)as stream:np.save(stream,a,allow_pickle=False);stream.flush()
            self._seal(file)
            record=dict(sha256=file.digest,dtype='uint32',shape=list(spec.shape),bytes=file.state[2]);spec.validate(record)
            file.description=canonical(record);self.records.append((name,file.description));self.retained.clear();self.check();return record
        except BaseException as primary:
            self.close(primary=primary);raise
    def verify(self):
        self.check()
        if len(self.records)!=len(self.specs):raise ValueError('incomplete key inventory')
        for file in self.files:file.readback()
        self.check();return {name:json.loads(raw)for name,raw in self.records}
    def write_manifest(self,raw):
        try:
            if type(raw)is not bytes or not 0<len(raw)<=(1<<16 if self.private else 1<<20):raise ValueError('bounded immutable manifest bytes required')
            self.verify()
            if self.completed:raise ValueError('manifest already published')
            file=self._open(self.manifest_name);offset=0
            while offset<len(raw):
                piece=memoryview(raw)[offset:offset+CHUNK];count=os.write(file.fd,piece)
                if type(count)is not int or not 0<count<=len(piece):raise OSError('invalid or short-progress manifest write')
                offset+=count
            self._seal(file)
            if file.digest!=hashlib.sha256(raw).hexdigest()or file.state[2]!=len(raw):raise ValueError('manifest readback differs')
            self.manifest_bytes=raw;self.completed=True;self.verify();os.fsync(self.directory.fd);os.fsync(self.parent.fd);self.check()
            return raw
        except BaseException as primary:
            self.close(primary=primary);raise
    def identity(self):
        self.check();return dict(root=str(self.root),device=self.directory_identity[0],inode=self.directory_identity[1],private=self.private)
    def close(self,primary=None):
        protect_body(primary)
        if self.closed:
            if primary is not None:raise primary
            if self.failed is not None:raise self.failed
            return
        parent=self.parent;directory=self.directory;files=tuple(self.files);retained=tuple(self.retained);shield=self.shield
        directory_check=self._directory_check;reads=tuple(f.readback for f in files);closes=tuple(f.close for f in files)
        dir_close=directory.close;parent_close=parent.close
        if primary is None and not self.completed:primary=protect_body(ValueError('incomplete key store cannot close successfully'))
        report={};final=Finalizer(report,primary)
        def retire():
            if directory.fd is not None:
                final.attempt('key directory final identity',directory_check)
                for file,readback in zip(files,reads):
                    if file.fd is not None and file.state is not None:final.attempt('key file final readback',readback)
                final.attempt('key directory fsync',lambda:os.fsync(directory.fd))
            for close in reversed(closes):final.attempt('key file close',close)
            final.attempt('key directory close',dir_close);final.attempt('key parent close',parent_close)
        try:final.shielded(shield,retire)
        finally:self.closed=True
        chosen=final.selected()
        if chosen is not None:
            self.failed=chosen;_QUARANTINE.append((self,parent,directory,files,retained,reads,closes,report,chosen));raise chosen
        self.retained.clear()
