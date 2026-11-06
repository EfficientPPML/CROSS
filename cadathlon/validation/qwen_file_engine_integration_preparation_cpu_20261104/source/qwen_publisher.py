"""Exclusive Qwen file publication. Actual role admission remains unavailable.

Directories/files are retained on failure; no replacement or borrowed path is
removed. A manifest's existence is not terminal evidence: a reader still needs
its independent producer/outer gate after all close/fsync/source obligations.
"""
from pathlib import Path
import hashlib,json,os,stat
from cleanup import cleanup_signals,_attempt
from cleanup_origin import protect_body,mark,select
from qwen_file_contract import (PublicContract,canonical,sha,same,hash_value,validate_request,
    ciphertext_binding,artifact,request_manifest,response)
from qwen_serialization import Serialized,serialize_array,serialize_ciphertext

_QUARANTINE=[]
CHUNK=1<<20


def create_actual_publisher(*args,**kwargs):
    raise PermissionError('Qwen publisher actual role requires independently admitted source/grant/context bindings')


def _identity(info):return (info.st_dev,info.st_ino)
def _state(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


class _Descriptor:
    def __init__(self,label):self.label=label;self.fd=None;self.failed=None
    def close(self):
        # Detach BEFORE the syscall; an uncertain close is never retried.
        fd=self.fd;self.fd=None
        if fd is not None:
            try:os.close(fd)
            except BaseException as error:
                self.failed=mark(error);_QUARANTINE.append((self,fd,error));raise


class _File(_Descriptor):
    def __init__(self,owner,name,raw):
        super().__init__(name);self.owner=owner;self.name=name;self.size=len(raw);self.digest=sha(raw);self.state=None;self.inode=None
    def check(self):
        self.owner._directory_check()
        held=os.fstat(self.fd);named=os.stat(self.name,dir_fd=self.owner.directory.fd,follow_symlinks=False)
        if not stat.S_ISREG(named.st_mode) or _identity(named)!=self.inode or _identity(held)!=self.inode:
            raise ValueError('owned file inode replaced')
        if self.state is not None and (_state(held)!=self.state or _state(named)!=self.state):
            raise ValueError('owned file metadata changed')
    def readback(self):
        self.check();digest=hashlib.sha256();offset=0
        while offset<self.size:
            part=os.pread(self.fd,min(CHUNK,self.size-offset),offset)
            if not part:raise ValueError('owned file truncated during readback')
            offset+=len(part);digest.update(part)
        if os.pread(self.fd,1,self.size)or digest.hexdigest()!=self.digest:
            raise ValueError('owned file content differs')
        self.check()


class Publisher:
    """Trusted component API for a single fresh request or response directory.

    Pure PublicContract validation is not execution authority. A future actual
    factory must independently bind expected IDs/grant/protected paths first.
    This class is exercised only on owned synthetic directories in preparation.
    """
    def __init__(self,public,request,role,root,*,input_digest=None,shield=cleanup_signals):
        if type(public)is not PublicContract:raise TypeError('exact public contract required')
        c=validate_request(public,request)
        if type(role)is not str or role not in('request','response'):raise ValueError('exact role required')
        if role=='request'and input_digest is not None:raise ValueError('request has no prior input digest')
        if role=='response':hash_value(input_digest)
        self.public=public;self.request=canonical(c);self.role=role;self.input_digest=input_digest
        self.count=c['input']['input_tile_count']if role=='request'else 38
        self.root=Path(root).absolute();self.parent_path=self.root.parent
        self.parent=_Descriptor('parent');self.directory=_Descriptor('directory')
        self.parent_identity=None;self.directory_identity=None;self.files=[];self.artifacts=[]
        self.published=False;self.closed=False;self.failed=None;self.manifest_bytes=None;self.shield=shield
        self._fixed=(public,self.request,role,input_digest,self.root,self.parent_path,self.parent,self.directory,shield)
        if self.root.name in('','.','..')or self.root==self.parent_path:raise ValueError('fresh child directory required')
        try:
            self.parent.fd=os.open(self.parent_path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            parent_info=os.fstat(self.parent.fd);self.parent_identity=_identity(parent_info)
            if _identity(self.parent_path.lstat())!=self.parent_identity:raise ValueError('parent path changed before creation')
            os.mkdir(self.root.name,mode=0o700,dir_fd=self.parent.fd)
            self.directory.fd=os.open(self.root.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=self.parent.fd)
            self.directory_identity=_identity(os.fstat(self.directory.fd));self._directory_check()
            os.fsync(self.parent.fd);self.check()
        except BaseException as primary:
            self.close(primary=primary);raise

    def _directory_check(self):
        if self.parent.fd is None or self.directory.fd is None:raise ValueError('owned directory descriptor unavailable')
        parent=self.parent_path.lstat();held_parent=os.fstat(self.parent.fd)
        named=os.stat(self.root.name,dir_fd=self.parent.fd,follow_symlinks=False);held=os.fstat(self.directory.fd)
        direct=self.root.lstat()
        if (not stat.S_ISDIR(parent.st_mode)or _identity(parent)!=self.parent_identity or _identity(held_parent)!=self.parent_identity or
            not stat.S_ISDIR(named.st_mode)or _identity(named)!=self.directory_identity or _identity(held)!=self.directory_identity or
            not stat.S_ISDIR(direct.st_mode)or _identity(direct)!=self.directory_identity):
            raise ValueError('owned output directory or parent replaced')

    def check(self):
        if self.closed:raise ValueError('publisher already closed')
        if self.failed is not None:raise self.failed
        current=(self.public,self.request,self.role,self.input_digest,self.root,self.parent_path,self.parent,self.directory,self.shield)
        if any(a is not b for a,b in zip(current,self._fixed)):raise ValueError('publisher binding changed')
        validate_request(self.public,json.loads(self.request));self._directory_check()
        if set(os.listdir(self.directory.fd))!={f.name for f in self.files}:
            raise ValueError('exclusive owned file inventory differs')
        for file in self.files:file.check()

    def _write(self,name,raw):
        self.check()
        if type(raw)is not bytes:raise TypeError('immutable payload bytes required')
        file=_File(self,name,raw);self.files.append(file)
        file.fd=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_NONBLOCK,0o600,dir_fd=self.directory.fd)
        info=os.fstat(file.fd)
        if not stat.S_ISREG(info.st_mode):raise ValueError('new regular output required')
        file.inode=_identity(info);file.check()
        offset=0
        while offset<len(raw):
            part=memoryview(raw)[offset:offset+CHUNK];written=os.write(file.fd,part)
            if type(written)is not int or not 0<written<=len(part):raise OSError('invalid or zero-progress write')
            offset+=written
        os.fsync(file.fd);file.state=_state(os.fstat(file.fd))
        if file.state[2]!=len(raw):raise ValueError('written file extent differs')
        file.readback();os.fsync(self.directory.fd);file.check()
        return file

    def _binding(self,index,actual_scale):
        self.check()
        if self.published or type(index)is not int or index!=len(self.artifacts)or index>=self.count:
            raise ValueError('one complete ordered output sequence required')
        return ciphertext_binding(self.public,json.loads(self.request),self.role,index,actual_scale=actual_scale,input_digest=self.input_digest)

    def _publish_serialized(self,index,serialized,binding):
        if type(serialized)is not Serialized:raise TypeError('exact opaque serializer result required')
        raw=serialized.check(binding)
        name=('tile'if self.role=='request'else'block')+f'{index:02}.npz'
        self._write(name,raw)
        row=artifact(self.public,json.loads(self.request),self.role,index,payload_sha256=sha(raw),actual_scale=binding['scale'],input_digest=self.input_digest)
        self.artifacts.append(canonical(row));self.check();return json.loads(canonical(row))

    def write_payload(self,index,payload,*,moduli,degree_layout,precision,actual_scale,nsd):
        try:
            binding=self._binding(index,actual_scale)
            raw=serialize_array(payload,binding,moduli=moduli,degree_layout=degree_layout,precision=precision,actual_scale=actual_scale,nsd=nsd)
            return self._publish_serialized(index,raw,binding)
        except BaseException as primary:
            self.close(primary=primary);raise

    def write_ciphertext(self,index,value,binding):
        try:
            expected=self._binding(index,getattr(value,'_ckks_scale',None));same(binding,expected,'opaque publisher complete binding differs')
            raw=serialize_ciphertext(value,expected)
            return self._publish_serialized(index,raw,expected)
        except BaseException as primary:
            self.close(primary=primary);raise

    def finish(self):
        try:
            self.check()
            if self.published or len(self.artifacts)!=self.count:raise ValueError('complete ordered payload inventory required before manifest')
            for file in self.files:file.readback()
            c=json.loads(self.request);rows=[json.loads(v)for v in self.artifacts]
            value=request_manifest(self.public,c,rows)if self.role=='request'else response(self.public,c,self.input_digest,rows)
            raw=canonical(value);self._write('request.json'if self.role=='request'else'response.json',raw)
            # Only this return declares a complete manifest; partial bytes after
            # an IO failure are retained and never become successful evidence.
            for file in self.files:file.readback()
            os.fsync(self.directory.fd);os.fsync(self.parent.fd);self.check()
            self.manifest_bytes=raw;self.published=True;return raw
        except BaseException as primary:
            self.close(primary=primary);raise

    def close(self,primary=None):
        protect_body(primary)
        if self.closed:
            if primary is not None:raise primary
            if self.failed is not None:raise self.failed
            return
        # Capture originals before cleanup actions can mutate/rethrow anything.
        parent=self.parent;directory=self.directory;files=tuple(self.files);shield=self.shield
        errors=[];entered=False
        if primary is None and not self.published:
            primary=protect_body(ValueError('incomplete publication cannot be successful'))
        def retire():
            if directory.fd is not None:
                _attempt(errors,'final directory identity',self._directory_check)
                for file in files:
                    if file.fd is not None:_attempt(errors,'final owned file readback',file.readback)
                _attempt(errors,'final directory fsync',lambda:os.fsync(directory.fd))
            for file in reversed(files):_attempt(errors,'owned file close',file.close)
            _attempt(errors,'owned directory close',directory.close)
            _attempt(errors,'owned parent close',parent.close)
        try:
            try:
                with shield():entered=True;retire()
            except BaseException as error:errors.append(('cleanup shield',mark(error)))
            if not entered:retire()
        finally:self.closed=True
        chosen=select(primary,errors)
        if chosen is not None:
            self.failed=chosen;_QUARANTINE.append((self,parent,directory,files,chosen));raise chosen

    def __enter__(self):self.check();return self
    def __exit__(self,kind,value,tb):self.close(primary=value);return False
