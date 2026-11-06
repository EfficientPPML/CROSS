"""Bounded public reads and fresh held-parent evidence publication.

Adapted from reviewed checker141 Output; all cleanup uses shared4543. No
borrowed-store write, buffered stream, replacement, deletion, or close retry.
"""
import hashlib
import os
import stat
from pathlib import Path
from cleanup_origin import protect_body, select
from qwen_finalizer import Finalizer

_QUARANTINE=[]


def identity(info): return (info.st_dev,info.st_ino)
def state(info): return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,stat.S_IMODE(info.st_mode))


def canonical_path(value):
    if type(value)is not str or not 1<len(value)<=4096 or value.startswith('//') or '\\'in value or '\x00'in value:
        raise ValueError('canonical absolute path required')
    path=Path(value)
    if not path.is_absolute() or str(path)!=value or '..'in path.parts or path==path.parent or path.resolve()!=path:
        raise ValueError('resolved canonical non-root path required')
    return path


def overlap(a,b): return a==b or a in b.parents or b in a.parents


def finish(primary,operation,label):
    protect_body(primary)
    try: operation()
    except BaseException as error:
        chosen=select(primary,[(label,error)])
        if chosen is not primary: raise chosen


def read(path,*,limit,expected=None):
    path=canonical_path(str(path))
    if type(limit)is not int or limit<=0: raise ValueError('exact positive read bound required')
    primary=None
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=limit:
            raise ValueError('bounded ordinary public input required')
        blocks=[];size=0
        while size<=limit:
            block=os.read(fd,min(1<<16,limit+1-size))
            if not block: break
            blocks.append(block);size+=len(block)
        raw=b''.join(blocks);after=os.fstat(fd);named=path.lstat()
        if state(before)!=state(after) or state(before)!=state(named) or not stat.S_ISREG(named.st_mode) or len(raw)!=before.st_size or size>limit:
            raise ValueError('public input identity/extent changed')
        if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError('public input digest differs')
        return raw
    except BaseException as error:
        primary=protect_body(error);raise
    finally:
        original_fd=fd;fd=None
        def close():
            try: os.close(original_fd)
            except BaseException as error:
                _QUARANTINE.append((path,original_fd,error));raise
        finish(primary,close,'public read descriptor close')


class Output:
    """Fresh output leaf; an existing trusted parent may have protected siblings."""
    def __init__(self,path):
        self.path=canonical_path(str(path));self.parent=self.path.parent
        self.parent_fd=None;self.fd=None;self.failure=None
        self.parent_identity=None;self.owner=None;self.parent_policy=None
        primary=None
        try:
            self.parent_fd=os.open(self.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
            p=os.fstat(self.parent_fd)
            self.parent_identity=identity(p);self.parent_policy=(p.st_uid,stat.S_IMODE(p.st_mode))
            self._parent_check()
            os.mkdir(self.path.name,0o700,dir_fd=self.parent_fd)
            self.fd=os.open(self.path.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=self.parent_fd)
            self.owner=identity(os.fstat(self.fd));self.check()
        except BaseException as error:
            primary=protect_body(error);self.close(primary=primary);raise

    def _parent_check(self):
        if self.parent.resolve()!=self.parent: raise ValueError('output parent alias differs')
        held=os.fstat(self.parent_fd);visible=self.parent.lstat()
        for info in (held,visible):
            mode=stat.S_IMODE(info.st_mode)
            if not stat.S_ISDIR(info.st_mode) or identity(info)!=self.parent_identity or (info.st_uid,mode)!=self.parent_policy:
                raise ValueError('original output parent identity differs')
            if info.st_uid not in (0,os.getuid()) or (mode&0o022 and not mode&stat.S_ISVTX):
                raise ValueError('output parent is not owned or sticky trusted storage')

    def check(self):
        if self.failure is not None: raise self.failure
        if self.fd is None or self.parent_fd is None: raise ValueError('output is closed')
        self._parent_check()
        if self.path.resolve()!=self.path: raise ValueError('output leaf alias differs')
        held=os.fstat(self.fd);visible=os.stat(self.path.name,dir_fd=self.parent_fd,follow_symlinks=False)
        for info in (held,visible):
            if not stat.S_ISDIR(info.st_mode) or identity(info)!=self.owner or info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)!=0o700:
                raise ValueError('original fresh output identity differs')

    def binding(self):
        self.check()
        return (str(self.path),self.parent_identity,self.parent_policy,self.owner)

    @classmethod
    def reopen(cls,binding):
        value=cls.__new__(cls)
        path,parent_identity,parent_policy,owner=binding
        value.path=canonical_path(path);value.parent=value.path.parent
        value.parent_identity=tuple(parent_identity);value.parent_policy=tuple(parent_policy);value.owner=tuple(owner)
        value.parent_fd=None;value.fd=None;value.failure=None
        try:
            value.parent_fd=os.open(value.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
            value._parent_check()
            value.fd=os.open(value.path.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC,dir_fd=value.parent_fd)
            value.check();return value
        except BaseException as error:
            primary=protect_body(error);value.close(primary=primary);raise

    def write(self,name,raw):
        if type(name)is not str or name not in ('report.json','failure.json') or type(raw)is not bytes or not 0<len(raw)<=2<<20:
            raise ValueError('one bounded fresh evidence document required')
        self.check();primary=None
        fd=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=self.fd)
        try:
            before=os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or stat.S_IMODE(before.st_mode)!=0o600 or before.st_uid!=os.getuid() or before.st_size!=0:
                raise ValueError('fresh owned evidence file required')
            position=0
            while position<len(raw):
                count=os.write(fd,raw[position:position+(1<<16)])
                if type(count)is not int or not 0<count<=min(1<<16,len(raw)-position):
                    raise OSError('evidence write made invalid progress')
                position+=count
            os.fsync(fd)
            out=hashlib.sha256();position=0
            while position<len(raw):
                block=os.pread(fd,min(1<<16,len(raw)-position),position)
                if not block: raise ValueError('truncated evidence readback')
                out.update(block);position+=len(block)
            final=os.fstat(fd);named=os.stat(name,dir_fd=self.fd,follow_symlinks=False)
            if (identity(final)!=identity(before) or state(final)!=state(named)
                    or not stat.S_ISREG(named.st_mode) or final.st_size!=len(raw)
                    or out.hexdigest()!=hashlib.sha256(raw).hexdigest()):
                raise ValueError('original evidence file/hash differs')
            self.check();os.fsync(self.fd);self.check()
        except BaseException as error:
            primary=protect_body(error);raise
        finally:
            original_fd=fd;fd=None
            def close():
                try: os.close(original_fd)
                except BaseException as error:
                    self.failure=error;_QUARANTINE.append((self,original_fd,error));raise
            finish(primary,close,'evidence file close')

    def close(self,primary=None):
        # Even when a prior file close failed, attempt the other original FDs.
        final=Finalizer({},primary)
        if self.failure is not None:
            error=self.failure
            final.attempt('prior uncertain evidence close',lambda:(_ for _ in ()).throw(error))
        for name in ('fd','parent_fd'):
            fd=getattr(self,name);setattr(self,name,None)
            if fd is None: continue
            def close(original=fd):
                try: os.close(original)
                except BaseException as error:
                    self.failure=error;_QUARANTINE.append((self,original,error));raise
            final.attempt('evidence '+name+' close',close)
        final.raise_if_failed()
