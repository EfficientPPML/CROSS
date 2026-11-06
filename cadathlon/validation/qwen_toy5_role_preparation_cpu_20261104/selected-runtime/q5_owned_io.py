"""Bounded in-memory encoding and synchronous owned-descriptor publication.

No buffered file writer survives raw descriptor retirement. Actual capacity
for payload, encoded buffer and returned bytes remains externally unresolved.
"""
import io,os,stat,re
from cleanup_origin import protect_body

CHUNK=1<<20

class BoundedBuffer(io.BytesIO):
    def __init__(self,limit):
        if type(limit)is not int or not 0<limit<=128*1024*1024:raise ValueError('bounded one-artifact buffer required')
        super().__init__();self._limit=limit
    def write(self,data):
        size=memoryview(data).nbytes
        if self.tell()+size>self._limit:raise ValueError('encoded artifact exceeds cap before write')
        return super().write(data)
    def writelines(self,lines):
        for line in lines:self.write(line)
    def seek(self,offset,whence=0):
        if type(offset)is not int or type(whence)is not int or whence not in(0,1,2):raise ValueError('exact bounded seek required')
        if whence==0:target=offset
        elif whence==1:target=self.tell()+offset
        else:
            position=self.tell();end=super().seek(0,2);super().seek(position);target=end+offset
        if not 0<=target<=self._limit:raise ValueError('encoded artifact seek exceeds cap')
        return super().seek(target,0)
    def truncate(self,size=None):
        if size is None:size=self.tell()
        if type(size)is not int or not 0<=size<=self._limit:raise ValueError('encoded artifact truncate exceeds cap')
        return super().truncate(size)

def encode(write,limit):
    """The in-memory ZIP writer is closed before any output FD is acquired."""
    buffer=BoundedBuffer(limit);raw=None;primary=None
    try:
        write(buffer);raw=buffer.getvalue()
        if type(raw)is not bytes or not 0<len(raw)<=limit:raise ValueError('complete bounded encoded bytes required')
    except BaseException as error:primary=protect_body(error);raise
    finally:
        from q5_checkpoint_transport import finish
        finish(primary,[('owned encoding buffer close',buffer.close,buffer)])
    return raw

def publish(owner,name,write,limit):
    """Borrow retained original directory owners; sole local raw-FD close."""
    from q5_checkpoint_transport import state,_hash,finish
    if type(name)is not str or re.fullmatch(r'[A-Za-z0-9_-]{1,80}\.(?:npz|npy|json)',name)is None:raise ValueError('one fixed flat artifact name required')
    owner.check();raw=encode(write,limit);owner.check()
    root=owner.root;parent=owner.parent;root_check=root.check;parent_check=parent.check
    fd=None;primary=None
    try:
        fd=os.open(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=root.fd)
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode)or before.st_size!=0 or before.st_uid!=root.euid or stat.S_IMODE(before.st_mode)!=0o600:
            raise ValueError('fresh owned file required')
        offset=0
        while offset<len(raw):
            block=memoryview(raw)[offset:offset+CHUNK];count=os.write(fd,block)
            if type(count)is not int or not 0<count<=len(block):raise OSError('invalid positive write progress')
            offset+=count
        os.fsync(fd);after=os.fstat(fd);named=os.stat(name,dir_fd=root.fd,follow_symlinks=False)
        if after.st_size!=len(raw)or state(after)!=state(named)or(before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino):
            raise ValueError('original owned artifact identity/extent differs')
        import hashlib
        digest=_hash(fd,after.st_size)
        if digest!=hashlib.sha256(raw).hexdigest():raise ValueError('original artifact readback differs')
        owner.check();root_check();parent_check();os.fsync(root.fd);os.fsync(parent.fd)
        return dict(name=name,sha256=digest,bytes=after.st_size,state=state(after))
    except BaseException as error:primary=protect_body(error);owner.failure=error;raise
    finally:
        original=fd;fd=None
        if original is not None:
            try:finish(primary,[('sole original output FD close',lambda:os.close(original),(owner,original,raw))])
            except BaseException as error:owner.failure=error;raise
