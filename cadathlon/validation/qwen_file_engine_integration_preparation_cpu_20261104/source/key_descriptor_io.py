"""Selected reviewed existing-key IO only; no key generation or writers."""
from contextlib import contextmanager
from cleanup_origin import mark,select,protect_body,is_cleanup
import hashlib,os,stat,sys
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
