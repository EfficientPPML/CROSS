"""Reviewed descriptor, publication and cleanup helpers for isolated client proof.

Copied exact function/class bodies from the completed286 server source; no
backend import or object construction. Private reads require the caller's
completed-server and reference gates before calling read.
"""
from pathlib import Path
import hashlib,io,json,os,stat,sys,tarfile,uuid,importlib.metadata,platform
_QUARANTINE=[]

def digest(raw):return hashlib.sha256(raw).hexdigest()

def parse(raw):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('duplicate public field')
            out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite public field')))

def state(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)

def read(path,maximum=16<<20):
    path=Path(path);before=path.lstat();fd=None;primary=None
    try:
        if not stat.S_ISREG(before.st_mode)or not 0<before.st_size<=maximum:raise ValueError('bounded regular public file required')
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK);opened=os.fstat(fd)
        if state(opened)!=state(before):raise ValueError('public file replaced before open')
        parts=[];left=maximum+1
        while left:
            block=os.read(fd,min(left,1<<20))
            if not block:break
            parts.append(block);left-=len(block)
        raw=b''.join(parts)
        if len(raw)!=before.st_size or state(os.fstat(fd))!=state(before)or state(path.lstat())!=state(before):raise ValueError('public file changed while reading')
        return raw
    except BaseException as error:primary=error;raise
    finally:
        if fd is not None:
            try:os.close(fd)
            except BaseException as error:
                # Never blindly retry a descriptor after an uncertain close.
                _QUARANTINE.append((fd,error))
                if primary is None:raise
                if error is not primary:primary.add_note('public descriptor close: '+repr(error))

def write(path,value):
    path=Path(path);raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp');fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);primary=None
    try:
        position=0
        while position<len(raw):
            n=os.write(fd,raw[position:])
            if n<=0:raise OSError('short report write')
            position+=n
    except BaseException as error:primary=error;raise
    finally:
        try:os.close(fd)
        except BaseException as error:
            _QUARANTINE.append((fd,error))
            if primary is None:raise
            primary.add_note('report descriptor close: '+repr(error))
    temp.replace(path)

def hashes(raw):return {n:digest(b)for n,b in raw.items()}

def archive(path,raw):
    with tarfile.open(path,'w:gz')as tar:
        for name,data in sorted(raw.items()):
            item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))

def check_archive(path,expected):
    with tarfile.open(fileobj=io.BytesIO(read(path,32<<20)))as tar:
        members=tar.getmembers()
        if len(members)!=len(expected)or len({m.name for m in members})!=len(members)or any(not m.isfile()or m.size<0 or m.size>16<<20 for m in members)or sum(m.size for m in members)>32<<20:raise ValueError('source archive inventory differs')
        actual={m.name:digest(tar.extractfile(m).read())for m in members}
    if actual!=expected:raise ValueError('source archive bytes differ')
    return True

def outside(path,protected):
    path=Path(path).absolute()
    for base in map(Path,protected):
        base=base.absolute()
        for a,b in ((path,base),(path.resolve(),base.resolve())):
            if a==b or b in a.parents or a in b.parents:raise ValueError('owned and retained paths must be disjoint')
    return path

class HeldDirectory:
    """Held inode only; never deletes contents. Uncertain close quarantines."""
    def __init__(self,path):
        self.path=Path(path);self.fd=None;self.failure=None
        try:
            self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            info=os.fstat(self.fd);self.identity=(info.st_dev,info.st_ino);self.check()
        except BaseException as primary:
            try:self.close()
            except BaseException as secondary:primary.add_note('held directory cleanup: '+repr(secondary))
            raise
    def check(self):
        if self.failure is not None:raise self.failure
        a=self.path.lstat();b=os.fstat(self.fd)
        if not stat.S_ISDIR(a.st_mode)or (a.st_dev,a.st_ino)!=self.identity or (b.st_dev,b.st_ino)!=self.identity:raise ValueError('held directory replaced')
        return True
    def close(self):
        if self.failure is not None:raise self.failure
        if self.fd is not None:
            try:os.close(self.fd)
            except BaseException as error:self.failure=error;_QUARANTINE.append(self);raise
            self.fd=None

class Finalizer:
    def __init__(self,report,primary=None):self.report=report;self.primary=primary;self.failure=None
    def attempt(self,label,fn):
        try:return fn()
        except BaseException as error:
            self.report.setdefault('cleanup_errors',[]).append(dict(operation=label,type=type(error).__name__,message=str(error)))
            if self.primary is not None:
                if error is not self.primary:self.primary.add_note(label+': '+repr(error))
            elif self.failure is None:self.failure=error
            elif not isinstance(error,Exception)and isinstance(self.failure,Exception):
                error.add_note('earlier cleanup: '+repr(self.failure));self.failure=error
            elif error is not self.failure:self.failure.add_note(label+': '+repr(error))
    def shielded(self,shield,fn):
        entered=False
        try:
            with shield():
                entered=True;fn()
        except BaseException as error:self.attempt('cleanup signal scope',lambda:(_ for _ in ()).throw(error))
        if not entered:
            # Installer failure must not skip resource retirement. This fallback
            # makes no claim of atomic process-wide signal masking.
            self.attempt('cleanup after shield installation failure',fn)
    def raise_if_failed(self):
        error=self.primary if self.primary is not None else self.failure
        if error is not None:raise error

def software_versions():
    versions={}
    for name in ('numpy','jax','jaxlib','torch','transformers'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions)

def environment():
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit CPU environment required')
    bad=[n for n in os.environ if n.startswith(('CROSS_','BOOTSTRAP_','PYTEST_','PYTHONINSPECT','PYTHONSTARTUP'))]
    if bad or os.environ.get('XLA_FLAGS','')or os.environ.get('JAX_DISABLE_JIT','').lower()not in ('','0','false'):
        raise ValueError('inherited runtime/mock override forbidden')
    return {n:os.environ.get(n)for n in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}

def finish(path,report,body,termination,shield):
    """True only after body cleanup AND outer handler restoration complete."""
    primary=None
    try:
        with termination():
            write(path,report);body(report)
        report['termination_scope_completed']=True
        report['passed']=bool(report.get('body_complete')and report.get('cleanup_completed'))
        write(path,report)
        return report
    except BaseException as error:
        primary=error;report.update(passed=False,termination_scope_completed=False,
            error=dict(type=type(error).__name__,message=str(error)))
        try:
            with shield():write(path,report)
        except BaseException as publication:
            if publication is not primary:primary.add_note('terminal republication: '+repr(publication))
        raise
