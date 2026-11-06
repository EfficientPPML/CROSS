"""Actual CPU dependency/resource contracts; functions run only after admission.

Policies are immutable external bytes authenticated by the process. This module
does not import NumPy/JAX, acquire capacity or create a context at import time.
The Python audit is an additional trusted-source guard, not an OS sandbox.
"""
import hashlib
import os
from pathlib import Path
import resource
import stat
import sys
import threading
from types import ModuleType
from cleanup_origin import protect_body
from qwen_file_contract import canonical,sha,same,hash_value
from qwen_public_parse import parse
from keygen_output import canonical_path,overlap,finish

_QUARANTINE=[]
FORBIDDEN=frozenset(('torch','transformers','tensorflow','flax','torch_xla','safetensors','tokenizers','libtpu'))
DEPENDENCY_FIELDS={'schema','interpreter','resolved_interpreter','python_major_minor','cores','environment','installed_sources','library_roots','backend_module_origins'}
RESOURCE_FIELDS={'schema','cores','threads','import_headroom_bytes','working_headroom_bytes','host_reserve_bytes',
    'live_object_allowance_bytes','descriptor_margin','minimum_soft_nofile','evidence_disk_bytes','cgroup_memory_current','cgroup_memory_max'}
POLICY_NAMES=('dependency_policy_sha256','role_policy_sha256','resource_policy_sha256','resource_review_sha256')


def object_bytes(raw,fields,schema,limit=2<<20):
    if type(raw)is not bytes or not 0<len(raw)<=limit:raise ValueError('bounded immutable policy bytes required')
    value=parse(raw)
    if type(value)is not dict or set(value)!=set(fields)or canonical(value)!=raw or value['schema']!=schema:
        raise ValueError('closed canonical policy differs')
    return value


def integer(value,minimum,label):
    if type(value)is not int or value<minimum:raise ValueError('exact positive '+label+' required')
    return value


def cores(value):
    if type(value)is not list or not value or any(type(x)is not int or x<0 for x in value)or value!=sorted(set(value)):
        raise ValueError('exact nonempty ordered CPU lane required')
    return value


def policies(raws,pins,source_digest,archive_digest):
    if type(raws)is not dict or set(raws)!=set(POLICY_NAMES):raise ValueError('all policy/review bytes required')
    for name in POLICY_NAMES:
        if type(raws[name])is not bytes or sha(raws[name])!=pins[name]:raise ValueError('externally pinned policy bytes differ')
    dep=object_bytes(raws['dependency_policy_sha256'],DEPENDENCY_FIELDS,'qwen-keygen-cpu-dependencies-v1')
    res=object_bytes(raws['resource_policy_sha256'],RESOURCE_FIELDS,'qwen-keygen-runtime-resources-v1')
    role=object_bytes(raws['role_policy_sha256'],('schema','role','source_manifest_sha256','scope'),'qwen-keygen-runtime-role-policy-v1',1<<16)
    review=object_bytes(raws['resource_review_sha256'],('schema','passed','source_manifest_sha256','source_archive_sha256',
        'dependency_policy_sha256','resource_policy_sha256','working_assumptions_reviewed','actual_execution_authorized'),
        'qwen-keygen-resource-policy-review-v1',1<<16)
    from keygen_authority import GRANT_SCOPE
    same(role,dict(schema='qwen-keygen-runtime-role-policy-v1',role='generate_fresh_qwen166_keys',
        source_manifest_sha256=source_digest,scope=GRANT_SCOPE),'independent role policy differs')
    same(review,dict(schema='qwen-keygen-resource-policy-review-v1',passed=True,source_manifest_sha256=source_digest,
        source_archive_sha256=archive_digest,dependency_policy_sha256=pins['dependency_policy_sha256'],
        resource_policy_sha256=pins['resource_policy_sha256'],working_assumptions_reviewed=True,
        actual_execution_authorized=False),'independent resource policy review differs')
    cores(dep['cores']);cores(res['cores']);same(dep['cores'],res['cores'],'CPU policies differ')
    # A reviewed virtual-environment executable may itself be a symlink. Bind
    # its exact launch spelling AND independently declared canonical target.
    executable=dep['interpreter']
    if (type(executable)is not str or not 1<len(executable)<=4096 or executable.startswith('//')
            or '\\'in executable or '\x00'in executable or not Path(executable).is_absolute()
            or str(Path(executable))!=executable or '..'in Path(executable).parts):raise ValueError('exact absolute interpreter spelling required')
    canonical_path(dep['resolved_interpreter'])
    version=dep['python_major_minor']
    if type(version)is not list or len(version)!=2 or any(type(x)is not int or x<0 for x in version):raise ValueError('exact Python version required')
    integer(res['threads'],1,'thread count')
    if res['threads']>len(res['cores']):raise ValueError('threads exceed granted CPU count')
    environment=dep['environment']
    if type(environment)is not dict or any(type(k)is not str or type(v)is not str or not 0<len(k)<=128 or len(v)>4096 for k,v in environment.items()):
        raise ValueError('bounded exact environment policy required')
    required={'JAX_PLATFORMS':'cpu','JAX_ENABLE_X64':'true','CROSS_NP_KEYGEN':'1','PYTHONDONTWRITEBYTECODE':'1',
        'OMP_NUM_THREADS':str(res['threads']),'OPENBLAS_NUM_THREADS':str(res['threads']),'MKL_NUM_THREADS':str(res['threads'])}
    if any(environment.get(k)!=v for k,v in required.items()):raise ValueError('explicit fixed CPU/branch/thread environment required')
    libraries=dep['library_roots']
    if type(libraries)is not list or not libraries or len(libraries)>32 or libraries!=sorted(set(libraries)):
        raise ValueError('exact bounded installed library roots required')
    roots=[canonical_path(v)for v in libraries]
    installed=dep['installed_sources']
    if type(installed)is not dict or not 3<=len(installed)<=16384:raise ValueError('complete bounded installed source closure required')
    for name,row in installed.items():
        p=canonical_path(name)
        if type(row)is not dict or set(row)!={'sha256','size'}:raise ValueError('exact installed file descriptor required')
        hash_value(row['sha256']);integer(row['size'],1,'installed file extent')
        if not any(p==root or root in p.parents for root in roots):raise ValueError('installed source outside declared libraries')
        if p.suffix not in('.py','.pyc','.so','.pyd')and '.so.'not in p.name:raise ValueError('only declared installed code/library files permitted')
    module_origins=dep['backend_module_origins']
    if type(module_origins)is not dict or not 3<=len(module_origins)<=32768:raise ValueError('explicit bounded backend module origin policy required')
    for name,row in module_origins.items():
        if (type(name)is not str or len(name)>512 or name.split('.')[0]not in('jax','jaxlib','numpy')
                or any(not part.isidentifier()for part in name.split('.'))):raise ValueError('exact backend module name required')
        if type(row)is not dict or set(row)!={'file','spec'}:raise ValueError('both independent backend origin fields required')
        for value in row.values():
            if value is None or type(value)is str and value in('built-in','frozen'):
                if name in('jax','jaxlib','numpy'):raise ValueError('genuine root backend origins required')
                continue
            if type(value)is not str or str(canonical_path(value))not in installed:raise ValueError('backend origin outside installed source policy')
        if name in('jax','jaxlib','numpy')and row['file']!=row['spec']:raise ValueError('genuine matching root backend origins required')
    if not all(name in module_origins for name in('jax','jaxlib','numpy')):raise ValueError('all genuine backend root origins required')
    for name in RESOURCE_FIELDS-{'schema','cores','threads','cgroup_memory_current','cgroup_memory_max'}:
        integer(res[name],1,name)
    # Reviewed source-derived floor, not a measured RSS or sufficiency claim.
    if res['live_object_allowance_bytes']<8<<30 or res['working_headroom_bytes']<res['live_object_allowance_bytes']:
        raise ValueError('working allowance below reviewed source liveness floor')
    for key,leaf in [('cgroup_memory_current','memory.current'),('cgroup_memory_max','memory.max')]:
        p=canonical_path(res[key])
        if Path('/sys/fs/cgroup')not in p.parents or p.name!=leaf:raise ValueError('exact cgroup-v2 memory metadata path required')
    if Path(res['cgroup_memory_current']).parent!=Path(res['cgroup_memory_max']).parent:raise ValueError('one cgroup memory scope required')
    return dep,res


def kernel_read(path,limit=1<<20):
    primary=None;fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        blocks=[];size=0
        while size<=limit:
            block=os.read(fd,min(65536,limit+1-size))
            if not block:break
            blocks.append(block);size+=len(block)
        if not 0<size<=limit:raise ValueError('bounded kernel metadata required')
        return b''.join(blocks).decode('ascii')
    except BaseException as error:primary=protect_body(error);raise
    finally:
        original=fd;fd=None
        def close():
            try:os.close(original)
            except BaseException as error:_QUARANTINE.append((path,original,error));raise
        finish(primary,close,'kernel metadata close')


def hash_installed(dep):
    observed={}
    for name,row in dep['installed_sources'].items():
        primary=None;fd=os.open(name,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            before=os.fstat(fd)
            if not stat.S_ISREG(before.st_mode)or before.st_size!=row['size']:raise ValueError('installed source extent differs')
            h=hashlib.sha256();count=0
            while count<row['size']:
                block=os.read(fd,min(1<<20,row['size']-count))
                if not block:raise ValueError('installed source truncated')
                h.update(block);count+=len(block)
            after=os.fstat(fd);named=os.stat(name,follow_symlinks=False)
            state=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,stat.S_IMODE(s.st_mode))
            if state(before)!=state(after)or state(before)!=state(named)or not stat.S_ISREG(named.st_mode)or h.hexdigest()!=row['sha256']:
                raise ValueError('installed source original identity/hash differs')
            observed[name]=h.hexdigest()
        except BaseException as error:primary=protect_body(error);raise
        finally:
            original=fd;fd=None
            def close():
                try:os.close(original)
                except BaseException as error:_QUARANTINE.append((name,original,error));raise
            finish(primary,close,'installed source descriptor close')
    return observed


def environment(dep):
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('fresh -I -B required')
    if list(sys.version_info[:2])!=dep['python_major_minor']or os.path.abspath(sys.executable)!=dep['interpreter']:
        raise ValueError('selected interpreter differs')
    if Path(sys.executable).resolve()!=Path(dep['resolved_interpreter']):raise ValueError('independently pinned interpreter resolution differs')
    if sorted(os.sched_getaffinity(0))!=dep['cores']:raise ValueError('actual CPU affinity differs')
    if any(os.environ.get(k)!=v for k,v in dep['environment'].items()):raise ValueError('actual CPU environment differs')


def imported_origins(dep,source_root):
    """Check every loaded backend against an independently reviewed allow-list.

    Explicit null/built-in/frozen child origins accommodate extension-created
    submodules only when named in that external policy. No missing origin is
    inferred to be safe. Declared but not yet loaded lazy modules are not forced
    to import here. All three root packages still require genuine file/spec.
    """
    observed={};root=Path(source_root)
    for name,module in tuple(sys.modules.items()):
        first=name.split('.')[0]
        if first in FORBIDDEN:raise ValueError('forbidden backend was imported')
        origins=(getattr(module,'__file__',None),getattr(getattr(module,'__spec__',None),'origin',None))
        backend=first in('jax','jaxlib','numpy')
        if backend:
            if not isinstance(module,ModuleType):raise ValueError('genuine backend module object required')
            policy=dep['backend_module_origins'].get(name)
            if policy is None:raise ValueError('backend module absent from reviewed origin policy')
            same(dict(file=origins[0],spec=origins[1]),policy,'actual backend independent origins differ')
        files=[]
        for value in origins:
            if value is None or isinstance(value,str)and value in('built-in','frozen'):continue
            if not isinstance(value,(str,os.PathLike)):
                if backend:raise ValueError('invalid backend module origin')
                continue
            p=Path(value)
            if p==root or root in p.parents:continue  # Complete local/core checks are separate.
            if not p.is_absolute()or p.resolve()!=p or str(p)not in dep['installed_sources']:
                raise ValueError('unbound installed module origin')
            files.append(str(p))
        if backend:observed[name]=dict(file=origins[0],spec=origins[1])
        elif files:observed[name]=dict(file=origins[0],spec=origins[1])
    if any(name not in sys.modules for name in('jax','jaxlib','numpy')):raise ValueError('genuine CPU dependencies not loaded')
    return observed


def capacity(intent,res,*,after_import):
    paths=intent['paths'];logical=intent['logical_storage']
    soft,_=resource.getrlimit(resource.RLIMIT_NOFILE);baseline=len(os.listdir('/proc/self/fd'))
    needed=baseline+logical['retained_store_descriptors']+logical['store_temporary_allowance']+res['descriptor_margin']
    if soft!=resource.RLIM_INFINITY and(soft<res['minimum_soft_nofile']or soft<needed):raise ValueError('descriptor baseline/headroom insufficient')
    # Match the declared cgroup to this process; never accept another group's headroom.
    cgroup=[x.split(':',2)for x in kernel_read('/proc/self/cgroup').splitlines()]
    unified=[x[2]for x in cgroup if len(x)==3 and x[:2]==['0','']]
    if len(unified)!=1:raise ValueError('one actual cgroup-v2 scope required')
    cg=Path('/sys/fs/cgroup')/unified[0].lstrip('/')
    if cg/'memory.current'!=Path(res['cgroup_memory_current']):raise ValueError('declared cgroup differs from current process')
    current=int(kernel_read(res['cgroup_memory_current']).strip());maximum=kernel_read(res['cgroup_memory_max']).strip()
    if current<0 or(maximum!='max'and(int(maximum)<current)):raise ValueError('invalid actual cgroup memory observation')
    mem=[x.split()for x in kernel_read('/proc/meminfo').splitlines()if x.startswith('MemAvailable:')]
    if len(mem)!=1 or len(mem[0])!=3 or mem[0][2]!='kB':raise ValueError('one MemAvailable kB observation required')
    available=int(mem[0][1])*1024
    status=[x.split()for x in kernel_read('/proc/self/status').splitlines()if x.startswith('VmRSS:')]
    if len(status)!=1 or len(status[0])!=3 or status[0][2]!='kB':raise ValueError('one actual RSS observation required')
    rss=int(status[0][1])*1024
    mounts=[]
    for line in kernel_read('/proc/self/mountinfo').splitlines():
        left,separator,right=line.partition(' - ')
        if not separator:raise ValueError('malformed mount metadata')
        fields=left.split();post=right.split()
        if len(fields)<6 or len(post)<3:raise ValueError('incomplete mount metadata')
        decoded=fields[4].replace('\\040',' ').replace('\\011','\t').replace('\\012','\n').replace('\\134','\\')
        mounts.append((Path(decoded),post[0]))
    by_device={};tmpfs_charge=0;observed=[]
    for name,amount in [('public_root',logical['public_required_bytes']),('private_root',logical['private_required_bytes']),('evidence_root',res['evidence_disk_bytes'])]:
        parent=canonical_path(paths[name]).parent;info=parent.lstat();space=os.statvfs(parent)
        if not stat.S_ISDIR(info.st_mode):raise ValueError('ordinary existing output parent required')
        matches=[(p,t)for p,t in mounts if p==parent or p in parent.parents]
        if not matches:raise ValueError('output mount type unavailable')
        mount,kind=max(matches,key=lambda row:len(row[0].parts))
        prior=by_device.setdefault(info.st_dev,[0,space.f_bavail*space.f_frsize])
        prior[0]+=amount;prior[1]=min(prior[1],space.f_bavail*space.f_frsize)
        if kind in('tmpfs','ramfs'):tmpfs_charge+=amount
        observed.append(dict(role=name,parent=str(parent),device=info.st_dev,mount=str(mount),filesystem=kind,required_bytes=amount))
    if any(need>free for need,free in by_device.values()):raise ValueError('combined same-filesystem store reserve insufficient')
    headroom=res['working_headroom_bytes']if after_import else res['import_headroom_bytes']
    required=headroom+res['host_reserve_bytes']+tmpfs_charge
    if available<required or(maximum!='max'and int(maximum)-current<required):raise ValueError('actual memory/cgroup headroom insufficient')
    return dict(after_import=after_import,open_fd_baseline=baseline,required_fd_capacity=needed,soft_nofile=soft,
        observed_rss_bytes=rss,mem_available_bytes=available,cgroup_current_bytes=current,cgroup_max=maximum,
        required_free_memory_bytes=required,tmpfs_new_store_charge_bytes=tmpfs_charge,filesystems=observed)


class AccessScope:
    def __init__(self,root,source_names,public_paths,intent,profile,dep,res,protected):
        self.denials=[];self.root=Path(root);self.dep=dep;self.res=res;self.enabled=False;self.active=True
        self.local=threading.local();self.original_open=os.open;self.wrapper=None
        self.public={Path(p)for p in public_paths}|{self.root/n for n in source_names}|set(map(Path,dep['installed_sources']))
        self.leaves={name:Path(path)for name,path in intent['paths'].items()}
        self.parents={p.parent for p in self.leaves.values()}
        self.owned={self.leaves['public_root']/s.name for s in profile.arrays}
        self.owned|={self.leaves['public_root']/'public_manifest.json',self.leaves['private_root']/'secret.npy',self.leaves['private_root']/'private_manifest.json'}
        self.evidence={self.leaves['evidence_root']/name for name in('report.json','failure.json')}
        self.protected=tuple(Path(p)for p in protected)
        self.kernel={Path('/proc/meminfo'),Path('/proc/self/status'),Path('/proc/self/cgroup'),Path('/proc/self/mountinfo'),Path(res['cgroup_memory_current']),Path(res['cgroup_memory_max'])}

    def deny(self,kind):
        self.denials.append(kind);raise PermissionError('keygen audit denied '+kind)

    def resolve(self,value,dir_fd=None):
        if type(value)is int:return Path(os.readlink('/proc/self/fd/'+str(value)))
        path=Path(os.fsdecode(value))
        if not path.is_absolute():path=(Path(os.getcwd())if dir_fd is None else Path(os.readlink('/proc/self/fd/'+str(dir_fd))))/path
        return path

    def access(self,p,flags):
        if type(flags)is not int or not p.is_absolute()or p.resolve()!=p or '..'in p.parts:self.deny('open alias or flags')
        write=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if write:
            need=os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW
            if not self.enabled or p not in self.owned|self.evidence or flags&need!=need or flags&(os.O_TRUNC|os.O_APPEND):self.deny('nonexclusive/unselected write')
            return
        if p in self.owned|self.evidence:
            if not self.enabled:self.deny('owned read before admission')
            return
        if any(p==r or r in p.parents for r in self.protected):self.deny('protected body')
        if p in self.public|self.parents|set(self.leaves.values())|self.kernel:return
        if p==self.root or self.root in p.parents:
            if p.is_dir():return
        self.deny('unselected read')

    def install(self):
        if os.open is not self.original_open:raise ValueError('open wrapper changed before audit')
        def wrapped(path,flags,mode=0o777,*,dir_fd=None):
            selected=self.resolve(path,dir_fd);self.access(selected,flags)
            prior=getattr(self.local,'open',None);self.local.open=(path,selected,flags)
            try:return self.original_open(path,flags,mode,dir_fd=dir_fd)
            finally:self.local.open=prior
        self.wrapper=wrapped
        def audit(event,args):
            if not self.active:return
            if event=='open':
                pending=getattr(self.local,'open',None)
                p=pending[1]if pending is not None and args[0]==pending[0]else self.resolve(args[0])
                self.access(p,args[2])
            elif event=='import'and args and str(args[0]).split('.')[0]in FORBIDDEN:self.deny('forbidden import')
            elif event.startswith('socket.')or event in('subprocess.Popen','os.fork','os.forkpty','os.posix_spawn','os.posix_spawnp','os.system','os.exec'):
                self.deny('network/process')
            elif event=='os.mkdir':
                p=self.resolve(args[0],args[2]if len(args)>2 and args[2]!=-1 else None)
                if not self.enabled or p not in self.leaves.values()or args[1]!=0o700:self.deny('unselected directory creation')
            elif event in('os.remove','os.rmdir','os.rename','os.link','os.symlink','os.truncate','os.chmod','os.chown'):
                self.deny('mutation outside exclusive create')
            elif event=='ctypes.dlopen'and args[0]is not None:
                if Path(str(args[0]))not in set(map(Path,self.dep['installed_sources'])):self.deny('unselected shared library')
        sys.addaudithook(audit);os.open=wrapped

    def check(self):
        if self.denials:raise PermissionError('sticky keygen audit denial')
        if self.active and os.open is not self.wrapper:raise ValueError('original audit wrapper changed')

    def close(self):
        primary=None
        try:self.check()
        except BaseException as error:primary=protect_body(error)
        finally:
            os.open=self.original_open;self.active=False
        if primary is not None:raise primary
