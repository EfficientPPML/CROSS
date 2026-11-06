"""Client dependency/source-origin policy, adapted from held keygen85.

Seven stdlib source/read/environment/origin helper function bodies are copied
exactly. Client policy schemas and body/audit scopes are separate. No installed
policy instance is supplied, and no backend is imported by this module.
"""
import hashlib
import os
from pathlib import Path
import stat
import sys
import threading
from types import ModuleType
from cleanup_origin import protect_body
from qwen_file_contract import canonical,sha,same,hash_value
from qwen_public_parse import parse
from model_output import canonical_path,overlap,finish
from qwen_client_resource import authenticate_policy
_QUARANTINE=[]

FORBIDDEN=frozenset(('torch','transformers','tensorflow','flax','torch_xla','safetensors','tokenizers','libtpu'))

DEPENDENCY_FIELDS={'schema','interpreter','resolved_interpreter','python_major_minor','cores','environment','installed_sources','library_roots','backend_module_origins'}


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



def policies(packet):
    packet.check();pins=packet.phase.conditional.expected_pins.check()
    dep=object_bytes(packet.dependency_policy,DEPENDENCY_FIELDS,'qwen-client-cpu-dependencies-v1')
    res=authenticate_policy(packet.phase,packet.resource_policy,packet.resource_review)
    role=object_bytes(packet.role_policy,('schema','roles','actual_entry_ready','default_decoder_sha256','scope'),'qwen-client-role-policy-v1',1<<16)
    same(role,dict(schema='qwen-client-role-policy-v1',roles=['request_producer','response_consumer'],actual_entry_ready=True,
        default_decoder_sha256='bc978cb7412b55416f7ed634a689ae616a263b0f47fd9eef996ba244f2505db6',
        scope=dict(key_generation=False,model_body_reads=False,scientific_acceptance=False,request_original_embeddings=True,response_complete_vocabulary=True)),
        'reviewed implemented actual client role policy required')
    cores(dep['cores']);same(dep['cores'],res['cores'],'client CPU policies differ')
    executable=dep['interpreter']
    if (type(executable)is not str or not 1<len(executable)<=4096 or executable.startswith('//')
            or '\\'in executable or '\x00'in executable or not Path(executable).is_absolute()
            or str(Path(executable))!=executable or '..'in Path(executable).parts):raise ValueError('exact absolute interpreter spelling required')
    canonical_path(dep['resolved_interpreter'])
    version=dep['python_major_minor']
    if type(version)is not list or len(version)!=2 or any(type(x)is not int or x<0 for x in version):raise ValueError('exact Python version required')
    environment=dep['environment']
    if type(environment)is not dict or any(type(k)is not str or type(v)is not str or not 0<len(k)<=128 or len(v)>4096 for k,v in environment.items()):
        raise ValueError('bounded exact environment policy required')
    required={'JAX_PLATFORMS':'cpu','JAX_ENABLE_X64':'true','PYTHONDONTWRITEBYTECODE':'1'}
    if any(environment.get(k)!=v for k,v in required.items()):raise ValueError('explicit CPU/x64/no-bytecode environment required')
    thread_values=[environment.get(n)for n in('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')]
    if (any(type(v)is not str or not v.isdecimal()or not 1<=int(v)<=len(dep['cores'])for v in thread_values)
            or len(set(thread_values))!=1):raise ValueError('exact externally chosen client thread count required')
    if environment.get('CROSS_SKIP_TOPLEVEL_ROTKEYS')=='1':raise ValueError('runtime key generation mode is forbidden')
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
    return dep,res


def exact_environment(dep):
    environment(dep)
    if dict(os.environ)!=dep['environment']:raise ValueError('actual environment has an undeclared value')


class AccessScope:
    """Exact client file sets with explicit input-before-private transitions.

    This is a Python audit guard for this source-bound dedicated process, not an
    OS sandbox. Metadata resolution/stat calls are not claimed to be hidden.
    Neither the public packet nor this object is independent launch authority.
    """
    def __init__(self,root,source_names,public_paths,phase,profile,dep,res,protected):
        phase.check();self.phase=phase;self.dep=dep;self.res=res;self.root=Path(root)
        self.denials=[];self.active=True;self.enabled=False;self.input_enabled=False;self.keys_enabled=False
        self.local=threading.local();self.original_open=os.open;self.wrapper=None
        grant=parse(phase.grant_bytes);self.role=grant['role']
        self.public=frozenset({Path(p)for p in public_paths}|{self.root/n for n in source_names}|set(map(Path,dep['installed_sources'])))
        self.source_directories=frozenset({self.root}|{p for name in source_names for p in (self.root/name).parents if p==self.root or self.root in p.parents})
        self.leaves={name:Path(row['path'])for name,row in grant['outputs'].items()}
        self.parents=frozenset(p.parent for p in self.leaves.values())
        self.key_roots=frozenset(Path(row['path'])for row in grant['key_stores'].values())
        public_root=Path(grant['key_stores']['public']['path']);private_root=Path(grant['key_stores']['private']['path'])
        self.keys=frozenset({public_root/s.name for s in profile.arrays}|{public_root/'public_manifest.json',private_root/'secret.npy',private_root/'private_manifest.json'})
        result=self.leaves['result'];evidence=self.leaves['evidence']
        if self.role=='request_producer':
            count=parse(phase.request_bytes)['input']['input_tile_count']
            self.owned=frozenset({result/f'tile{i:02}.npz'for i in range(count)}|{result/'request.json'})
            embedding=Path(grant['input']['path']);self.inputs=frozenset((embedding,));self.input_roots=frozenset((embedding.parent,))
        else:
            response=Path(grant['input']['root']['path'])
            self.inputs=frozenset({response/f'block{i:02}.npz'for i in range(38)}|{response/'response.json'})
            self.input_roots=frozenset((response,))
            self.owned=frozenset((result/'logits.npy',result/'result.json'))
        self.evidence=frozenset((evidence/'report.json',evidence/'failure.json'))
        self.protected=tuple(Path(p)for p in protected)
        self.kernel=frozenset(map(Path,('/proc/meminfo','/proc/self/status','/proc/self/cgroup','/proc/self/mountinfo',res['cgroup_memory_current'],res['cgroup_memory_max'])))
        self._fixed=(phase,dep,res,self.root,self.public,self.source_directories,self.parents,self.key_roots,self.keys,self.owned,self.inputs,self.input_roots,self.evidence,self.protected,self.kernel)

    def deny(self,kind):
        self.denials.append(kind);raise PermissionError('client audit denied '+kind)

    def resolve(self,value,dir_fd=None):
        if type(value)is int:return Path(os.readlink('/proc/self/fd/'+str(value)))
        path=Path(os.fsdecode(value))
        if not path.is_absolute():path=(Path(os.getcwd())if dir_fd is None else Path(os.readlink('/proc/self/fd/'+str(dir_fd))))/path
        return path

    def access(self,p,flags):
        if type(flags)is not int or not p.is_absolute()or '..'in p.parts:self.deny('open spelling or flags')
        # Protected trees cannot be turned into an allowed body by a suffix or
        # broad installed-library root; exact owned/input/key sets are separate.
        if any(p==r or r in p.parents for r in self.protected):self.deny('protected body')
        if p.resolve()!=p:self.deny('open alias')
        write=bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if write:
            need=os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW
            if not self.enabled or p not in self.owned|self.evidence or flags&need!=need or flags&(os.O_TRUNC|os.O_APPEND):self.deny('nonexclusive/unselected write')
            return
        if p in self.owned|self.evidence:
            if not self.enabled:self.deny('owned read before admission')
            return
        if p in self.keys|self.key_roots:
            if not self.keys_enabled:self.deny('key read before complete input admission')
            return
        if p in self.inputs|self.input_roots:
            if not self.input_enabled:self.deny('input read before phase admission')
            return
        if p in self.public|self.parents|set(self.leaves.values())|self.kernel|self.source_directories:return
        self.deny('unselected read')

    def install(self):
        if os.open is not self.original_open:raise ValueError('open wrapper changed before audit')
        def wrapped(path,flags,mode=0o777,*,dir_fd=None):
            selected=self.resolve(path,dir_fd);self.access(selected,flags)
            previous=getattr(self.local,'open',None);self.local.open=(path,selected,flags)
            try:return self.original_open(path,flags,mode,dir_fd=dir_fd)
            finally:self.local.open=previous
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
        current=(self.phase,self.dep,self.res,self.root,self.public,self.source_directories,self.parents,self.key_roots,self.keys,self.owned,self.inputs,self.input_roots,self.evidence,self.protected,self.kernel)
        if any(a is not b for a,b in zip(current,self._fixed)):raise ValueError('original client audit scope changed')
        if self.denials:raise PermissionError('sticky client audit denial')
        if not self.active or os.open is not self.wrapper:raise ValueError('original active client audit wrapper changed')

    def close(self):
        primary=None
        try:self.check()
        except BaseException as error:primary=protect_body(error)
        finally:os.open=self.original_open;self.active=False
        if primary is not None:raise primary
