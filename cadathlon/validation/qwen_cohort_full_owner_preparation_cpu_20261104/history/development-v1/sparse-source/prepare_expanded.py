"""Isolated public/synthetic preparation only; no actual execution mode.

External manifest admission precedes local helpers. The caller separately owns
direct wait and group-absence evidence. Python audit is not an OS sandbox.
"""
from pathlib import Path
import argparse,hashlib,importlib.abc,importlib.util,json,os,site,stat,sys,sysconfig,time,xml.etree.ElementTree as ET

TESTS=('test_expanded_contract.py','test_expanded_fixture.py','test_full_plan.py','test_full_owner.py',
       'test_profile.py','test_setup_scope.py','test_full_callback_negative.py','test_full_returned.py','test_full_entry_failure.py','test_wrapper_callback.py')
EXPECTED_TESTS=180
FORBIDDEN=('jax','jaxlib','torch','transformers','safetensors','tokenizers','tensorflow','flax','torch_xla','libtpu')
HISTORICAL=('collect_focused','collect_public','collect_tail','verify_public','verify_tail','audit_public','apply_successor')
SYNTHETIC=('jaxite_word','jaxite_word.bootstrapping','encrypted_packed_qwen','encrypted_refresh','support')
INFRASTRUCTURE_EXIT=70
def digest(raw):return hashlib.sha256(raw).hexdigest()
def parse(raw):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def bounded(path,limit=8*1024**2):
    """Reviewed observer bootstrap read/identity/close-origin pattern."""
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW|os.O_CLOEXEC);primary=None
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=limit:raise ValueError('bounded regular source required')
        chunks=[];size=0
        while size<=limit:
            block=os.read(fd,min(65536,limit+1-size))
            if not block:break
            chunks.append(block);size+=len(block)
        after=os.fstat(fd);named=os.stat(path,follow_symlinks=False)
        fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if fields(before)!=fields(after) or (named.st_dev,named.st_ino)!=(before.st_dev,before.st_ino) or not stat.S_ISREG(named.st_mode) or size!=before.st_size or size>limit:raise ValueError('source changed during read')
        return b''.join(chunks)
    except BaseException as error:
        primary=error
        if getattr(error,'_qwen_observer_cleanup_origin',False)is not True:error._qwen_observer_body_origin=True
        raise
    finally:
        try:os.close(fd)
        except BaseException as error:
            if getattr(error,'_qwen_observer_body_origin',False)is not True:error._qwen_observer_cleanup_origin=True
            if primary is None:raise
            if getattr(primary,'_qwen_observer_body_origin',False)is not True and getattr(primary,'_qwen_observer_cleanup_origin',False)is True and isinstance(primary,Exception) and not isinstance(error,Exception):raise error from primary
            if error is not primary:primary.add_note('bootstrap close failure: '+type(error).__name__)
def source_map(root,expected):
    actual=set()
    def walk(directory,prefix=''):
        with os.scandir(directory) as entries:
            for e in entries:
                name=prefix+e.name
                if e.is_symlink():raise ValueError('symlink in source closure')
                if e.is_dir(follow_symlinks=False):
                    if not any(n.startswith(name+'/') for n in expected):raise ValueError('undeclared source directory/namespace')
                    walk(e.path,name+'/')
                elif e.is_file(follow_symlinks=False):actual.add(name)
                else:raise ValueError('nonregular source member')
    walk(root)
    if actual!=set(expected):raise ValueError('undeclared or missing source member')
    blobs={n:bounded(root/n) for n in expected}
    if sum(map(len,blobs.values()))>32*1024**2:raise ValueError('total public source limit')
    hashes={n:digest(b) for n,b in blobs.items()}
    if hashes!=expected:raise ValueError('source closure differs')
    return hashes,blobs
def local_modules(expected):
    lookup={}
    for name in expected:
        if name.endswith('.py') and ('/'not in name or name.startswith('checkout/cadathlon/common/')):
            stem=Path(name).stem
            if stem in lookup:raise ValueError('ambiguous local module')
            lookup[stem]=name
    return lookup
def preloaded(root,original,lookup):
    for name,module in tuple(sys.modules.items()):
        if name.split('.')[0] in FORBIDDEN or name in HISTORICAL or name in SYNTHETIC or name in lookup and name!='__main__':raise ValueError('preloaded backend/local helper')
        values=(getattr(module,'__file__',None),getattr(getattr(module,'__spec__',None),'origin',None))
        for value in values:
            if not isinstance(value,(str,os.PathLike)) or value in ('built-in','frozen'):continue
            path=Path(value).resolve()
            if path.is_relative_to(root) or path.is_relative_to(original):
                if name!='__main__' or path!=root/'prepare_expanded.py':raise ValueError('preloaded local origin/alias')
def output_path(value,roots):
    path=Path(value).absolute()
    for root in roots:
        if path==root or path.is_relative_to(root) or root.is_relative_to(path):raise ValueError('output/source overlap')
    resolved=path.resolve()
    for root in roots:
        if resolved==root or resolved.is_relative_to(root) or root.is_relative_to(resolved):raise ValueError('resolved output/source overlap')
    return path
class FrozenLoader(importlib.abc.Loader):
    def __init__(self,path,data):self.path,self.data=path,data
    def create_module(self,spec):return None
    def exec_module(self,module):
        module.__file__=str(self.path);exec(compile(self.data,str(self.path),'exec'),module.__dict__)
class FrozenImports(importlib.abc.MetaPathFinder):
    def __init__(self,root,blobs,lookup):self.root,self.blobs,self.lookup=root,blobs,lookup
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in FORBIDDEN or fullname in HISTORICAL or fullname.split('.')[0] in ('jaxite_word','before87','checkout','reference_sources'):raise ImportError('forbidden preparation module')
        if fullname in self.lookup:
            key=self.lookup[fullname];origin=self.root/key
            return importlib.util.spec_from_loader(fullname,FrozenLoader(origin,self.blobs[key]),origin=str(origin))
        return None
def trusted_paths():
    paths=sysconfig.get_paths();stdlib=Path(paths['stdlib']).resolve()
    sites={Path(paths[k]).resolve() for k in ('purelib','platlib')}
    sites.update(Path(p).resolve() for p in site.getsitepackages())
    sites.update((stdlib/name).resolve() for name in ('site-packages','dist-packages'))
    sites.update(Path(p).resolve() for p in sys.path if p and Path(p).name in ('site-packages','dist-packages'))
    packages=[]
    for site_root in sites:
        for name in ('numpy','numpy.libs','pytest','_pytest','pluggy','packaging','iniconfig','pygments','py.py','typing_extensions.py'):
            packages.append(site_root/name)
        for pattern in ('numpy-*.dist-info','pytest-*.dist-info','pluggy-*.dist-info','packaging-*.dist-info','iniconfig-*.dist-info','Pygments-*.dist-info'):
            packages.extend(site_root.glob(pattern))
    return stdlib,sites,tuple(p.resolve() for p in packages)
class PublicAudit:
    def __init__(self,sources,out,owner):
        self.sources=set(sources);self.out=out;self.owner=owner;self.denied=[];self.pending_write=None
        self.stdlib,self.sites,self.packages=trusted_paths()
        self.os_files={Path(p) for p in ('/dev/null','/dev/urandom','/etc/localtime','/etc/os-release','/usr/lib/os-release','/proc/self/status','/proc/self/stat','/proc/self/maps','/proc/cpuinfo','/proc/meminfo')}
    def deny(self,kind,value):self.denied.append(dict(kind=kind,value=str(value)));raise PermissionError('public preparation boundary')
    def trusted(self,path):
        return (path.is_relative_to(self.stdlib) and not any(path.is_relative_to(p) for p in self.sites)) or any(path==p or path.is_relative_to(p) for p in self.packages)
    def __call__(self,event,args):
        if event.startswith(('subprocess.','os.fork','os.posix_spawn','os.spawn','os.exec')) or event=='os.system':self.deny('process_creation',event)
        if event.startswith('socket.'):self.deny('network',event)
        if event=='import' and args and (str(args[0]).split('.')[0] in FORBIDDEN or args[0] in HISTORICAL):self.deny('forbidden_import',args[0])
        if event=='ctypes.dlopen':
            value=args[0] if args else None
            if value is not None and (not isinstance(value,(str,os.PathLike)) or not self.trusted(Path(value).resolve())):self.deny('native_library',value)
        if event!='open' or not args:return
        if type(args[0])is int:self.deny('untracked_fd_open',args[0])
        if not isinstance(args[0],(str,bytes,os.PathLike)):self.deny('unknown_open',repr(args[0]))
        name=os.fsdecode(args[0]);path=Path(os.path.abspath(name))
        mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
        writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or (type(flags)is int and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)))
        if self.pending_write==name and writing:self.owner.check();path=self.out/name
        if any('libtpu' in p.lower() for p in path.parts):self.deny('tpu_library',path)
        if writing:
            self.owner.check()
            if not path.is_relative_to(self.out) or not path.resolve().is_relative_to(self.out.resolve()):self.deny('write_outside_owned_output',path)
            return
        # Exact authenticated source files take precedence over historical
        # filenames; arbitrary JSON/text, private roots and device files do not.
        if path in self.sources:
            if path.resolve()!=path:self.deny('authenticated_source_alias',path)
            return
        resolved=path.resolve()
        if path.is_relative_to(self.out) and resolved.is_relative_to(self.out.resolve()):self.owner.check();return
        if path.suffix.lower() in ('.npy','.npz','.safetensors','.parquet') or path.parts[:2]==('/','dev') and path not in self.os_files:self.deny('body_or_device',path)
        if path in self.os_files and resolved in {p.resolve() for p in self.os_files}:return
        if self.trusted(path) and self.trusted(resolved):return
        self.deny('unadmitted_read',path)
def loaded_local_origins(root,lookup,expected):
    origins={}
    for name,module in tuple(sys.modules.items()):
        if name.split('.')[0] in FORBIDDEN or name in HISTORICAL:raise ValueError('forbidden module loaded')
        if name in SYNTHETIC:
            if getattr(module,'__file__',None)is not None or getattr(getattr(module,'__spec__',None),'origin',None)is not None:raise ValueError('synthetic seam became a real module')
            continue
        for value in (getattr(module,'__file__',None),getattr(getattr(module,'__spec__',None),'origin',None)):
            if not isinstance(value,(str,os.PathLike)) or value in ('built-in','frozen'):continue
            path=Path(value).resolve()
            if path.is_relative_to(root):
                relative=str(path.relative_to(root));key='prepare_expanded.py' if name=='__main__' else lookup.get(name)
                if relative!=key or relative not in expected or digest(bounded(path))!=expected[relative]:raise ValueError('loaded local module origin differs')
                origins[name]=dict(path=relative,sha256=expected[relative])
    return origins
def terminal_status(action):
    try:
        status=action()
        if type(status)is not int or status not in (0,1):raise TypeError('exact normal preparation return required')
        return status
    except BaseException as error:
        try:print(type(error).__name__,file=sys.stderr,flush=True)
        except BaseException:pass
        return INFRASTRUCTURE_EXIT
def run(args):
    if not sys.flags.isolated or not sys.dont_write_bytecode or not __debug__:raise RuntimeError('Python -I -B with assertions required')
    root=Path(__file__).resolve().parent;original=Path(args.original_root).resolve()
    raw=bounded(args.source_manifest,1024**2)
    if digest(raw)!=args.manifest_sha256:raise ValueError('external source manifest differs')
    expected=parse(raw)
    if type(expected)is not dict or not expected or len(expected)>256:raise ValueError('bounded source inventory required')
    for name,pin in expected.items():
        if type(name)is not str or '\\' in name or any(p in ('','.','..') for p in name.split('/')) or type(pin)is not str or len(pin)!=64 or any(x not in '0123456789abcdef' for x in pin):raise ValueError('canonical source map required')
    if not {'prepare_expanded.py','freeze_io.py','cleanup_origin.py'}<=expected.keys():raise ValueError('complete preparation entry closure required')
    lookup=local_modules(expected);preloaded(root,original,lookup)
    execution_before,blobs=source_map(root,expected);original_before,_=source_map(original,expected)
    if sorted(os.sched_getaffinity(0))!=[112,113,114,115]:raise ValueError('fixed CPU lane required')
    out=output_path(args.output,(root,original));sys.meta_path.insert(0,FrozenImports(root,blobs,lookup))
    # Only these authenticated I/O helpers run before the read audit. They do
    # not read model/input bodies; Output exclusively acquires a fresh directory.
    from freeze_io import Output
    from cleanup_origin import protect_body,cleanup_origin,choose
    owner=Output(out);audit=None;primary=None
    report=dict(schema='qwen-expanded-full-owner-preparation-v2',passed=False,scientific_calibration_verified=False,actual_execution_authorized=False,
        learned_wholegraph_executed=False,synthetic_tests_started=False,synthetic_component_tests_completed=False,encrypted_execution=False,outer_receipt_required=True,
        pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp(),cpu_affinity=sorted(os.sched_getaffinity(0)),source_manifest_sha256=args.manifest_sha256,
        execution_root=str(root),original_root=str(original),execution_sources_before=execution_before,original_sources_before=original_before,
        scope='Public certificates/contracts plus explicit synthetic component stand-ins. Python audit, not OS isolation; outer wait/group receipt is mandatory.')
    start=time.monotonic()
    def publish(name,value):
        owner.check()
        if audit is not None:audit.pending_write=name
        try:return owner.json(name,value)
        finally:
            if audit is not None:audit.pending_write=None
    try:
        (out/'tmp').mkdir();os.environ['TMPDIR']=str(out/'tmp');os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
        os.environ['JAX_PLATFORMS']='cpu';os.environ['CUDA_VISIBLE_DEVICES']=''
        import tempfile
        tempfile.tempdir=str(out/'tmp')
        audit=PublicAudit({root/n for n in expected}|{original/n for n in expected},out,owner);sys.addaudithook(audit)
        sys.path[:0]=[str(root),str(root/'checkout/cadathlon/common')]
        import derive_expanded
        report['fresh_derivation']=derive_expanded.verify_generated()
        import pytest
        report['synthetic_tests_started']=True
        code=pytest.main(['-q','-s','-p','no:cacheprovider','--noconftest','--confcutdir='+str(root),'-c','/dev/null','-o','log_file='+str(out/'pytest.log'),*[str(root/n) for n in TESTS],'--rootdir='+str(root),'--junitxml='+str(out/'tests.xml')])
        report['test_exit_code']=int(code);rows=list(ET.parse(out/'tests.xml').getroot().iter('testcase'))
        report['tests']=dict(tests=len(rows),failures=sum(x.find('failure')is not None for x in rows),errors=sum(x.find('error')is not None for x in rows),skipped=sum(x.find('skipped')is not None for x in rows))
        if int(code)!=0 or report['tests']!=dict(tests=EXPECTED_TESTS,failures=0,errors=0,skipped=0):raise ValueError('all180 selected checks required')
        report['synthetic_component_tests_completed']=True
        import plan_full
        plan=plan_full.derive();report['public_plan_sha256']=publish('public_plan.json',plan)['sha256'];report['actions_completed']=True
    except BaseException as error:primary=protect_body(error);report['error_type']=type(error).__name__
    finally:
        body=primary if primary is not None and not cleanup_origin(primary) else None
        for label,action in (('execution_sources_after',lambda:source_map(root,expected)[0]),('original_sources_after',lambda:source_map(original,expected)[0]),('loaded_local_modules',lambda:loaded_local_origins(root,lookup,expected)),('output_identity',lambda:(owner.check(),list(owner.owner))[1])):
            try:report[label]=action()
            except BaseException as error:report[label+'_error']=type(error).__name__;primary=choose(primary,error,label,body_primary=body)
        report['denied_attempts']=audit.denied if audit else [];report['elapsed_s']=time.monotonic()-start
        report['passed']=primary is None and report.get('actions_completed')is True and not report['denied_attempts']
        try:publish('report.json',report)
        except BaseException as error:primary=choose(primary,error,'publication',body_primary=body)
        try:owner.check()
        except BaseException as error:primary=choose(primary,error,'output identity before close',body_primary=body)
        try:owner.close(primary)
        except BaseException as error:primary=choose(primary,error,'output close',body_primary=body)
    if primary is not None:raise primary
    print(json.dumps(dict(passed=report['passed'],tests=report.get('tests'),sources=len(expected))))
    return 0 if report['passed'] else 1
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',required=True);p.add_argument('--manifest-sha256',required=True);p.add_argument('--original-root',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();return terminal_status(lambda:run(args))
if __name__=='__main__':raise SystemExit(main())
