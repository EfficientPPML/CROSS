"""Stdlib-only admission before any validator helper, gate, runtime or report.

The caller supplies an external manifest digest. This does not authenticate its
own caller or erase code already executed before entering a dedicated process.
"""
from pathlib import Path
import ast,hashlib,importlib.machinery,json,os,stat,sys
HERE=Path(__file__).resolve().parent
_UNCERTAIN=[]

def read(path,maximum=8<<20):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK);primary=None
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode)or not 0<before.st_size<=maximum:raise ValueError('bounded regular source file required')
        parts=[];size=0
        while True:
            block=os.read(fd,min(1<<20,maximum+1-size))
            if not block:break
            parts.append(block);size+=len(block)
            if size>maximum:raise ValueError('source bound exceeded')
        after=os.fstat(fd)
        state=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if state(before)!=state(after)or size!=before.st_size:raise ValueError('source changed during read')
        return b''.join(parts)
    except BaseException as error:primary=error;raise
    finally:
        try:os.close(fd)
        except BaseException as error:
            _UNCERTAIN.append((fd,error))
            if primary is None:raise
            primary.add_note('source descriptor close: '+repr(error))

def parse(raw):
    def pairs(items):
        out={}
        for name,value in items:
            if name in out:raise ValueError('duplicate source metadata key')
            out[name]=value
        return out
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in()).throw(ValueError('nonfinite source metadata')))

def relative(name):
    if type(name)is not str or not name or Path(name).is_absolute()or '..'in Path(name).parts or Path(name).as_posix()!=name:raise ValueError('canonical relative source member required')
    return name

def digest(raw):return hashlib.sha256(raw).hexdigest()

def expected_manifest(path,pin):
    if type(pin)is not str or len(pin)!=64 or any(c not in '0123456789abcdef'for c in pin):raise ValueError('external manifest SHA256 required')
    raw=read(path,1<<20)
    if digest(raw)!=pin:raise ValueError('external source manifest digest differs')
    expected=parse(raw)
    if type(expected)is not dict or not 163<len(expected)<=1024:raise ValueError('complete typed consumer source inventory required')
    for name,h in expected.items():
        relative(name)
        if type(h)is not str or len(h)!=64 or any(c not in '0123456789abcdef'for c in h):raise ValueError('typed source digest required')
    return expected

def resolve(name,roots):
    search=[str(p)for p in roots];spec=None
    for i,part in enumerate(name.split('.')):
        spec=importlib.machinery.PathFinder.find_spec(part,search)
        if spec is None:return None
        if i<len(name.split('.'))-1:
            if spec.submodule_search_locations is None:return None
            search=list(spec.submodule_search_locations)
    if spec.origin in (None,'built-in','frozen'):return None
    return Path(spec.origin).resolve()

def check(expected,root=HERE,*,fresh=True,modules=None,entry=None):
    if type(fresh)is not bool:raise ValueError('typed freshness required')
    root=Path(root).resolve();raw={}
    for name,h in expected.items():
        path=root/relative(name)
        for part in (path,*path.parents):
            if part==root:break
            if part.is_symlink():raise ValueError('source symlink forbidden')
        raw[name]=read(path)
        if digest(raw[name])!=h:raise ValueError('source byte identity differs: '+name)
    if sum(map(len,raw.values()))>16<<20:raise ValueError('consumer source total exceeds16MiB')
    inventory=parse(raw['source_inventory.json'])
    if type(inventory)is not list or any(type(n)is not str for n in inventory)or len(set(inventory))!=len(inventory)or set(inventory)!=set(expected):raise ValueError('exact complete declared consumer inventory required')
    actual=set()
    for path in root.rglob('*'):
        if '__pycache__'in path.parts or '.pytest_cache'in path.parts:continue
        if path.is_symlink():raise ValueError('uncaptured source symlink')
        if path.is_file():actual.add(path.relative_to(root).as_posix())
    if actual!=set(expected):raise ValueError('extra or missing source-tree member')
    roots=(root,root/'runtime/stage',root/'common',root/'runtime',root/'runtime/jaxite_word')
    critical={}
    # Match the actual ordered import search path, including flat/package aliases.
    for directory in reversed(roots):
        for path in directory.glob('*.py'):
            if path.name!='__init__.py':critical[path.stem]=path.relative_to(root).as_posix()
    for path in (root/'runtime/jaxite_word').glob('*.py'):
        name='jaxite_word'if path.stem=='__init__'else'jaxite_word.'+path.stem
        critical[name]=path.relative_to(root).as_posix()
    resolved={};pending=list(critical);seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name);path=resolve(name,roots)
        if path is None or not path.is_relative_to(root):continue
        relative_path=path.relative_to(root).as_posix()
        if relative_path not in expected:raise ValueError('reachable uncaptured local import: '+name)
        resolved[name]=relative_path
        for node in ast.walk(ast.parse(raw[relative_path])):
            if isinstance(node,ast.Import):pending.extend(a.name for a in node.names)
            elif isinstance(node,ast.ImportFrom)and node.level==0 and node.module:
                pending.append(node.module)
                pending.extend(node.module+'.'+a.name for a in node.names if a.name!='*')
    for name,path in critical.items():
        if resolved.get(name)!=path:raise ValueError('static import origin differs: '+name)
    loaded={};modules=sys.modules if modules is None else modules
    for name,module in tuple(modules.items()):
        if module is None:continue
        file=getattr(module,'__file__',None)
        local=file is not None and Path(file).is_absolute()and Path(file).resolve().is_relative_to(root)
        if name in resolved or local:
            if not file:raise ValueError('preloaded helper without origin')
            path=Path(file).resolve();relative_path=path.relative_to(root).as_posix()if path.is_relative_to(root)else None
            if name in resolved and relative_path!=resolved[name]:raise ValueError('preloaded helper wrong origin: '+name)
            if relative_path not in expected:raise ValueError('loaded uncaptured source')
            allowed=(name=='validator_source_guard'and relative_path=='validator_source_guard.py')or(name=='__main__'and entry is not None and path==Path(entry).resolve())
            if fresh and not allowed:raise ValueError('helper/runtime preloaded or aliased before source gate: '+name)
            origin=getattr(getattr(module,'__spec__',None),'origin',None)
            if origin not in (None,'built-in','frozen')and Path(origin).resolve()!=path:raise ValueError('loaded spec origin differs')
            loaded[name]=relative_path
    for name in ('polynomial','ckks_ctx','mapping'):
        if name in modules and'jaxite_word.'+name in modules and modules[name]is not modules['jaxite_word.'+name]:raise ValueError('split core module identity')
    return dict(passed=True,fresh=fresh,source_count=len(expected),source_bytes=sum(map(len,raw.values())),static_imports=resolved,loaded=loaded)

def bootstrap(manifest,pin,*,root=HERE,entry=None):
    expected=expected_manifest(manifest,pin)
    report=check(expected,root,fresh=True,entry=entry);report['manifest_sha256']=pin
    return expected,report
