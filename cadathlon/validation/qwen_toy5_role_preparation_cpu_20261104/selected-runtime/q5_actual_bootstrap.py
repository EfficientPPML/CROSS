"""Closed source/import preparation. No actual Q5 role can be enabled here."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,importlib.abc,importlib.util,json,os,stat,sys

CORE_PREFIX='runtime/jaxite_word/'
BACKENDS=frozenset(('numpy','jax','jaxlib','jax_plugins','libtpu','scipy','ml_dtypes','opt_einsum'))

def _canonical(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def _hash(raw):return hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class SourceSnapshot:
    root:str
    map_bytes:bytes
    expected_sha256:str
    blobs:tuple
    def check(self):
        if type(self)is not SourceSnapshot or type(self.root)is not str or type(self.blobs)is not tuple or any(type(row)is not tuple or len(row)!=2 for row in self.blobs)or type(self.map_bytes)is not bytes or _hash(self.map_bytes)!=self.expected_sha256:
            raise ValueError('independently expected exact source map required')
        table=json.loads(self.map_bytes)
        if type(table)is not dict or _canonical(table)!=self.map_bytes or tuple(sorted(table))!=tuple(n for n,_ in self.blobs):raise ValueError('closed canonical source names required')
        root=Path(self.root)
        if str(root)!=self.root or not root.is_absolute()or root.resolve()!=root:raise ValueError('original resolved source root required')
        found={}
        for directory,dirs,files in os.walk(root,followlinks=False):
            for name in dirs:
                if (Path(directory)/name).is_symlink():raise ValueError('source directory alias forbidden')
            for name in files:
                path=Path(directory)/name;relative=path.relative_to(root).as_posix()
                if path.is_symlink()or not stat.S_ISREG(path.lstat().st_mode):raise ValueError('ordinary exact source files required')
                found[relative]=path
        if set(found)!=set(table):raise ValueError('source extras or missing files')
        for name,raw in self.blobs:
            if type(raw)is not bytes or _hash(raw)!=table[name]or found[name].read_bytes()!=raw:raise ValueError('original source bytes changed')
        return dict(self.blobs)

class _SourceLoader(importlib.abc.Loader):
    def __init__(self,gate,name,path,raw,package):self.gate=gate;self.name=name;self.path=path;self.raw=raw;self.package=package
    def create_module(self,spec):return None
    def exec_module(self,module):
        self.gate.check();module.__file__=self.path
        if self.package:module.__path__=[str(Path(self.path).parent)]
        self.gate.records[self.name]=(module,module.__spec__,self)
        exec(compile(self.raw,self.path,'exec'),module.__dict__)
        self.gate.check()

class FrozenImports(importlib.abc.MetaPathFinder):
    """Historical parent/reference-frozen copies can never shadow active code.

    Fixed core is delegated only to the exact shared core finder at index1.
    Genuine NumPy/backend imports remain denied in this preparation; no arbitrary
    installed-path policy or native device permission is inferred here.
    """
    def __init__(self,snapshot,selection):
        if type(snapshot)is not SourceSnapshot or type(selection)is not dict:raise TypeError('exact closed source selection required')
        blobs=snapshot.check();self.snapshot=snapshot;self.selection=tuple(sorted(selection.items()));self.blobs=blobs
        self.records={};self.core_enabled=False;self.closed=False
        names={n for n,_ in self.selection}
        if any(n in sys.modules for n in names):raise ImportError('selected helper was preloaded')
        if any(n.split('.')[0]in BACKENDS for n in sys.modules):raise ImportError('numeric/backend preloaded before source admission')
        for name,path in self.selection:
            if type(name)is not str or type(path)is not str or path not in blobs or path.startswith(('parent/','reference/frozen/'))or not path.endswith('.py'):raise ValueError('declared active module source required')
    def check(self):
        if self.closed or not sys.meta_path or sys.meta_path[0]is not self:raise ValueError('original source gate order differs')
        self.snapshot.check()
        for name,(module,spec,loader)in self.records.items():
            if sys.modules.get(name)is not module or module.__spec__ is not spec or spec.loader is not loader or module.__file__!=loader.path or spec.origin!=loader.path or module.__loader__ is not loader:
                raise ValueError('original active module binding changed')
            if loader.package and list(module.__path__)!=[str(Path(loader.path).parent)]:raise ValueError('original active package search path changed')
    def find_spec(self,fullname,path=None,target=None):
        root=fullname.split('.')[0]
        if root in BACKENDS:raise PermissionError('genuine numeric/backend policy is not activated in source preparation')
        if root=='jaxite_word' or any(fullname==Path(n).stem for n in self.blobs if n.startswith(CORE_PREFIX)and n.endswith('.py')):
            if not self.core_enabled:raise PermissionError('core import before complete runtime authority')
            return None
        selected=dict(self.selection)
        if fullname in selected:
            relative=selected[fullname];raw=self.blobs[relative];filename=str(Path(self.snapshot.root)/relative)
            loader=_SourceLoader(self,fullname,filename,raw,relative.endswith('/__init__.py'))
            return importlib.util.spec_from_loader(fullname,loader,origin=filename,is_package=loader.package)
        if root.startswith('q5_')or root in('parent','reference')or root.startswith('encrypted_'):
            raise ImportError('undeclared local or historical import forbidden')
        return None

def capture_source(root,map_bytes,expected_sha256):
    if not sys.flags.isolated or not sys.dont_write_bytecode or not __debug__:raise ValueError('isolated -I -B nonoptimized process required')
    if type(map_bytes)is not bytes or _hash(map_bytes)!=expected_sha256:raise ValueError('external source-map digest differs before parse')
    table=json.loads(map_bytes)
    if type(table)is not dict or _canonical(table)!=map_bytes:raise ValueError('canonical compact source map required')
    base=Path(root)
    if not base.is_absolute()or base.resolve()!=base:raise ValueError('resolved explicit execution source required')
    blobs=[]
    for name,pin in sorted(table.items()):
        if type(name)is not str or '\\'in name or name.startswith('/')or '..'in Path(name).parts or str(Path(name))!=name:raise ValueError('flat relative source path required')
        p=base/name
        if p.is_symlink()or p.resolve()!=p or not p.is_file():raise ValueError('source aliases forbidden')
        raw=p.read_bytes()
        if _hash(raw)!=pin:raise ValueError('source digest differs before helper imports')
        blobs.append((name,raw))
    result=SourceSnapshot(str(base),map_bytes,expected_sha256,tuple(blobs));result.check();return result
