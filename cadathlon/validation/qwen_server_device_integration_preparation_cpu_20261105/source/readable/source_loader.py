"""Test-only explicit source finder, never the actual FrozenImports issuer."""
import ast
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

FORBIDDEN=frozenset(('numpy','jax','jaxlib','jaxite_word','libtpu','jax_plugins','torch','tensorflow','transformers','tokenizers','safetensors','flax','torch_xla'))

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def verify_tree(root,expected):
    actual={}
    for path in root.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():actual[path.relative_to(root).as_posix()]=sha(path.read_bytes())
        else:assert path.is_dir()
    assert actual==expected

class Loader(importlib.abc.Loader):
    def __init__(self,owner,name,relative):self.owner,self.name,self.relative=owner,name,relative
    def create_module(self,spec):return None
    def get_filename(self,name):
        assert name==self.name;return str(self.owner.root/self.relative)
    def exec_module(self,module):
        path=self.owner.root/self.relative;raw=path.read_bytes()
        assert sha(raw)==self.owner.full_map[self.relative]
        assert module.__file__==str(path)and module.__spec__.origin==str(path)and module.__loader__ is self
        self.owner.loaded[self.name]=(module,self,path)
        exec(compile(raw,str(path),'exec'),module.__dict__)

class SourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self,root,full_map,selection):
        self.root,self.full_map,self.selection=root,full_map,selection
        self.loaded={};self.denied=[];self.temporary={}
        self.local_stems={Path(n).stem for n in full_map if n.endswith('.py')}
    def find_spec(self,name,path=None,target=None):
        first=name.split('.')[0]
        if first in FORBIDDEN:
            self.denied.append(['discovery',name]);raise ImportError('genuine numerical/backend discovery forbidden')
        if name in self.selection:
            if target is not None or name in self.loaded:raise ImportError('selected test source reload forbidden')
            loader=Loader(self,name,self.selection[name])
            return importlib.util.spec_from_file_location(name,loader.get_filename(name),loader=loader)
        if first in self.local_stems or first in('runtime','historical','accepted_plan')or first.startswith('encrypted_'):
            self.denied.append(['unselected_source',name]);raise ImportError('outside the fixed test helper selection')
        return None
    def check(self):
        assert not self.denied
        assert set(self.loaded)<=set(self.selection)
        for name,(module,loader,path)in self.loaded.items():
            assert sys.modules.get(name)is module
            assert module.__file__==str(path)and module.__spec__.origin==str(path)and module.__loader__ is loader
            assert module.__spec__.loader is loader and path.resolve()==path
            assert sha(path.read_bytes())==self.full_map[self.selection[name]]
        found={n:m for n,m in sys.modules.items()if n.split('.')[0]in FORBIDDEN}
        assert found==self.temporary

def entry_definitions(runtime,provenance):
    path=runtime/'qwen_server_entry.py';raw=path.read_bytes();tree=ast.parse(raw)
    dump=lambda node:ast.dump(node,include_attributes=False)
    terminal=ast.parse("if __name__=='__main__':\n    raise SystemExit(main())\n").body[0]
    assert [i for i,n in enumerate(tree.body)if dump(n)==dump(terminal)]==[len(tree.body)-1]
    definitions=ast.Module(body=tree.body[:-1],type_ignores=tree.type_ignores)
    assert provenance['entry_sha256']==sha(raw)
    assert provenance['excluded_dispatch_ast_sha256']==sha(dump(tree.body[-1]).encode())
    assert provenance['definition_ast_sha256']==sha(dump(definitions).encode())
    entry=ModuleType('__main__');entry.__file__=str(path);entry.__spec__=None;entry.__package__=None;entry.__loader__=None
    return entry,compile(definitions,str(path),'exec')
