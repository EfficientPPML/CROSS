"""Fresh owned child; explicit definition-only entry, not CLI authentication."""
import ast
import hashlib
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
from types import ModuleType

FORBIDDEN=frozenset(('numpy','jax','jaxlib','jaxite_word','libtpu','jax_plugins','torch','tensorflow','transformers','tokenizers','safetensors','flax','torch_xla'))
DENIED=[]
TEMPORARY={}
PROTECTED=()

def absent():
    found={n:m for n,m in sys.modules.items()if n.split('.')[0]in FORBIDDEN}
    assert found==TEMPORARY and not DENIED

class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:
            DENIED.append(('discovery',fullname));raise ImportError('genuine backend discovery excluded')
        return None

def audit(event,args):
    if event=='import'and args and str(args[0]).split('.')[0]in FORBIDDEN:
        DENIED.append(('import',str(args[0])));raise ImportError('genuine native import excluded')
    if event=='open'and args and type(args[0])in(str,bytes):
        name=os.fsdecode(args[0])
        if any(name==p or name.startswith(p+'/')for p in('/proc','/sys','/dev',*PROTECTED)):
            DENIED.append(('body_or_probe',name));raise PermissionError('probe/protected body unavailable in synthetic integration')

def main():
    global TEMPORARY,PROTECTED
    assert sys.flags.isolated and sys.dont_write_bytecode and __debug__
    scenario,root_arg,result_arg=sys.argv[1:]
    root=Path(root_arg).resolve();runtime=root/'runtime';result_path=Path(result_arg)
    expected=json.loads((root/'runtime_sources.json').read_bytes());assert len(expected)==391
    def sources():
        out={}
        for p in runtime.rglob('*'):
            assert not p.is_symlink()
            if p.is_file():out[p.relative_to(runtime).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        assert out==expected
    absent();sources();finder=DenyBackend();sys.meta_path.insert(0,finder);sys.addaudithook(audit)
    sys.path[:0]=[str(runtime),str(root)]
    from integration_cases import CASES
    assert scenario in CASES
    raw=(runtime/'qwen_server_entry.py').read_bytes();tree=ast.parse(raw)
    terminal=ast.parse("if __name__=='__main__':\n    raise SystemExit(main())\n").body[0]
    dump=lambda node:ast.dump(node,include_attributes=False)
    matches=[i for i,node in enumerate(tree.body)if dump(node)==dump(terminal)]
    assert matches==[len(tree.body)-1]
    node_hash=hashlib.sha256(dump(tree.body[-1]).encode()).hexdigest()
    definitions=ast.Module(body=tree.body[:-1],type_ignores=tree.type_ignores)
    provenance=json.loads((root/'entry_fixture_provenance.json').read_bytes())
    assert provenance['entry_sha256']==hashlib.sha256(raw).hexdigest()
    assert provenance['excluded_dispatch_ast_sha256']==node_hash
    assert provenance['definition_ast_sha256']==hashlib.sha256(dump(definitions).encode()).hexdigest()
    original_main=sys.modules['__main__'];entry=ModuleType('__main__')
    entry.__file__=str(runtime/'qwen_server_entry.py');entry.__spec__=None;entry.__package__=None
    fixture=None;result=None
    try:
        sys.modules['__main__']=entry
        exec(compile(definitions,entry.__file__,'exec'),entry.__dict__)
        assert 'qwen_server_entry'not in sys.modules
        assert not any(name in sys.modules for name in entry.ROOT_MODULES if name!='qwen_server_entry')
        PROTECTED=entry.EARLY_PROTECTED
        blobs=entry.snapshot(expected);entry.origins(expected)
        if scenario.startswith('terminal_'):
            from integration_checks import terminal
            detail=terminal(entry,scenario);seams=['definition-only entry; exact terminal_status function']
            counts={};events=[]
        else:
            from integration_fixture import Fixture
            from integration_checks import run_direct,run_process
            fixture=Fixture(runtime,entry,expected,blobs,result_path.parent,
                mode='process'if scenario.startswith('process_')else'direct',scenario=scenario)
            TEMPORARY=fixture.temporary
            if scenario.startswith('process_'):run_process(fixture,scenario)
            else:run_direct(fixture,scenario)
            counts=dict(fixture.counts);events=list(fixture.events);seams=list(fixture.seams)
            detail=dict(function_boundary_only=True,genuine_bootstrap=False)
            absent();entry.origins(expected,core_enabled=fixture.gate.core_enabled)
        result=dict(scenario=scenario,assertions_completed=True,pid=os.getpid(),pgid=os.getpgrp(),
            source_count=391,entry_dispatch_excluded_sha256=node_hash,definition_only_entry=True,
            genuine_cli_bootstrap=False,recording_seams=seams,counters=counts,events=events,detail=detail,
            genuine_backend_imported=False,scientific_admission=False,capacity_measurement=False,
            model_key_ciphertext_body_reads=False,actual_server_execution=False,backend_denials=[])
    finally:
        try:
            if fixture is not None:fixture.close()
        finally:
            sys.modules['__main__']=original_main;TEMPORARY={}
    absent();sources();assert sys.modules['__main__']is original_main and sys.getprofile()is None
    assert result is not None
    result.update(original_main_restored=True,temporary_names_removed=True,profile_restored=True,
        declared_exports_restored=True,runtime_sources_unchanged=True)
    with result_path.open('x')as stream:json.dump(result,stream,sort_keys=True);stream.write('\n')

if __name__=='__main__':main()
