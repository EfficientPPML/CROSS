"""Enclosing pytest worker: exact59 prefixture selection; no runtime imports here."""
import sys, json, os, hashlib, importlib.abc, time
from pathlib import Path

raw = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == sys.argv[2]
config = json.loads(raw)
out = Path(config['proof_root']); source = out / 'execution'
assert config['synthetic_only'] is True and config['actual_execution_authorized'] is False
assert sys.flags.isolated and sys.dont_write_bytecode and __debug__
assert sys.executable == config['python']
assert os.sched_getaffinity(0) == {112,113,114,115}
assert os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
assert os.environ.get('FULL28_FOCUSED_SOURCE_MAP')==str(out/'sources-before.json')
assert os.environ.get('FULL28_FOCUSED_SOURCE_MAP_SHA256')==config['source_manifest_sha256']
temporary = Path(config['owned_tmpdir'])
assert temporary == out/'tmp' and os.environ.get('TMPDIR') == str(temporary)
assert temporary.is_dir() and not temporary.is_symlink() and temporary.resolve() == temporary
assert temporary.stat().st_uid == os.geteuid() and not list(temporary.iterdir())
def dump(name, value):
    (out/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def ticks(): return int((Path('/proc')/str(os.getpid())/'stat').read_text().rsplit(')',1)[1].split()[19])
def hashes(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}
start_ticks=ticks(); runtime_before=hashes(source/'runtime')
assert len(runtime_before)==403 and runtime_before==config['runtime_source_expectations']
forbidden=set(config['worker_forbidden_roots']); denied=[]; collected=[]
def forbidden_name(name): return name.split('.')[0] in forbidden or name.split('.')[0].startswith('encrypted_')
assert not any(forbidden_name(n) for n in sys.modules)
def audit(event,args):
    if event=='import' and args and isinstance(args[0],str) and forbidden_name(args[0]):
        denied.append(args[0]); raise PermissionError('enclosing worker forbids numerical/backend imports')
class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if forbidden_name(fullname):
            denied.append(fullname); raise PermissionError('enclosing worker forbids backend discovery')
sys.addaudithook(audit); sys.meta_path.insert(0,DenyBackend()); sys.path.insert(0,str(source))
dump('worker-early.json',dict(pid=os.getpid(),pgid=os.getpgrp(),start_ticks=start_ticks,
    source_manifest_sha256=config['source_manifest_sha256'],config_sha256=sys.argv[2]))
status=None; primary=None; secondary=[]; loaded=[]; runtime_after=None
try:
    import tempfile
    assert Path(tempfile.gettempdir()).resolve()==temporary
    import pytest
    class ExactSelection:
        def pytest_collection_finish(self,session):
            collected.extend(item.nodeid for item in session.items)
            if collected!=config['selected_nodes']:
                raise pytest.UsageError('exact ordered59 required before fixtures')
    selected=config['selected_files']; assert selected==['test_device_integration.py']
    basetemp=Path(config['pytest_basetemp']); assert basetemp==out/'pytest-owned'
    assert not os.path.lexists(basetemp)
    status=int(pytest.main(['-q','-p','no:cacheprovider','-c','/dev/null',
        '--rootdir='+str(source),'--confcutdir='+str(source),'--import-mode=importlib',
        '--basetemp='+str(basetemp),*[str(source/n)for n in selected],
        '--junitxml='+str(out/'tests.xml')],plugins=[ExactSelection()]))
except BaseException as error:
    primary=error
finally:
    try:
        loaded=sorted(n for n in sys.modules if forbidden_name(n))
        runtime_after=hashes(source/'runtime')
        assert ticks()==start_ticks
        assert runtime_after==runtime_before
        assert not denied and not loaded
    except BaseException as error: secondary.append(error)
    code=70 if primary is not None or secondary else status
    try:
        dump('worker-audit.json',dict(pid=os.getpid(),pgid=os.getpgrp(),start_ticks=start_ticks,
            denied=denied,loaded_backend=loaded,forbidden_roots=sorted(forbidden),
            collected_nodes=collected,pytest_exit_code=status,planned_exit_code=code,
            runtime_sources_before=runtime_before,runtime_sources_after=runtime_after,
            runtime_module_origins={},
            numpy_imported=any(n=='numpy'or n.startswith('numpy.')for n in sys.modules),
            affinity=sorted(os.sched_getaffinity(0)),temporary_root=str(temporary),
            synthetic_only=True,actual_role=False,backend_context=False,
            primary_type=None if primary is None else type(primary).__name__,
            cleanup_types=[type(e).__name__ for e in secondary]))
        dump('worker-terminal.json',dict(pid=os.getpid(),pgid=os.getpgrp(),start_ticks=start_ticks,
            pytest_exit_code=status,planned_exit_code=code,normal_main_return=primary is None and not secondary,
            actual_exit_and_reap_authority='original enclosing collector wait, not this pre-exit receipt'))
    except BaseException as error: secondary.append(error)
    chosen=primary if primary is not None else(secondary[0]if secondary else None)
    if chosen is not None:
        for error in secondary:
            if error is not chosen:
                try: chosen.add_note('worker finalization: '+type(error).__name__)
                except BaseException: pass
        raise chosen
raise SystemExit(code)
