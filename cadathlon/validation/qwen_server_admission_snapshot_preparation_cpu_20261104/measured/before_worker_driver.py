"""Unrun snapshot recording-state worker; exact prefixture selection only."""
import sys,json,os,hashlib,importlib.abc
from pathlib import Path

raw=Path(sys.argv[1]).read_bytes();assert hashlib.sha256(raw).hexdigest()==sys.argv[2]
config=json.loads(raw)
assert config['synthetic_only']is True and config['actual_execution_authorized']is False
out=Path(config['proof_root']);source=out/'execution'
assert sys.flags.isolated and sys.dont_write_bytecode and __debug__
assert os.sched_getaffinity(0)=={112,113,114,115}
assert os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD')=='1'
temporary=Path(config['owned_tmpdir']);assert temporary==out/'tmp'
assert os.environ.get('TMPDIR')==str(temporary)and temporary.is_dir()and not temporary.is_symlink()
assert temporary.resolve()==temporary and temporary.stat().st_uid==os.geteuid()
assert not any(temporary.iterdir())
assert config['runtime_source_expectations']=={}and config['runtime_module_origins']=={}
forbidden={'ckks_ctx','flax','jax','jax_plugins','jaxite_word','jaxlib','key_gen','libtpu','numpy','polynomial','safetensors','tensorflow','tokenizers','torch','torch_xla','transformers'}
assert sorted(forbidden)==config['forbidden_roots']
denied=[];collected=[]
assert not any(n.split('.')[0]in forbidden for n in sys.modules)
def audit(event,args):
    if event=='import'and args and str(args[0]).split('.')[0]in forbidden:
        denied.append(str(args[0]));raise PermissionError('snapshot proof forbids numeric/model backend imports')
class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in forbidden:
            denied.append(fullname);raise PermissionError('snapshot proof forbids numeric/model backend discovery')
sys.addaudithook(audit);sys.meta_path.insert(0,DenyBackend())
sys.path.insert(0,str(source))
import tempfile
assert Path(tempfile.gettempdir()).resolve()==temporary
import pytest
class ExactSelection:
    def pytest_collection_finish(self,session):
        collected.extend(item.nodeid for item in session.items)
        if collected!=config['selected_nodes']:raise pytest.UsageError('exact snapshot cases required before fixtures')
selected=config['selected_files'];assert selected==['test_snapshot_guard.py']
basetemp=Path(config['pytest_basetemp']);assert basetemp==out/'pytest-owned'
assert not os.path.lexists(basetemp)
assert hashlib.sha256((source/'pytest.ini').read_bytes()).hexdigest()==config['pytest_ini_sha256']
status=pytest.main(['-q','-p','no:cacheprovider','-c',str(source/'pytest.ini'),
    '--rootdir='+str(source),'--confcutdir='+str(source),'--import-mode=importlib',
    '--basetemp='+str(basetemp),*[str(source/n)for n in selected],
    '--junitxml='+str(out/'tests.xml')],plugins=[ExactSelection()])
loaded=[n for n in sys.modules if n.split('.')[0]in forbidden]
assert Path(tempfile.gettempdir()).resolve()==temporary
(out/'worker-audit.json').write_text(json.dumps(dict(denied=denied,loaded_backend=loaded,
    collected_nodes=collected,forbidden_roots=sorted(forbidden),
    numpy_imported=any(n=='numpy'or n.startswith('numpy.')for n in sys.modules),
    pytest_exit_code=int(status),affinity=sorted(os.sched_getaffinity(0)),pid=os.getpid(),pgid=os.getpgrp(),
    runtime_sources_before={},runtime_sources_after={},runtime_module_origins={},
    temporary_root=str(temporary),synthetic_only=True,scientific_admission_tested=False),sort_keys=True,indent=2)+'\n')
raise SystemExit(int(status)if not denied and not loaded else 70)
