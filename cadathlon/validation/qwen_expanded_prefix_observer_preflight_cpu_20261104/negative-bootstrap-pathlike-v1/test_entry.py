"""Fresh isolated source-bootstrap probes with inert synthetic helper bodies."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,types
import pytest

ENTRY=Path(__file__).parent/'entry.py'
REQUIRED=('freeze_io.py','observer_process.py','input_gate.py','cleanup_origin.py','model_runtime.py',
    'native_observer.py','ranges.py','workload.py','reference.py','arithmetic.py',
    'execution_authorization.json','external_inputs.json','public/protocol.json',
    'public/model_pins.json','public/candidate3_manifest.json','backend_sources.json')


def prepared(tmp_path):
    root=tmp_path/'source';root.mkdir();marker=tmp_path/'marker'
    (root/'entry.py').write_bytes(ENTRY.read_bytes())
    for name in REQUIRED:
        path=root/name;path.parent.mkdir(exist_ok=True)
        path.write_text('{}' if path.suffix=='.json' else '# inert synthetic helper\n')
    (root/'observer_process.py').write_text('from pathlib import Path\nPath('+repr(str(marker))+').write_text("AFTER_SOURCE_AUTH")\ndef dispatch(*args): return 0\n')
    manifest=tmp_path/'manifest.json'
    names=sorted(p for p in root.rglob('*') if p.is_file())
    manifest.write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in names},sort_keys=True))
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    args=[str(root/'entry.py'),'--manifest',str(manifest),'--manifest-sha256',digest,
        '--mode','test-worker','--output',str(tmp_path/'out')]
    return root,marker,manifest,args


def launch(args,wrapper=None):
    command=[sys.executable,'-I','-B',*( ['-c',wrapper] if wrapper else args)]
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    output,error=child.communicate()
    assert child.returncode is not None
    with pytest.raises(ProcessLookupError):os.killpg(child.pid,0)
    return child.returncode,output,error


def test_canonical_bootstrap_control(tmp_path):
    root,marker,manifest,args=prepared(tmp_path)
    code,_,_=launch(args)
    assert code==0 and marker.read_text()=='AFTER_SOURCE_AUTH'


@pytest.mark.parametrize('origin',['file','spec'])
@pytest.mark.parametrize('symlink',[False,True])
def test_pathlike_preloaded_alias_rejected_before_helper(tmp_path,origin,symlink):
    root,marker,manifest,args=prepared(tmp_path)
    target=root/'input_gate.py'
    if symlink:
        link=tmp_path/'alias_link';link.symlink_to(target);target=link
    setup=('module.__file__=value' if origin=='file' else
           'module.__spec__=types.SimpleNamespace(origin=value)')
    wrapper=('import pathlib,types,sys,runpy\nmodule=types.ModuleType("unknown_alias")\n'
        'value=pathlib.Path('+repr(str(target))+')\n'+setup+'\nsys.modules["unknown_alias"]=module\n'
        'sys.argv='+repr(args)+'\nrunpy.run_path('+repr(args[0])+',run_name="__main__")\n')
    code,_,error=launch(args,wrapper)
    assert code!=0 and not marker.exists(),error.decode()


@pytest.mark.parametrize('mutation',['changed','missing','extra_py','extra_bytecode','extra_native','symlink','digest'])
def test_bad_closure_never_executes_helper(tmp_path,mutation):
    root,marker,manifest,args=prepared(tmp_path)
    if mutation=='changed':(root/'input_gate.py').write_text('bad')
    elif mutation=='missing':(root/'input_gate.py').unlink()
    elif mutation=='extra_py':(root/'platform.py').write_text('bad')
    elif mutation=='extra_bytecode':(root/'module.pyc').write_bytes(b'bad')
    elif mutation=='extra_native':(root/'module.so').write_bytes(b'bad')
    elif mutation=='symlink':(root/'link.py').symlink_to(root/'input_gate.py')
    elif mutation=='digest':args[4]='0'*64
    code,_,_=launch(args)
    assert code!=0 and not marker.exists()
