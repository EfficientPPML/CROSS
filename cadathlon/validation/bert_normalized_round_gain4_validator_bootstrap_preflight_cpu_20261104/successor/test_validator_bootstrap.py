"""Real fresh-interpreter rejection; no key/context/private/payload reads."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys
import pytest
HERE=Path(__file__).resolve().parent

def setup(tmp_path):
    root=tmp_path/'execution';root.mkdir()
    names=json.loads((HERE/'source_inventory.json').read_bytes())
    for name in names:
        target=root/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/name,target)
    expected={n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in names}
    manifest=tmp_path/'expected.json';manifest.write_text(json.dumps(expected,sort_keys=True)+'\n')
    return root,manifest,expected

def invoke(root,manifest,*,entry='validator_worker.py',pin=None,preload=None):
    pin=hashlib.sha256(manifest.read_bytes()).hexdigest()if pin is None else pin
    args=[str(root/entry),'--source-manifest',str(manifest),'--source-manifest-sha256',pin,'--execute']
    if entry=='validator_worker.py':args+=['--config','/synthetic/absent-config','--config-sha256','00'*32,'--output',str(root.parent/'worker-report.json')]
    else:args+=['--output',str(root.parent/'collector-output'),'--key-evidence','/synthetic/absent-keys','--key-workspace','/synthetic/absent-store']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for n in ['PYTHONPATH','TPU_VISIBLE_CHIPS','XLA_FLAGS']:env.pop(n,None)
    if preload:
        code="import runpy,sys,types; from pathlib import Path; root=Path(sys.argv[1]); mode=sys.argv[2]; module=types.ModuleType('synthetic'); module.__file__=str(root/'validator_common.py') if mode!='wrong-origin' else '/foreign/validator_common.py'; sys.modules['alias_helper' if mode=='alias' else 'validator_source_guard' if mode=='guard' else 'validator_common']=module; sys.argv=sys.argv[3:]; runpy.run_path(sys.argv[0],run_name='__main__')"
        command=[sys.executable,'-I','-B','-c',code,str(root),preload,*args]
    else:command=[sys.executable,'-I','-B',*args]
    return subprocess.run(command,env=env,capture_output=True,text=True,check=False)

@pytest.mark.parametrize('entry',['validator_worker.py','collect_validator.py'])
def test_original_root_canary_rejected_before_helper_import(tmp_path,entry):
    root,manifest,_=setup(tmp_path);canary=tmp_path/'new-canary'
    helper=root/'validator_common.py';helper.write_text('from pathlib import Path\nPath('+repr(str(canary))+').write_text("unexpected")\n'+helper.read_text())
    result=invoke(root,manifest,entry=entry)
    assert result.returncode!=0 and'source byte identity differs'in result.stderr
    assert not canary.exists()and not (tmp_path/'worker-report.json').exists()and not (tmp_path/'collector-output').exists()

@pytest.mark.parametrize('change',['missing','extra','digest','manifest-extra','manifest-bool','manifest-missing','symlink'])
def test_complete_manifest_before_helpers(tmp_path,change):
    root,manifest,expected=setup(tmp_path)
    if change=='missing':(root/'validator_common.py').unlink()
    elif change=='extra':(root/'untracked_helper.py').write_text('raise AssertionError("must not import")')
    elif change=='manifest-extra':expected['not_present.py']='00'*32
    elif change=='manifest-bool':expected['validator_common.py']=True
    elif change=='manifest-missing':expected.pop('validator_common.py')
    elif change=='symlink':
        helper=root/'validator_common.py';helper.rename(tmp_path/'foreign.py');helper.symlink_to(tmp_path/'foreign.py')
    if change.startswith('manifest-'):manifest.write_text(json.dumps(expected)+'\n')
    result=invoke(root,manifest,pin='00'*32 if change=='digest'else None)
    assert result.returncode!=0 and'disabled'not in result.stderr
    assert not (tmp_path/'worker-report.json').exists()

@pytest.mark.parametrize('preload',['known','wrong-origin','alias','guard'])
def test_preloaded_or_aliased_helpers_rejected(tmp_path,preload):
    root,manifest,_=setup(tmp_path);result=invoke(root,manifest,preload=preload)
    assert result.returncode!=0 and ('preloaded'in result.stderr or'wrong origin'in result.stderr)
    assert not (tmp_path/'worker-report.json').exists()

@pytest.mark.parametrize('entry',['validator_worker.py','collect_validator.py'])
def test_exact_fresh_source_reaches_disabled_gate_without_publication(tmp_path,entry):
    root,manifest,_=setup(tmp_path);result=invoke(root,manifest,entry=entry)
    assert result.returncode!=0 and'validator execution remains disabled'in result.stderr,result.stderr
    assert not (tmp_path/'worker-report.json').exists()and not (tmp_path/'collector-output').exists()
