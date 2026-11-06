"""Actual fresh full-source CLI rejection, with no actual grant or bodies."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
import pytest

HERE=Path(__file__).parent


@pytest.mark.parametrize('mode',['collect-heldout','heldout-worker','collect-row0','row0-worker'])
def test_actual_full_closure_no_grant_denies_before_body_or_output(tmp_path,mode):
    expected={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in HERE.rglob('*')if p.is_file()}
    manifest=tmp_path/'source-manifest.json';manifest.write_text(json.dumps(expected,sort_keys=True,indent=2)+'\n')
    output=tmp_path/'never-created';private=tmp_path/'private-never-created';marker=tmp_path/'forbidden-body-marker'
    args=[str(HERE/'heldout_entry.py'),'--manifest',str(manifest),'--manifest-sha256',hashlib.sha256(manifest.read_bytes()).hexdigest(),
          '--mode',mode,'--output',str(output),'--private-output',str(private)]
    wrapper='''import sys,os,pathlib,runpy
marker=pathlib.Path(MARKER)
def audit(event,args):
 if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
  value=os.path.abspath(os.fsdecode(args[0]))
  forbidden=value.startswith('/home/jianming_gatech/cross_validation_assets/')or value.startswith('/tmp/qwen-expanded-prefix-token-inputs-')or value.startswith('/tmp/qwen-expanded-candidate-checker-private-')or value.startswith('/tmp/qwen-expanded-candidate-checker-actual-')or value.endswith(('.safetensors','.parquet','.npy','.npz'))
  if forbidden:
   marker.write_text('FORBIDDEN_BODY_ATTEMPT')
   raise PermissionError('Actual body access in no-grant probe')
sys.addaudithook(audit)
sys.argv=ARGS
runpy.run_path(ENTRY,run_name='__main__')
'''.replace('MARKER',repr(str(marker))).replace('ARGS',repr(args)).replace('ENTRY',repr(args[0]))
    child=subprocess.Popen([sys.executable,'-I','-B','-c',wrapper],stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    stdout,stderr=child.communicate()
    assert child.returncode!=0 and b'Independent frozen-workflow grant required'in stderr,stderr[-4000:]
    assert not marker.exists()and not output.exists()and not private.exists()
    with pytest.raises(ProcessLookupError):os.killpg(child.pid,0)
