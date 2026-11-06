"""One bounded copied-source synthetic regression; no backend or body access."""
from pathlib import Path
import hashlib, importlib.abc, json, os, sys
P=Path(__file__).resolve().parent
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['TMPDIR']=str(P/'tmp')
(P/'tmp').mkdir(exist_ok=True)
import tempfile
tempfile.tempdir=str(P/'tmp')
FORBIDDEN={'jax','jaxlib','torch','transformers','tensorflow','safetensors','tokenizers','libtpu'}
hits=[]
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:
            hits.append(fullname);raise ImportError('no backend in synthetic regression')
def audit(event,args):
    if event.startswith(('socket.','subprocess.','os.fork','os.posix_spawn')):
        hits.append(event);raise PermissionError('no network/child process')
    if event=='open'and args and isinstance(args[0],(str,bytes,os.PathLike)):
        p=Path(os.fsdecode(args[0]))
        if p.suffix.lower()in('.npy','.npz','.safetensors','.parquet')or any(n in str(p)for n in('validation_inputs','calibration_inputs','client_gamma_fixtures','candidate-checker-private')):
            hits.append(str(p));raise PermissionError('no model/input/private body')
sys.meta_path.insert(0,Deny());sys.addaudithook(audit)
import pytest
code=pytest.main(['-q','-p','no:cacheprovider','--noconftest','-c','/dev/null',str(P/'test_wrapper_callback.py'),'--junitxml='+str(P/'tests.xml')])
(P/'audit.json').write_text(json.dumps(dict(denied_attempts=hits,backend_modules=sorted(n for n in sys.modules if n.split('.')[0]in FORBIDDEN),pid=os.getpid(),pgid=os.getpgrp()),indent=2)+'\n')
raise SystemExit(int(code))
