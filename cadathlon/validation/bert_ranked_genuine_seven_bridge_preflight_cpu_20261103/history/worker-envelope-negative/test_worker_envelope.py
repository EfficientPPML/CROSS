"""Execute the exact run AST with public no-I/O dependencies for all seven phases."""
import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import time
import pytest
SOURCE=Path(__file__).parent/'sources/bridge_worker.py'
PHASES=('client_key_generation','server_preprocess_model','client_preprocess_input','client_encode_encrypt_input','server_encrypted_compute','client_decrypt_decode','client_postprocess')
@contextmanager
def scope(*a,**kw):yield
@pytest.mark.parametrize('phase,stage',zip(PHASES,(1,2,3,3,4,5,5)))
def test_valid_nested_envelope_reaches_held_owner_and_phase(phase,stage):
    expected={'public.py':'a'*64}; events=[]; saved=[]
    envelope={'request':{'stage':stage,'staged_root':'/public/owned-attempt'},'phase':phase}
    raw=b'public envelope'
    class Held:
        def __init__(self,path):events.append(('held',path))
        def check(self):events.append(('check',))
        def close(self):events.append(('close',))
    def execute(request,*args):events.append(('execute',request['phase']))
    ns=dict(Path=Path,time=time,os=SimpleNamespace(umask=lambda n:0o077,environ={'JAX_PLATFORMS':'cpu'},getpid=lambda:123),
      write=lambda p,v:saved.append(dict(v)),hashes=lambda e:dict(e),
      key_gate=SimpleNamespace(parse=lambda b:expected if b==b'manifest' else envelope,read=lambda p,*a:b'manifest' if p=='manifest' else raw,digest=lambda b:'d'*64,sha_field=lambda x:x),
      authenticate=lambda r,e:(None,None,None,None,None,None,{'public':'binding'}),
      protocol=SimpleNamespace(binding=lambda *a:{'bound':True},declared_phases=lambda p:{'phase':p}),
      execution=SimpleNamespace(HeldDirectory=Held),local_modules=lambda e:{},execute=execute,
      terminal_scope=scope,cleanup_signals=scope)
    tree=ast.parse(SOURCE.read_bytes());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),ns)
    report=ns['run']('request','d'*64,'output','manifest')
    assert report['passed'] is True
    assert ('held','/public/owned-attempt') in events
    assert ('execute',phase) in events
    assert events[-1]==('close',)
