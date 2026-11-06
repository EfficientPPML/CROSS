"""Focused synthetic-only test imports; no actual backend/model entry."""
import gc,importlib.abc,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
FORBIDDEN={'jax','jaxlib','jaxite_word','torch','transformers','tensorflow','safetensors','libtpu'}
class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:raise RuntimeError('synthetic adapter tests forbid backend/model imports: '+fullname)
guard=DenyBackend()
sys.meta_path.insert(0,guard)
@pytest.fixture(autouse=True)
def no_backend():
    assert not (FORBIDDEN&set(sys.modules))
    yield
    assert not (FORBIDDEN&set(sys.modules))
    # Only these new synthetic component quarantine lists are reset by tests;
    # no real contexts, mappings or producer-owned resources exist here.
    for name in ('qwen_public_parameters','qwen_file_lifecycle','reused_client_io'):
        module=sys.modules.get(name)
        if module is not None:module._QUARANTINE.clear()
    gc.collect()
@pytest.fixture(scope='session')
def public():
    from qwen_file_contract import PublicContract
    from public_source_pins import PUBLIC_PINS
    return PublicContract.from_blobs({name:(HERE/'parent208'/name).read_bytes()for name in PUBLIC_PINS})
@pytest.fixture
def request_case(public):
    from qwen_file_contract import request
    identity=public.identity(context_id='1'*64,model_id='2'*64,producer_sources='3'*64,consumer_sources='4'*64)
    return request(public,identity,'synthetic_request',[0,3,19,31,40,100,101,102,103,104,105,106,107,108,109,110,200])
