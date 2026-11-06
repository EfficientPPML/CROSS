"""Focused recording/owned-small-file tests; no backend or model admission."""
import importlib.abc,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
FORBIDDEN={'jax','jaxlib','jaxite_word','torch','transformers','tensorflow','safetensors','libtpu'}
class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:raise RuntimeError('synthetic producer forbids backend/model imports: '+fullname)
sys.meta_path.insert(0,DenyBackend())
if FORBIDDEN.intersection(sys.modules):raise RuntimeError('backend/model already loaded before producer tests')
@pytest.fixture(autouse=True)
def no_backend():
    assert not FORBIDDEN.intersection(sys.modules)
    yield
    assert not FORBIDDEN.intersection(sys.modules)
    # These tests retire actual small-file descriptors; controlled close-error
    # cases call real close before raising. No real key/context owners exist.
    for name,attribute in (('qwen_key_store','_QUARANTINE'),('qwen_key_producer','_QUARANTINE'),('key_descriptor_io','_IO_FAILURE_QUARANTINE')):
        if name in sys.modules:getattr(sys.modules[name],attribute).clear()
@pytest.fixture(scope='session')
def public():
    from qwen_file_contract import PublicContract
    from public_source_pins import PUBLIC_PINS
    return PublicContract.from_blobs({n:(HERE/'public_inputs'/n).read_bytes()for n in PUBLIC_PINS})
