"""Recording-only composition tests; selected backend modules are never imported."""
import importlib.abc,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
FORBIDDEN={'jax','jaxlib','jaxite_word','torch','transformers','tensorflow','safetensors','libtpu'}
class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:raise RuntimeError('recording preparation forbids backend/model imports: '+fullname)
guard=DenyBackend();sys.meta_path.insert(0,guard)
@pytest.fixture(autouse=True)
def no_backend():
    assert not FORBIDDEN.intersection(sys.modules)
    yield
    assert not FORBIDDEN.intersection(sys.modules)
    import qwen_backend_composition as b
    b._QUARANTINE.clear()
    assert not b._ACTIVE.locked()
@pytest.fixture(scope='session')
def public():
    from qwen_file_contract import PublicContract
    from public_source_pins import PUBLIC_PINS
    return PublicContract.from_blobs({n:(HERE/'adapter238/parent208'/n).read_bytes()for n in PUBLIC_PINS})
