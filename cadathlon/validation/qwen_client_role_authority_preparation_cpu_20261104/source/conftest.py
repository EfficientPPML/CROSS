"""Pure metadata fixture setup. NumPy and every model/backend import are forbidden."""
import importlib.abc
from pathlib import Path
import sys
import pytest
HERE=Path(__file__).resolve().parent
FORBIDDEN={'numpy','jax','jaxlib','jaxite_word','torch','transformers','tensorflow','safetensors','libtpu'}
class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:raise RuntimeError('pure client authority forbids backend/NumPy discovery')
if any(n.split('.')[0]in FORBIDDEN for n in sys.modules):raise RuntimeError('forbidden preloaded module')
sys.meta_path.insert(0,DenyBackend());sys.path.insert(0,str(HERE))
@pytest.fixture(autouse=True)
def no_backend():
    assert not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)
    yield
    assert not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)
@pytest.fixture
def client_chain():
    from qwen_file_contract import PublicContract
    from public_source_pins import PUBLIC_PINS
    from synthetic_client_chain import build_client_chain
    public=PublicContract.from_blobs({name:(HERE/'public_inputs'/name).read_bytes()for name in PUBLIC_PINS})
    aggregate=(HERE/'synthetic_aggregate_metadata.json').read_bytes()
    return lambda **changes:build_client_chain(public,aggregate,**changes)
