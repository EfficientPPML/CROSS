"""Parent-only guard and exact prefixture collection; no candidate imports."""
import importlib.abc
import json
from pathlib import Path
import sys
import pytest

FORBIDDEN=frozenset(('numpy','jax','jaxlib','jaxite_word','libtpu','jax_plugins','torch','tensorflow','transformers','tokenizers','safetensors','flax','torch_xla'))
DENIED=[]
assert not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0]in FORBIDDEN:
            DENIED.append(name);raise ImportError('genuine native parent imports forbidden')
sys.meta_path.insert(0,Deny())
def pytest_collection_modifyitems(session,config,items):
    expected=json.loads((Path(__file__).parent/'selected_tests.json').read_bytes())['nodes']
    assert [item.nodeid for item in items]==expected
@pytest.fixture(autouse=True)
def no_native_parent():
    yield
    assert not DENIED and not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)
