"""Parent-only import guard; candidates run only in isolated owned children."""
import importlib.abc
import sys
import pytest
FORBIDDEN=frozenset(('numpy','jax','jaxlib','jaxite_word','libtpu','jax_plugins','torch','tensorflow','transformers','tokenizers','safetensors','flax','torch_xla'))
DENIED=[]
assert not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:
            DENIED.append(fullname);raise ImportError('native imports unavailable in integration parent')
sys.meta_path.insert(0,Deny())
@pytest.fixture(autouse=True)
def no_native_parent():
    yield
    assert not DENIED and not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)
