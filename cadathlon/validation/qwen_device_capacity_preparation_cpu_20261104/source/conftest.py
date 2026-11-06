"""Parent guard: no NumPy, genuine backend or candidate runtime import here."""
import importlib.abc
import sys
import pytest
FORBIDDEN={'numpy','jax','jaxlib','libtpu','jax_plugins','jaxite_word','torch','tensorflow','transformers','tokenizers','safetensors'}
DENIED=[]
assert not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)
class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:
            DENIED.append(fullname);raise ImportError('device component parent forbids backends')
sys.meta_path.insert(0,DenyBackend())
@pytest.fixture(autouse=True)
def no_backend():
    yield
    assert not DENIED and not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)
