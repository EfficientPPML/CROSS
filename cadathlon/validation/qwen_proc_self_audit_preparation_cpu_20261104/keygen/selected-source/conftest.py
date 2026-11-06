"""Closed focused selection; no candidate/backend imports in pytest parent."""
import importlib.abc
import sys
import pytest

FORBIDDEN={'numpy','jax','jaxlib','jaxite_word','libtpu','torch','tensorflow','flax',
    'torch_xla','transformers','safetensors','tokenizers'}
ATTEMPTS=[]
assert not any(name.split('.')[0]in FORBIDDEN for name in sys.modules)

class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:
            ATTEMPTS.append(fullname)
            raise PermissionError('backend forbidden in proc integration parent')

sys.meta_path.insert(0,Deny())

@pytest.fixture(autouse=True)
def no_actual_backend():
    yield
    assert ATTEMPTS==[]
    assert not any(name.split('.')[0]in FORBIDDEN for name in sys.modules)
