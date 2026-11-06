"""Backend exclusion before selected pytest module import; no candidate imports."""
import importlib.abc
import sys

FORBIDDEN=frozenset(('numpy','jax','jaxlib','libtpu','jax_plugins','jaxite_word','torch','tensorflow','transformers','tokenizers','safetensors'))
DENIALS=[]
assert not any(name.split('.')[0]in FORBIDDEN for name in sys.modules)
class NoBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:
            DENIALS.append(fullname);raise ImportError('snapshot tests forbid numeric/model backends')
        return None
sys.meta_path.insert(0,NoBackend())

def pytest_runtest_teardown():
    assert not DENIALS
    assert not any(name.split('.')[0]in FORBIDDEN for name in sys.modules)
