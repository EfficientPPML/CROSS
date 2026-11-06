"""Source gate and genuine NumPy-only guard; no candidate import at authoring."""
import hashlib
import importlib.abc
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parent
DENIED=[]
FORBIDDEN={'jax','jaxlib','jaxite_word','libtpu','torch','torch_xla','tensorflow','transformers','tokenizers','safetensors','flax','ckks_ctx','key_gen','polynomial'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            DENIED.append(fullname);raise ImportError('Q5 focused proof forbids backend imports')
        return None

def audit(event,args):
    if event=='import' and args and str(args[0]).split('.')[0]in FORBIDDEN:
        DENIED.append(str(args[0]));raise ImportError('Q5 backend audit denial')

def pytest_sessionstart(session):
    expected=json.loads((ROOT/'runtime_sources.json').read_bytes())
    assert len(expected)==250
    for name,pin in expected.items():
        p=ROOT/'runtime'/name;assert not p.is_symlink() and sha(p)==pin
    pins=json.loads((ROOT/'numpy_pins.json').read_bytes())
    for path,pin in pins['sources'].items():assert sha(Path(path))==pin
    assert not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)
    sys.meta_path.insert(0,Deny());sys.addaudithook(audit)
    sys.path.insert(0,str(ROOT/'runtime'))
    sys.path.insert(0,str(ROOT))

def pytest_collection_modifyitems(session,config,items):
    selected=json.loads((ROOT/'selected_tests.json').read_bytes())['ordered_nodes']
    assert [item.nodeid for item in items]==selected,'exact focused collection required before fixtures'

def pytest_sessionfinish(session,exitstatus):
    expected=json.loads((ROOT/'runtime_sources.json').read_bytes())
    for name,pin in expected.items():assert sha(ROOT/'runtime'/name)==pin
    pins=json.loads((ROOT/'numpy_pins.json').read_bytes())
    for path,pin in pins['sources'].items():assert sha(Path(path))==pin
    for name,origin in pins['required_origins'].items():
        module=sys.modules.get(name);assert module is not None and str(Path(module.__file__).resolve())==origin
    assert not DENIED and not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)

@pytest.fixture(scope='session')
def math_fixture():
    import numpy as np
    from q5_test_fixtures import fixed_math
    return fixed_math(np)
