"""Pure-parent guard and exact pre-fixture collection; no current run claim."""
import importlib.abc
import json
from pathlib import Path
import sys

FORBIDDEN=frozenset(('numpy','jax','jaxlib','jaxite_word','torch','transformers',
    'tensorflow','flax','torch_xla','safetensors','tokenizers','libtpu',
    'key_gen','ckks_ctx','polynomial'))
DENIALS=[]

class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:
            DENIALS.append(fullname)
            raise ImportError('backend forbidden in pure kernel-policy parent')
        return None

def pytest_sessionstart(session):
    if any(n.split('.')[0]in FORBIDDEN for n in sys.modules):raise AssertionError('forbidden preloaded backend')
    sys.meta_path.insert(0,Guard())

def pytest_collection_finish(session):
    selected=json.loads((Path(__file__).parent/'selected_tests.json').read_bytes())
    if [item.nodeid for item in session.items]!=selected['nodes']:
        raise AssertionError('exact24 nodes/order required before fixtures')

def pytest_sessionfinish(session,exitstatus):
    if DENIALS or any(n.split('.')[0]in FORBIDDEN for n in sys.modules):
        session.exitstatus=1
