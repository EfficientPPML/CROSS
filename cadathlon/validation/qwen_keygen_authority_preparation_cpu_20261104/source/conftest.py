"""Pure source/metadata fixtures only; no actual model/key authority."""
import importlib.abc
from pathlib import Path
import sys
import pytest

HERE = Path(__file__).resolve().parent
FORBIDDEN = {'numpy', 'jax', 'jaxlib', 'jaxite_word', 'torch', 'tensorflow',
             'transformers', 'safetensors', 'libtpu'}


class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise RuntimeError('pure keygen authority tests forbid backend/NumPy discovery: '+fullname)


if FORBIDDEN.intersection(sys.modules):
    raise RuntimeError('backend/NumPy preloaded before pure authority tests')
sys.meta_path.insert(0, DenyBackend())
sys.path.insert(0, str(HERE))


@pytest.fixture(autouse=True)
def no_backend():
    assert not FORBIDDEN.intersection(sys.modules)
    yield
    assert not FORBIDDEN.intersection(sys.modules)


@pytest.fixture
def chain():
    from qwen_file_contract import PublicContract
    from public_source_pins import PUBLIC_PINS
    from synthetic_keygen_chain import build_chain
    public = PublicContract.from_blobs({name:(HERE/'public_inputs'/name).read_bytes()
                                       for name in PUBLIC_PINS})
    aggregate = (HERE/'synthetic_aggregate_metadata.json').read_bytes()
    return lambda **changes: build_chain(public, aggregate, **changes)
