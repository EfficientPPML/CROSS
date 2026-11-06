"""Static selection check before any fixture; no runtime or NumPy imports."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def pytest_collection_modifyitems(session,config,items):
    expected=[r['nodeid']for r in json.loads((ROOT/'accepted-case-plan.json').read_bytes())['cases']]
    actual=[x.nodeid for x in items]
    if actual!=expected:raise RuntimeError('exact ordered55 focused nodes required before fixtures')
