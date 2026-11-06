"""External synthetic-only driver: exactly nineteen new client boundaries."""
import sys, json, os, importlib.abc
from pathlib import Path

source = Path(sys.argv[1])
out = Path(sys.argv[2])
assert os.sched_getaffinity(0) == {108, 109, 110, 111}
forbidden = {'jax', 'jaxlib', 'jaxite_word', 'torch', 'transformers',
             'tensorflow', 'safetensors', 'libtpu'}
denied = []

def audit(event, args):
    if event == 'import' and args and str(args[0]).split('.')[0] in forbidden:
        denied.append(str(args[0]))
        raise PermissionError('backend import disabled in recording client proof')

class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            denied.append(fullname)
            raise PermissionError('backend discovery disabled in recording client proof')

sys.addaudithook(audit)
sys.meta_path.insert(0, DenyBackend())
sys.path.insert(0, str(source))
import pytest
status = pytest.main(['-q', '-p', 'no:cacheprovider', '--rootdir=' + str(source),
                     '--confcutdir=' + str(source), '--import-mode=importlib',
                     str(source / 'test_client_stage.py'),
                     '--junitxml=' + str(out / 'tests.xml')])
loaded = [n for n in sys.modules if n.split('.')[0] in forbidden]
(out / 'worker-audit.json').write_text(json.dumps(dict(
    denied=denied, loaded_backend=loaded, pytest_exit_code=int(status),
    affinity=sorted(os.sched_getaffinity(0)), pid=os.getpid(), pgid=os.getpgrp(),
    synthetic_only=True, actual_backend_imported=False), sort_keys=True, indent=2) + '\n')
raise SystemExit(int(status) if not denied and not loaded else 70)
