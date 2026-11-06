"""Exact focused-node runner, with backend import and discovery denials."""
import sys, json, os, hashlib, importlib.abc
from pathlib import Path

raw = Path(sys.argv[1]).read_bytes()
assert hashlib.sha256(raw).hexdigest() == sys.argv[2]
config = json.loads(raw)
assert config['synthetic_only'] is True and config['actual_execution_authorized'] is False
source = Path(config['proof_root']) / 'execution'
out = Path(config['proof_root'])
runtime_expected = config['runtime_source_expectations']
runtime_before = {n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in runtime_expected}
assert runtime_before == runtime_expected
assert os.sched_getaffinity(0) == {112, 113, 114, 115}
assert sys.flags.isolated and sys.dont_write_bytecode
assert os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD') == '1'
forbidden = set(['flax', 'jax', 'jaxite_word', 'jaxlib', 'libtpu', 'safetensors', 'tensorflow', 'tokenizers', 'torch', 'torch_xla', 'transformers'])
assert sorted(forbidden) == config['forbidden_roots'] and 'numpy' not in forbidden
assert len(runtime_expected) == len(config['runtime_module_origins']) == 3
denied, collected = [], []
assert not any(n.split('.')[0] in forbidden for n in sys.modules)
def audit(event, args):
    if event == 'import' and args and str(args[0]).split('.')[0] in forbidden:
        denied.append(str(args[0]))
        raise PermissionError('backend import disabled in owned held-parent proof')
class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in forbidden:
            denied.append(fullname)
            raise PermissionError('backend discovery disabled in owned held-parent proof')
sys.addaudithook(audit)
sys.meta_path.insert(0, DenyBackend())
sys.path.insert(0, str(source))
import pytest
class ExactSelection:
    def pytest_collection_finish(self, session):
        collected.extend(item.nodeid for item in session.items)
        if collected != config['selected_nodes']:
            raise pytest.UsageError('focused node selection differs before test execution')
selected = config['selected_files']
assert type(selected) is list and selected and len(selected) == len(set(selected))
assert all(type(n) is str and n.startswith('test_') and n.endswith('.py') and '/' not in n for n in selected)
status = pytest.main(['-q', '-p', 'no:cacheprovider', '--rootdir=' + str(source),
                     '--confcutdir=' + str(source), '--import-mode=importlib',
                     *[str(source / n) for n in selected],
                     '--junitxml=' + str(out / 'tests.xml')], plugins=[ExactSelection()])
loaded = [n for n in sys.modules if n.split('.')[0] in forbidden]
runtime_after = {n: hashlib.sha256(Path(n).read_bytes()).hexdigest() for n in runtime_expected}
module_origins = {}
for name, expected_path in config['runtime_module_origins'].items():
    module = sys.modules.get(name)
    observed = dict(file=str(Path(getattr(module, '__file__', '')).resolve()),
                    spec=str(Path(getattr(getattr(module, '__spec__', None), 'origin', '')).resolve()))
    module_origins[name] = observed
    assert observed == dict(file=expected_path, spec=expected_path)
assert runtime_after == runtime_before == runtime_expected
(out / 'worker-audit.json').write_text(json.dumps(dict(
    denied=denied, loaded_backend=loaded, collected_nodes=collected,
    runtime_sources_before=runtime_before, runtime_sources_after=runtime_after,
    runtime_module_origins=module_origins, forbidden_roots=sorted(forbidden),
    numpy_imported='numpy' in sys.modules,
    pytest_exit_code=int(status), affinity=sorted(os.sched_getaffinity(0)),
    pid=os.getpid(), pgid=os.getpgrp(), synthetic_only=True), sort_keys=True, indent=2) + '\n')
raise SystemExit(int(status) if not denied and not loaded else 70)
