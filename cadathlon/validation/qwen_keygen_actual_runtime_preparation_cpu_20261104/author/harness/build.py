"""Stdlib-only source copying and static harness finalization; no candidate imports."""
from pathlib import Path
import ast
import difflib
import hashlib
import itertools
import json
import shutil
import tarfile

BASE = Path('/tmp/qwen-client-authority-boundary-proof-20261104/preparation-v2')
HANDOFF = Path('/tmp/qwen-keygen-actual-runtime-test-handoff-20261104')
OUT = Path('/tmp/qwen-keygen-actual-focused-proof-20261104/preparation-v1')
sha = lambda b: hashlib.sha256(b).hexdigest()
pretty = lambda v: (json.dumps(v, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
compact = lambda v: (json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
def inventory(root):
    paths = sorted(root.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in paths if p.is_file()}

raw = (HANDOFF / 'handoff.json').read_bytes()
assert sha(raw) == '3c7ff98036d62820c67a8361af9bc39d44a5746db1e77e238628f194e34395f3'
handoff = json.loads(raw)
source = Path(handoff['source_root'])
map_raw = (HANDOFF / 'sources.json').read_bytes()
expected = json.loads(map_raw)
assert map_raw == compact(expected)
assert sha(map_raw) == handoff['source_manifest_sha256'] == '59d8d58cd42d6beb8265b632b9c08277de2d913f5f0190364ca2d40689e3d8d3'
assert inventory(source) == expected and len(expected) == 97
assert sha((HANDOFF / 'sources.tar.gz').read_bytes()) == handoff['source_archive_sha256']
parent_raw = (source / 'parent85_sources.json').read_bytes()
parent = json.loads(parent_raw)
assert parent_raw == compact(parent)
assert sha(parent_raw) == '28782d1927a4334c778dc8429d92637776bfe6a97a6a42e0d25f99cc82f35699'
assert len(parent) == 85 and inventory(source / 'runtime') == parent
nodes = []
for filename in handoff['selected_files']:
    tree = ast.parse((source / filename).read_text())
    for fn in tree.body:
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith('test_'):
            continue
        axes = []
        for d in reversed(fn.decorator_list):
            assert isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == 'parametrize'
            assert len(d.args) == 2 and not d.keywords
            ast.literal_eval(d.args[0])
            axis = ast.literal_eval(d.args[1])
            assert type(axis) is list
            axes.append(axis)
        suffixes = [''] if not axes else ['[' + '-'.join(map(str, row)) + ']' for row in itertools.product(*axes)]
        nodes.extend(filename + '::' + fn.name + suffix for suffix in suffixes)
assert nodes == handoff['selected_nodes'] and len(nodes) == len(set(nodes)) == 61
xml = [dict(classname=n.split('::')[0][:-3], name=n.split('::')[1]) for n in nodes]
assert xml == handoff['selected_xml_cases']
with tarfile.open(HANDOFF / 'sources.tar.gz') as archive:
    members = archive.getmembers()
    assert len(members) == len({m.name for m in members}) == 97
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    assert {m.name: sha(archive.extractfile(m).read()) for m in members} == expected

OUT.mkdir(parents=True, exist_ok=False)
shutil.copytree(source, OUT / 'execution')
(OUT / 'sources-before.json').write_bytes(map_raw)
shutil.copy2(HANDOFF / 'sources.tar.gz', OUT / 'sources.tar.gz')
(OUT / 'source-handoff.json').write_bytes(raw)
shutil.copy2(source / 'selected_tests.json', OUT / 'selected-tests.json')

base_driver = (BASE / 'executed_driver.py').read_text()
base_worker = (BASE / 'worker_driver.py').read_text()
assert sha(base_driver.encode()) == '0cbe8adabba255e6c3b0b4bb53017201eaeee0ec417c05a8b6e93c79e82fe554'
assert sha(base_worker.encode()) == '3dfb645c744143d1e1b7699791680dffd549ec179b16d708348e0e67c4b4ec97'
worker = base_worker.replace('{108, 109, 110, 111}', '{112, 113, 114, 115}').replace('pure client-authority proof', 'synthetic keygen-runtime proof')
(OUT / 'worker_driver.py').write_text(worker)
worker_sha = sha(worker.encode())
scope = ('Exactly61 new checks:15 owned evidence-output,19 pinned policy/fake-kernel metadata,6 recorded generation-lifetime,21 entry/terminal cases. '
         'All85 runtime bytes remain exact. Real filesystem work is restricted to tiny newly owned test fixtures and one default-denied isolated CLI. '
         'Generation uses degree2 recording doubles and335 metadata descriptors, with no real arrays, keygen primitives, RNG, model/key/private/CT bodies or capacity acquisition. '
         'Exactly three run-flow cases temporarily insert source-reviewed ModuleType jax/numpy name doubles; origin tests use an isolated module dictionary. '
         'No genuine NumPy/backend import is allowed; all backend names must be absent at worker completion. '
         'The numpy_imported audit field observes terminal module presence, not historical temporary name injection. No inherited suite or actual role authority.')
config = json.loads((BASE / 'run_config.json').read_bytes())
config.update(schema='qwen-keygen-runtime-focused-proof-v1', source_root=str(source), proof_root=str(OUT),
    source_count=97, source_manifest_sha256=sha(map_raw), source_archive_sha256=handoff['source_archive_sha256'],
    source_manifest_encoding='canonical sorted compact JSON plus newline', cpu_affinity=[112,113,114,115],
    selected_files=handoff['selected_files'], selected_nodes=nodes, selected_xml_cases=xml,
    selected_nodes_sha256=sha(pretty(nodes)), selected_xml_cases_sha256=sha(pretty(xml)),
    selection_digest_encoding='sorted indent=2 JSON plus newline', scope=scope,
    runtime_source_expectations={}, runtime_module_origins={},
    temporary_module_name_doubles=handoff['temporary_module_name_doubles'],
    genuine_backend_imports_authorized=False, capacity_acquisition_authorized=False)
config_raw = pretty(config)
(OUT / 'run_config.json').write_bytes(config_raw)
config_sha = sha(config_raw)
driver = base_driver.replace('client-authority', 'keygen-runtime')
driver = driver.replace('qwen-keygen-runtime-focused-proof-v1', config['schema'])
driver = driver.replace('e146143187f957f9b5c1908c42a57462f42dbd0bb4ee08b1e2c7f1c8d242ce1c', config_sha)
driver = driver.replace('3dfb645c744143d1e1b7699791680dffd549ec179b16d708348e0e67c4b4ec97', worker_sha)
driver = driver.replace('{108, 109, 110, 111}', '{112, 113, 114, 115}').replace('[108, 109, 110, 111]', '[112, 113, 114, 115]').replace("'108-111'", "'112-115'")
needle = "canonical = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\\n').encode()"
assert needle in driver
driver = driver.replace(needle, needle + "\ncompact_map = lambda value: (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\\n').encode()")
driver = driver.replace('assert manifest == canonical(expected)', "assert config['source_manifest_encoding'] == 'canonical sorted compact JSON plus newline'\n    assert manifest == compact_map(expected)")
driver = driver.replace("dump('sources-after.json', current)", "(out / 'sources-after.json').write_bytes(compact_map(current))")
driver = driver.replace("dump('execution-after.json', after)", "(out / 'execution-after.json').write_bytes(compact_map(after))")
driver = driver.replace("config_sha256=CONFIG_SHA256, scope=config['scope'])", "config_sha256=CONFIG_SHA256, scope=config['scope'],\n        temporary_module_name_doubles=config['temporary_module_name_doubles'],\n        genuine_backend_imports_authorized=False, capacity_acquisition_authorized=False)")
(OUT / 'executed_driver.py').write_text(driver)
driver_sha = sha(driver.encode())
for name, before, after in [('executed_driver.py',base_driver,driver),('worker_driver.py',base_worker,worker)]:
    ast.parse(after)
    (OUT / (name + '.patch')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='admitted-pure53-v2/'+name,tofile='keygen97/'+name)))

def popen_owner(text):
    tree=ast.parse(text)
    found=[]
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(isinstance(x,ast.Assign) and isinstance(x.value,ast.Call)
             and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='Popen' for x in node.body):
            assert len(node.body)==2 and isinstance(node.body[1],ast.Try)
            found.append(ast.dump(node,include_attributes=False))
    assert len(found)==1
    return found[0]
assert popen_owner(base_driver)==popen_owner(driver)

proposal=json.loads((BASE/'launch-proposal.json').read_bytes())
proposal.update(schema='qwen-keygen-runtime-focused-launch-proposal-v1',source_count=97,
    source_manifest_sha256=sha(map_raw),source_archive_sha256=handoff['source_archive_sha256'],
    driver_sha256=driver_sha,worker_sha256=worker_sha,config_sha256=config_sha,
    selected_files=handoff['selected_files'],selected_source_sha256=handoff['selected_tests_sha256'],
    static_selected_count=61,scope=scope)
proposal['argv']=[s.replace('108-111','112-115').replace(str(BASE),str(OUT)) for s in proposal['argv']]
(OUT/'launch-proposal.json').write_bytes(pretty(proposal))
provenance=dict(schema='qwen-keygen-runtime-focused-harness-provenance-v1',predecessor_root=str(BASE),
    predecessor_driver_sha256=sha(base_driver.encode()),predecessor_worker_sha256=sha(base_worker.encode()),
    parent85_compact_map_sha256=sha(parent_raw),all85_runtime_exact=True,
    source_manifest_encoding=config['source_manifest_encoding'],
    selection_digest_encoding=config['selection_digest_encoding'],
    explicit_changes=['97 source/map/archive and61 selected cases/config pins','CPU108-111 to112-115',
      'source map validation and two terminal map outputs use compact sorted JSON plus newline',
      'schema/scope labels and temporary ModuleType disclosure; empty installed sources/origins retained',
      'worker changes only affinity and two diagnostic labels'],
    immediate_post_Popen_ownership_AST_identical=True, no_candidate_import_collection_execution=True,
    genuine_numpy_backend_forbidden=True, runtime_source_expectations={},runtime_module_origins={},
    scope=scope,execution_authorized=False)
(OUT/'driver_provenance.json').write_bytes(pretty(provenance))
(OUT/'README.md').write_text('''# Focused key-generation runtime preparation

This is a held, unexecuted harness for the exact 97-source tree and 61 statically selected cases. Root review and a separate explicit launch authorization are required. No collection, candidate import, test execution, original body read, capacity observation, RNG or actual role is performed by this preparation.

The driver and worker are adapted from the admitted pure53-v2 harness. CPU affinity is 112–115. The source map and eventual terminal source maps use canonical compact sorted JSON plus newline. Config, reports and selection digest documents use sorted indent=2 JSON plus newline. The immediate post-Popen try, direct wait without a healthy deadline, exceptional process-group cleanup and direct reap are AST-identical to the admitted driver.

Genuine NumPy and backend imports are forbidden; installed dependency hashes and module-origin expectations are empty. Three recording cases temporarily install `ModuleType` jax/numpy doubles; these are not backend implementations or actual imports. Separate origin tests use an isolated synthetic module dictionary. The existing audit and final module check require zero import denials and no forbidden names remaining. The inherited `numpy_imported` audit field means module presence at worker completion, not that temporary names were never used.

The selected cases cover small owned output files, public-byte policy admission, fake-kernel resource observations, degree2 recorded generation and terminal handling. One isolated real CLI checks the default-denied entry. There are no real key arrays, full-size allocations, backend primitives, context construction or original model/key/private/ciphertext reads. All 85 runtime bytes are unchanged, and inherited tests are unselected. No result or readiness claim is made.
''')
result=dict(schema='qwen-keygen-runtime-prepared-harness-handoff-v1',proof_root=str(OUT),source_root=str(source),
    source_count=97,static_count=61,source_manifest_sha256=sha(map_raw),source_archive_sha256=handoff['source_archive_sha256'],
    source_manifest_encoding=config['source_manifest_encoding'],config_sha256=config_sha,driver_sha256=driver_sha,
    worker_sha256=worker_sha,launch_proposal_sha256=sha((OUT/'launch-proposal.json').read_bytes()),
    execution_authorized=False,imports_collection_execution=False,cpu_affinity=[112,113,114,115],
    NumPy_forbidden=True,runtime_source_expectations={},runtime_module_origins={},
    source_handoff_sha256=sha(raw),source_peer_review_sha256='2497b7eb9c5582ff53b2ec5920133fae4928e3e9b1c80597aaf2e99b831d916d',
    temporary_module_name_doubles=handoff['temporary_module_name_doubles'],
    immediate_post_Popen_ownership_AST_identical=True)
(OUT/'handoff.json').write_bytes(pretty(result))
assert inventory(source)==inventory(OUT/'execution')==expected
assert inventory(source/'runtime')==parent
assert not any((OUT/n).exists() for n in ['report.json','tests.xml','tests.log','live-observation.json','worker-audit.json'])
print(json.dumps(dict(**result,handoff_sha256=sha((OUT/'handoff.json').read_bytes())),sort_keys=True))
