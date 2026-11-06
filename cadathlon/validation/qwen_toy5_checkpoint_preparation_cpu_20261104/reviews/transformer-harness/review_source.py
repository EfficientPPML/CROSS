"""Q5 harness source-only peer readback. No candidate imports or collection."""
import ast
import difflib
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).parent
PROOF=Path('/tmp/qwen-toy5-checkpoint-focused-proof-20261104/preparation-v1')
SOURCE=Path('/tmp/qwen-toy5-checkpoint-tests-20261104')
BASE=Path('/tmp/qwen-client-authority-boundary-proof-20261104/preparation-v2')
sha=lambda b:hashlib.sha256(b).hexdigest()
raw={n:(PROOF/n).read_bytes()for n in('executed_driver.py','worker_driver.py','run_config.json','handoff.json','driver_provenance.json','sources-before.json','source-handoff.json')}
assert sha(raw['executed_driver.py'])=='d32615321f455240e3bbd599a29b64c1ac5395f1c02e67f45d6358269b662241'
assert sha(raw['worker_driver.py'])=='d2c04361908ba05b84b4b69369dd1cef94091def36454d6863dfe077b67bd8e1'
assert sha(raw['run_config.json'])=='c6e5dcc8fe8118f7a2b7751af9ef3cbf1b4b98e4e4b5397f9a25720fe584a14b'
config=json.loads(raw['run_config.json']);handoff=json.loads(raw['handoff.json']);provenance=json.loads(raw['driver_provenance.json'])
for name,h in handoff['files'].items():assert sha((PROOF/name).read_bytes())==h,name
mapping=json.loads(raw['sources-before.json'])
assert len(mapping)==261 and sha(raw['sources-before.json'])==config['source_manifest_sha256']
assert raw['sources-before.json']==(json.dumps(mapping,sort_keys=True,separators=(',',':'))+'\n').encode()
def inventory(root):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    return {str(p.relative_to(root)):sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}
assert inventory(SOURCE)==inventory(PROOF/'execution')==mapping
archive=(PROOF/'sources.tar.gz').read_bytes()
assert sha(archive)==config['source_archive_sha256']
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
    members=tar.getmembers();assert len(members)==261 and all(m.isfile()for m in members)
    assert len({m.name for m in members})==261
    assert{m.name:sha(tar.extractfile(m).read())for m in members}==mapping
selected_raw=(SOURCE/'selected_tests.json').read_bytes();selected=json.loads(selected_raw)
assert sha(selected_raw)==config['selection_sha256']
assert config['selected_nodes']==selected['ordered_nodes']and len(config['selected_nodes'])==len(set(config['selected_nodes']))==48
assert config['selected_files']==list(dict.fromkeys(n.split('::')[0]for n in config['selected_nodes']))
canonical=lambda x:(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
assert sha(canonical(config['selected_nodes']))==config['selected_nodes_sha256']
assert sha(canonical(config['selected_xml_cases']))==config['selected_xml_cases_sha256']
assert config['selected_xml_cases']==[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1])for n in config['selected_nodes']]
runtime=json.loads((SOURCE/'runtime_sources.json').read_bytes());assert len(runtime)==250
assert{n.removeprefix('runtime/'):h for n,h in mapping.items()if n.startswith('runtime/')}==runtime
numpyraw=(SOURCE/'numpy_pins.json').read_bytes();numpy=json.loads(numpyraw)
assert sha(numpyraw)==config['numpy_pins_sha256']
assert numpy['sources']==config['runtime_source_expectations']and len(numpy['sources'])==4
assert numpy['required_origins']==config['runtime_module_origins']and set(numpy['required_origins'])=={'numpy'}
for path,h in numpy['sources'].items():assert sha(Path(path).read_bytes())==h
old={n:(BASE/n).read_bytes()for n in('executed_driver.py','worker_driver.py')}
assert sha(old['executed_driver.py'])==provenance['base_driver_sha256']
assert sha(old['worker_driver.py'])==provenance['base_worker_sha256']
for n in old:
    patch=''.join(difflib.unified_diff(old[n].decode().splitlines(True),raw[n].decode().splitlines(True),fromfile=str(BASE/n),tofile=str(PROOF/n))).encode()
    assert patch==(PROOF/(n+'.patch')).read_bytes()
def owned_try(blob):
    tree=ast.parse(blob)
    nodes=[n for n in ast.walk(tree)if isinstance(n,ast.Try)and any(isinstance(a,ast.Attribute)and a.attr=='getpgid'for a in ast.walk(n))]
    assert len(nodes)==1
    return ast.dump(nodes[0],include_attributes=False)
assert owned_try(raw['executed_driver.py'])==owned_try(old['executed_driver.py'])
conftest=(SOURCE/'conftest.py').read_bytes();ct=ast.parse(conftest)
worker=raw['worker_driver.py'].decode();driver=raw['executed_driver.py'].decode()
assert worker.index('runtime_before =')<worker.index('import pytest')
assert worker.index('sys.addaudithook(audit)')<worker.index('import pytest')
assert worker.index('sys.meta_path.insert(0, DenyBackend())')<worker.index('import pytest')
assert 'def pytest_collection_finish' in worker and 'collected != config[\'selected_nodes\']'in worker
assert any(isinstance(n,ast.FunctionDef)and n.name=='pytest_sessionstart'for n in ct.body)
assert any(isinstance(n,ast.FunctionDef)and n.name=='pytest_collection_modifyitems'for n in ct.body)
assert '--rootdir='in worker and '--confcutdir='in worker and '--basetemp='in worker and "'no:cacheprovider'"in worker
assert 'child = subprocess.Popen' in driver and 'code = child.wait()'in driver
assert "env = dict(PATH='/usr/bin:/bin'"in driver and "'108-111'"in driver
for name,value in raw.items():assert(PROOF/name).read_bytes()==value
assert inventory(SOURCE)==inventory(PROOF/'execution')==mapping
report={
 'schema':'q5-focused-harness-source-peer-review-v1','passed':True,'scope':'Harness/selection/source ownership review only; no independent scientific test-source verdict.',
 'harness_pins':{n:sha(v)for n,v in raw.items()},'source_count':261,'runtime_unchanged':250,'static_cases':48,
 'source_archive_sha256':sha(archive),'conftest_sha256':sha(conftest),
 'checks':{'all_handoff_pins_exact':True,'current_execution_archive_exact':True,'ordered48_selection_and_xml_exact':True,'numpy_four_sources_and_single_root_origin_pins_match':True,'immediate_owned_child_try_ast_matches_admitted_base':True,'exact_declared_driver_worker_patches':True,'all_reviewed_bytes_unchanged_at_end':True},
 'source_conclusions':[
  'Outer driver verifies exact config/worker, canonical261 source map, current execution bytes and archive members before spawning. It launches fixed isolated bytecode-disabled Python on108-111 in a sanitized environment with plugin autoload disabled.',
  'Every post-Popen action is within the original ownership try. Abnormal cleanup attempts owned group kill and direct wait in nested finally, preserving the original exception; normal wait has no healthy deadline. Postwait group absence and original child PID/PGID are required for a passing report.',
  'Worker hashes NumPy4 Python sources before pytest and installs backend import/discovery denials first. Selected conftest sessionstart checks runtime250 and NumPy hashes before test module collection, then installs its own sticky guards and explicit runtime paths.',
  'Only four selected test files are passed; collection-modify and collection-finish hooks require exact48 ordered nodes before fixtures. Parent requires ordered48 XML, no failures/errors/skips, zero denials/loaded forbidden roots and clean worker exit.',
  'NumPy root __file__/spec origin is required; all4 selected Python files are rehashed after. Unused NPY helper modules are not artificially imported and a full native NumPy closure is not claimed.',
  'Pytest basetemp must be a fresh leaf under the proof root; retained regular-file count/byte limits and link rejection are post-run retention checks, not OS confinement.',
  'No selected test asks for tmp_path or launches another process; nested cleanup here refers to the preserved owned worker cleanup structure, not a claimed nested scientific process.',
 ],
 'scoped_notes':[
  'Worker has explicit rootdir/confcutdir, importlib import mode, cache suppression and sanitized environment, but no explicit -c config. These alone do not prove ancestor pytest configuration discovery is disabled. Root stated it will independently inspect ancestor config absence before any grant; this reviewer did not inspect that environment or claim a concrete ambient-config failure.',
  'The reviewed audit is backend import/discovery exclusion, not arbitrary filesystem/network/native-code confinement. Installed NumPy4 source hashes and root origin do not certify all native library bytes.',
  'The driver does not independently certify its final publication close; a future actual result still requires external owner exit/reap/root admission. No grant follows from this source receipt.',
  'Public toy arithmetic with ideal refresh slots and synthetic scale metadata is the declared future scope; prior140 PASS, actual encrypted checkpoint capture, HE/model execution, resource measurement and first encrypted-divergence claims are excluded.',
 ],
 'no_additional_concrete_blocker_in_bounded_harness_scope':True,
 'candidate_imports':False,'numpy_imports':False,'pytest_collection':False,'tests_or_probes':False,'body_reads':False,'capacity_observation':False,'launch_authorized':False,
 'review_script_sha256':sha(Path(__file__).read_bytes()),
}
out=HERE/'review.json'
with out.open('xb')as f:f.write((json.dumps(report,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps({'passed':True,'scope':'source only','receipt':str(out),'sha256':sha(out.read_bytes())}))
