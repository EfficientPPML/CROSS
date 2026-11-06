"""Author an unrun, fixed focused harness; do not import or collect candidates."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

SOURCE=Path('/tmp/qwen-toy5-checkpoint-tests-20261104')
MAP=Path('/tmp/qwen-toy5-checkpoint-test-sources-v1-20261104.json')
SOURCE_HANDOFF=Path('/tmp/qwen-toy5-checkpoint-test-source-handoff-20261104')
OUT=Path('/tmp/qwen-toy5-checkpoint-focused-proof-20261104/preparation-v1')
BASE=Path('/tmp/qwen-client-authority-boundary-proof-20261104/preparation-v2')
BUILD=Path('/tmp/qwen-toy5-checkpoint-harness-build-20261104')

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def compact(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def save(name,raw):
    if not isinstance(raw,bytes):raw=raw.encode()if isinstance(raw,str)else canonical(raw)
    with(OUT/name).open('xb')as f:f.write(raw)
def change(source,before,after):
    assert source.count(before)==1,before
    return source.replace(before,after)

map_raw=MAP.read_bytes();expected=json.loads(map_raw)
assert sha(map_raw)=='91a4bc8c1715f8900edb49e97cc9c2be696b30a5037b5efd4a136ac088664339'
assert map_raw==compact(expected)and len(expected)==261
assert sorted(str(p.relative_to(SOURCE))for p in SOURCE.rglob('*')if p.is_file())==sorted(expected)
assert not any(p.is_symlink()for p in SOURCE.rglob('*'))
execution=OUT/'execution';execution.mkdir()
for name,pin in expected.items():
    raw=(SOURCE/name).read_bytes();assert sha(raw)==pin
    target=execution/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
save('sources-before.json',map_raw)
archive=(SOURCE_HANDOFF/'sources.tar.gz').read_bytes()
assert sha(archive)=='6d890c097757327e1f4de6a624a1496dd40e466703210e617975bbf64e26e21a'
save('sources.tar.gz',archive)
with tarfile.open(OUT/'sources.tar.gz')as tar:
    assert {m.name:sha(tar.extractfile(m).read())for m in tar}==expected
selection=json.loads((SOURCE/'selected_tests.json').read_bytes())
numpy=json.loads((SOURCE/'numpy_pins.json').read_bytes())
assert selection['expected_tests']==len(selection['ordered_nodes'])==48
for path,pin in numpy['sources'].items():assert sha(Path(path).read_bytes())==pin
forbidden=sorted(['flax','jax','jaxite_word','jaxlib','libtpu','safetensors','tensorflow','tokenizers','torch','torch_xla','transformers','ckks_ctx','key_gen','polynomial'])
config=dict(schema='qwen-toy5-checkpoint-focused-proof-v1',source_root=str(SOURCE),proof_root=str(OUT),
    source_count=261,source_manifest_sha256=sha(map_raw),source_manifest_encoding='canonical sorted compact JSON plus newline',source_archive_sha256=sha(archive),
    selected_files=selection['test_files'],selected_nodes=selection['ordered_nodes'],selected_xml_cases=selection['ordered_xml_cases'],
    selected_nodes_sha256=sha(canonical(selection['ordered_nodes'])),selected_xml_cases_sha256=sha(canonical(selection['ordered_xml_cases'])),
    selection_sha256=sha((SOURCE/'selected_tests.json').read_bytes()),cpu_affinity=[108,109,110,111],
    pytest_basetemp=str(OUT/'pytest-owned'),pytest_basetemp_policy='Fresh leaf under this proof only; no selected test requests tmp_path. Inspect at most128 regular files/16MiB after execution; refuse links. This is a retention bound, not OS confinement.',
    pytest_basetemp_max_files=128,pytest_basetemp_max_bytes=16*1024*1024,
    runtime_source_expectations=numpy['sources'],runtime_module_origins=numpy['required_origins'],numpy_pins_sha256=sha((SOURCE/'numpy_pins.json').read_bytes()),
    forbidden_roots=forbidden,synthetic_only=True,actual_execution_authorized=False,
    scope='Exactly48 new focused Q5 cases:7 newly computed47-checkpoint literal/dense/diagnostic tests,21 binary64 metadata cases consuming19 synthetic actual-return observations,12 metadata-only Ledger state cases,8 immediate-denial/terminal cases. Fixed public seed9105 toy math and genuine NumPy only; refresh slots ideal identity, metadata synthetic. No inherited140 PASS, first encrypted-divergence claim, RNS/transport/loader/backend/context/body/capacity/actual role. All250 runtime files unchanged. No inherited suite.')
save('run_config.json',config);config_sha=sha((OUT/'run_config.json').read_bytes())

worker_before=(BASE/'worker_driver.py').read_text();worker=worker_before
worker=change(worker,"forbidden = set(['flax', 'jax', 'jaxite_word', 'jaxlib', 'libtpu', 'numpy', 'safetensors', 'tensorflow', 'tokenizers', 'torch', 'torch_xla', 'transformers'])",'forbidden = set('+repr(forbidden)+')')
worker=change(worker,"assert runtime_expected == {} and config['runtime_module_origins'] == {}", "assert len(runtime_expected) == 4 and set(config['runtime_module_origins']) == {'numpy'}")
worker=worker.replace('pure client-authority proof','Q5 public-toy checkpoint proof')
worker=change(worker,"selected = config['selected_files']", "basetemp = Path(config['pytest_basetemp'])\nassert basetemp == out / 'pytest-owned' and basetemp.parent.resolve() == out.resolve()\nassert not os.path.lexists(basetemp)\nselected = config['selected_files']")
worker=change(worker,"'--confcutdir=' + str(source), '--import-mode=importlib',", "'--confcutdir=' + str(source), '--import-mode=importlib',\n                     '--basetemp=' + str(basetemp),")
save('worker_driver.py',worker);worker_sha=sha(worker.encode())

driver_before=(BASE/'executed_driver.py').read_text();driver=driver_before
driver=driver.replace('client-authority','Q5 checkpoint')
driver=change(driver,"assert config['schema'] == 'qwen-Q5 checkpoint-focused-proof-v1'", "assert config['schema'] == 'qwen-toy5-checkpoint-focused-proof-v1'")
# Only the literal pins differ from the known admitted owned-Popen model.
tree=ast.parse(driver)
values={n.targets[0].id:ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in ('CONFIG_SHA256','WORKER_SHA256')}
for name,new in [('CONFIG_SHA256',config_sha),('WORKER_SHA256',worker_sha)]:
    driver=change(driver,name+' = '+repr(values[name]),name+' = '+repr(new))
driver=change(driver,"canonical = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\\n').encode()", "canonical = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\\n').encode()\nmap_canonical = lambda value: (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\\n').encode()")
driver=change(driver,'assert manifest == canonical(expected)','assert manifest == map_canonical(expected)')
driver=change(driver,"    start = time.monotonic()", "    basetemp = Path(config['pytest_basetemp'])\n    assert basetemp == out / 'pytest-owned' and basetemp.parent.resolve() == out.resolve()\n    assert not os.path.lexists(basetemp)\n    start = time.monotonic()")
driver=change(driver,"    dump('sources-after.json', current)\n    dump('execution-after.json', after)", "    (out / 'sources-after.json').write_bytes(map_canonical(current))\n    (out / 'execution-after.json').write_bytes(map_canonical(after))\n    owned_temp = {}\n    if basetemp.exists():\n        assert basetemp.is_dir() and not basetemp.is_symlink()\n        total = 0\n        for base, dirs, files in os.walk(basetemp, followlinks=False):\n            for name in dirs:\n                assert not (Path(base) / name).is_symlink()\n            for name in files:\n                path = Path(base) / name\n                assert not path.is_symlink() and path.is_file()\n                size = path.stat().st_size\n                total += size\n                assert len(owned_temp) < config['pytest_basetemp_max_files'] and total <= config['pytest_basetemp_max_bytes']\n                owned_temp[path.relative_to(basetemp).as_posix()] = dict(bytes=size, sha256=sha(path.read_bytes()))\n    dump('owned-pytest-metadata.json', owned_temp)")
driver=change(driver,"and audit['numpy_imported'] is False","and audit['numpy_imported'] is True")
driver=change(driver,"        config_sha256=CONFIG_SHA256, scope=config['scope'])", "        config_sha256=CONFIG_SHA256, scope=config['scope'],\n        public_toy_arithmetic=True, synthetic_refresh_metadata=True, actual_encrypted_metadata_captured=False,\n        prior140_PASS_inherited=False, first_encrypted_divergence_claim=False, capacity_measurement=False,\n        pytest_basetemp=str(basetemp), owned_pytest_files=len(owned_temp))")
save('executed_driver.py',driver);driver_sha=sha(driver.encode())

# Prove the original immediate-Popen try/except/wait structure is unchanged.
def owned_block(text):
    root=ast.parse(text)
    fn=next(n for n in root.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    block=next(n for n in fn.body if isinstance(n,ast.With))
    return ast.dump(block,include_attributes=False)
assert owned_block(driver)==owned_block(driver_before)
for name,text in [('worker_driver.py',worker),('executed_driver.py',driver)]:ast.parse(text,filename=name)
save('executed_driver.py.patch',''.join(difflib.unified_diff(driver_before.splitlines(True),driver.splitlines(True),fromfile=str(BASE/'executed_driver.py'),tofile=str(OUT/'executed_driver.py'))))
save('worker_driver.py.patch',''.join(difflib.unified_diff(worker_before.splitlines(True),worker.splitlines(True),fromfile=str(BASE/'worker_driver.py'),tofile=str(OUT/'worker_driver.py'))))
save('source-handoff.json',(SOURCE_HANDOFF/'handoff.json').read_bytes())
save('driver_provenance.json',dict(schema='q5-focused-harness-source-provenance-v1',base=str(BASE),base_driver_sha256=sha(driver_before.encode()),base_worker_sha256=sha(worker_before.encode()),same_immediate_Popen_ownership_AST=True,changes=['Fixed Q5 schema/scope/count/pins/current source and proof paths','Compact canonical source-map validation and post-source maps','NumPy four source hashes and single required root origin; genuine backends denied','Fresh bounded retained pytest basetemp under own proof root','Exact48 collection/ordered XML; no inherited suite'],source_count=261,static_tests=48,collection=False,imports=False,execution=False,actual_grant=False))
argv=['env','-i','PATH=/usr/bin:/bin','HOME=/home/jianming_gatech','LANG=C.UTF-8','TZ=UTC','PYTHONDONTWRITEBYTECODE=1','PYTEST_DISABLE_PLUGIN_AUTOLOAD=1','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','taskset','-c','108-111','/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(OUT/'executed_driver.py')]
save('launch-proposal.json',dict(schema='q5-focused-unrun-launch-proposal-v1',authorized=False,executed=False,argv=argv,driver_sha256=driver_sha,worker_sha256=worker_sha,config_sha256=config_sha,source_manifest_sha256=sha(map_raw),source_archive_sha256=sha(archive),selected_cases=48,scope=config['scope'],healthy_deadline=False,automatic_retry=False,owner='assigned externally by root at any future grant'))
save('README.md','''# Held Q5 focused harness — UNRUN

This exact source proposal is not a launch grant. Root and peer review plus a distinct authorization are required before imports, collection or execution. It selects only the declared four files/48 nodes over exact261 sources, including unchanged nested250. Expected source-map bytes are canonical sorted compact JSON with newline; config, reports and selection hashes use sorted indented JSON with newline.

The source retains the admitted pure53v2 immediate-Popen ownership block exactly. All post-spawn actions are inside its ownership try; abnormal cleanup nests direct wait in finally after group cleanup. Normal execution has no healthy deadline. CPU108–111, isolated Python, bytecode/plugin suppression and sanitized environment are fixed. NumPy4 Python sources are rechecked, and only the genuine NumPy root origin is mandatory; other NPY modules are not artificially imported. The required backend roots remain forbidden by discovery and audit guards.

Pytest receives one fresh own `pytest-owned` leaf under this proof. No selected test requests tmp_path. Its optional post-run retention is limited to128 regular files/16MiB and refuses links; this is not OS confinement or a resource-capacity guarantee. The original source tree and isolated execution copy must match the exact tar/map before and after. Any actual result requires independent owner terminal/reap and root acceptance; a report cannot certify its own final close.

No tests, collection, NumPy arithmetic/imports, backend, metadata/capacity probes, body reads or actual role occurred in preparing this harness. New47 comparisons are pending; previous140 PASS does not transfer.
''')
metadata={p.name:sha(p.read_bytes())for p in OUT.iterdir()if p.is_file()}
handoff=dict(schema='q5-focused-harness-held-handoff-v1',proof_root=str(OUT),source_root=str(SOURCE),source_count=261,static_cases=48,source_manifest_sha256=sha(map_raw),source_archive_sha256=sha(archive),config_sha256=config_sha,worker_sha256=worker_sha,driver_sha256=driver_sha,proposal_sha256=metadata['launch-proposal.json'],files=metadata,argv=argv,source_current_execution_archive_equal=True,immediate_Popen_ownership_AST_exact=True,collection=False,candidate_imports=False,numpy_imports=False,execution=False,metadata_probes=False,capacity_observations=False,actual_role_authorized=False,ready_for_source_review_only=True)
save('handoff.json',handoff)
print(json.dumps({k:v for k,v in handoff.items()if k not in ('files','argv')},sort_keys=True))
