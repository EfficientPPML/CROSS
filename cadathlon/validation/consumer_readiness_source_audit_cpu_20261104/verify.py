"""Public integrity/static-pin verification; no backend or captured code imports."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(name):return (ROOT/name).read_bytes()
def data(name):return json.loads(read(name))
def constant(name,key):
    found=[n.value for n in ast.parse(read(name)).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name)and t.id==key for t in n.targets)]
    assert len(found)==1
    return ast.literal_eval(found[0])
def function(name,key):
    return next(n for n in ast.parse(read(name)).body if isinstance(n,ast.FunctionDef)and n.name==key)

inventory=data('public_inventory.json');hashes=data('artifact_hashes.json')
assert inventory['schema']=='consumer-readiness-public-inventory-v1'
names=inventory['files'];assert type(names)is list and names==sorted(set(names))
assert set(names)==set(hashes)|{'artifact_hashes.json'}
actual=[]
for path in ROOT.rglob('*'):
    assert not path.is_symlink()
    if path.is_file():actual.append(path.relative_to(ROOT).as_posix())
assert sorted(actual)==names
for name,digest in hashes.items():
    path=Path(name)
    assert not path.is_absolute() and '..'not in path.parts and path.as_posix()==name
    assert sha(read(name))==digest
assert sha(read('review.json'))=='106fa16bed38bfc03c0982859dd7c9a3a3a83c585d3b19a26335e15aabc8caec'
report=data('review.json');assert report['passed']is True
maps={}
for label,identity in report['identities'].items():
    raw=read('maps/'+label+'.json');assert sha(raw)==identity['manifest_sha256']
    maps[label]=json.loads(raw);assert len(maps[label])==identity['source_count']
    for name,digest in report['examined_source_hashes'][label].items():
        assert sha(read('context/'+label+'/'+name))==digest==maps[label][name]
old='ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a'
grant='c3983fd975047bba16066106e85f9ae0768ed1e8c0fa30bef4da02b53253cd80'
assert constant('context/issuer6/freeze_issuer.py','SOURCE')==old
assert constant('context/issuer6/freeze_issuer.py','GRANT')==grant
assert constant('context/reconstructor44/contract.py','SOURCE_MANIFEST_SHA256')==old
assert constant('context/heldout171/heldout_authorization.py','CALIBRATION_SOURCE')==old
assert constant('context/heldout171/heldout_authorization.py','CALIBRATION_GRANT')==grant
assert read('context/issuer6/freeze_admission.py')==read('context/heldout171/freeze_admission.py')
assert read('context/candidate139/checker_process.py')==read('context/candidate141/checker_process.py')
assert ast.dump(function('context/candidate141/entry.py','terminal_status'),include_attributes=False)==ast.dump(function('context/heldout171/heldout_entry.py','terminal_status'),include_attributes=False)
for name in ('checker_runtime.py','candidate_domains.py','strict_metrics.py','test_checker_runtime.py','test_candidate_domains.py','test_strict_metrics.py'):
    assert maps['candidate139'][name]==maps['candidate141'][name]==maps['heldout171'][name]
assert all(maps['heldout171'][name]==digest for name,digest in maps['candidate139'].items())
assert not any(name.endswith('ckks_ctx.py') for source in maps.values() for name in source)
for label in ('candidate139','candidate141','heldout171'):
    assert data('context/'+label+'/execution_authorization.json')=={'execute':False}
assert 'if not passed:'in read('context/reconstructor44/producer_admission.py').decode()
assert 'externally barred'in read('README.md').decode()
print(json.dumps(dict(passed=True,public_files=len(names),selected_source_files=sum(map(len,report['examined_source_hashes'].values())),source_maps=len(maps),scope='public integrity/static bindings only; no numerical test or execution authority'),sort_keys=True))
