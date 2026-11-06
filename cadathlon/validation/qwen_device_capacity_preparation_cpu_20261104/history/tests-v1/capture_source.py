"""Source-only static selection and closure capture; no candidate imports."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/tmp/qwen-device-component-tests-20261104')
OUT=Path(__file__).resolve().parent
MANIFEST=Path('/tmp/qwen-device-component-test-sources-v1-20261104.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
encode=lambda x:(json.dumps(x,sort_keys=True,indent=2)+'\n').encode()
copies=json.loads((ROOT/'copied_sources.json').read_bytes())
assert len(copies)==36
for name,row in copies.items():
    assert sha((ROOT/name).read_bytes())==row['sha256']==sha(Path(row['path']).read_bytes())
assert copies['runtime/qwen_server_resource.py']['sha256']=='912d811a1a3a366fc1b10a2677d46673d0650ef3979f2458abc8431a630e1238'
assert copies['runtime/qwen_device_capacity.py']['sha256']=='9853af30057ec5851f6f69430afe4a526ad3e3958b6f5f895f9c33b7bb696f97'
runtime={p.relative_to(ROOT/'runtime').as_posix():sha(p.read_bytes())for p in sorted((ROOT/'runtime').rglob('*'))if p.is_file()}
assert len(runtime)==34
(ROOT/'runtime_sources.json').write_bytes(encode(runtime))
case_tree=ast.parse((ROOT/'device_cases.py').read_bytes())
cases=ast.literal_eval(next(n.value for n in case_tree.body if isinstance(n,ast.Assign)))
assert len(cases)==len(set(cases))==30 and sum(n.startswith('document_')for n in cases)==8
selected=dict(schema='qwen-device-component-static-selection-v1',selected_files=['test_device_component.py'],
    static_count=30,document_cases=8,state_cases=22,
    static_nodeids=['test_device_component.py::test_device_component['+n+']'for n in cases],
    child_entry='device_child.py',runtime_files=34,device_files=6,
    genuine_backend_origins={},recording_jax_source='runtime/recording_jax.py',
    whole_resource_validator_seam=False,phase_validation_seams=2,resource_code_profile_observer=True,
    collected=False,executed=False,scientific_admission_tested=False,genuine_capacity_measured=False)
(ROOT/'selected_tests.json').write_bytes(encode(selected))
old=Path('/tmp/qwen-server-admission-snapshot-tests-v3-20261104/test_snapshot_guard.py').read_bytes()
new=(ROOT/'test_device_component.py').read_bytes()
def wait_try(raw):
    return next(n for n in ast.walk(ast.parse(raw))if isinstance(n,ast.Try)and any(isinstance(c,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='code'for t in c.targets)for c in n.body))
assert ast.dump(wait_try(old),include_attributes=False)==ast.dump(wait_try(new),include_attributes=False)
(OUT/'parent_snapshot38.py').write_bytes(old)
(OUT/'parent_owner_adaptation.patch').write_text(''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='snapshot38/test_snapshot_guard.py',tofile='device/test_device_component.py')))
plan=Path('/tmp/qwen-device-component-test-plan-20261104/PLAN.md')
refs={
    'accepted_plan':str(plan),
    'device6_handoff':'/tmp/qwen-server-device-admission-handoff-v2-20261104/handoff.json',
    'device5_finding':'/tmp/qwen-device-component-transformers-review-20261104/review.json',
    'device6_peer':'/tmp/qwen-device-identity-cnn-review-20261104/review.json',
    'packet27_map':'/tmp/qwen-server-public-packet-sources-v1-20261104.json',
    'snapshot_parent':'/tmp/qwen-server-admission-snapshot-tests-v3-20261104/test_snapshot_guard.py',
}
origin=dict(schema='qwen-device-component-test-origin-v1',references={n:dict(path=p,sha256=sha(Path(p).read_bytes()))for n,p in refs.items()},
    copied_file_count=36,copied_sources_manifest_sha256=sha((ROOT/'copied_sources.json').read_bytes()),
    runtime_files=34,device6_preserved=True,pure27_exact=True,resource_and_cleanup_helpers_exact=True,
    source_only_plan_refinement='Real source912d authenticate_policy and logical_facts selected; whole-resource seam removed before source hold.',
    two_phase_checks_only_recording=True,resource_functions_replaced=False,profile_observer_only=True,
    synthetic_process_and_jax_are_new=['runtime/qwen_server_process.py','runtime/recording_jax.py'],
    registry_reset=False,wait_cleanup_AST_exact=True,actual_server387_edited=False,
    candidate_imports=False,collection=False,execution=False,device_capacity_probes=False,actual_policy_instances=False)
(ROOT/'origin.json').write_bytes(encode(origin))
for p in ROOT.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
mapping={p.relative_to(ROOT).as_posix():sha(p.read_bytes())for p in sorted(ROOT.rglob('*'))if p.is_file()}
assert len(mapping)==49
MANIFEST.write_bytes(encode(mapping));(OUT/'sources.json').write_bytes(MANIFEST.read_bytes())
with tarfile.open(OUT/'sources.tar.gz','w:gz')as tar:
    for n in sorted(mapping):tar.add(ROOT/n,arcname=n,recursive=False)
with tarfile.open(OUT/'sources.tar.gz')as tar:
    assert{m.name:sha(tar.extractfile(m).read())for m in tar.getmembers()}==mapping
assert all(sha((ROOT/n).read_bytes())==d for n,d in mapping.items())
assert all(sha(Path(r['path']).read_bytes())==r['sha256']for r in copies.values())
handoff=dict(schema='qwen-device-component-focused-test-source-handoff-v1',source_root=str(ROOT),source_count=len(mapping),
    source_manifest=str(MANIFEST),source_manifest_sha256=sha(MANIFEST.read_bytes()),source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
    copied_files=36,runtime_files=34,device6_exact=True,pure27_exact=True,resource912d_exact=True,static_count=30,
    selected_files=selected['selected_files'],static_nodeids=selected['static_nodeids'],source_hashes=mapping,
    wait_cleanup_AST_exact=True,parent_adaptation_sha256=sha((OUT/'parent_owner_adaptation.patch').read_bytes()),
    recording_jax_explicit=True,genuine_backend_imported=False,resource_validator_seam=False,phase_check_seams=2,
    candidate_imported=False,collected=False,executed=False,genuine_capacity_measured=False,actual_grants=False,
    held_for_source_review_only=True)
(OUT/'handoff.json').write_bytes(encode(handoff))
print(json.dumps({k:v for k,v in handoff.items()if k not in('source_hashes','static_nodeids')},sort_keys=True))
print('handoff_sha256='+sha((OUT/'handoff.json').read_bytes()))
