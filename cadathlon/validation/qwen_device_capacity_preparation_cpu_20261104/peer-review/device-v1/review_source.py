"""Source-only device component peer review; no component import or probe."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).parent
ROOT=Path('/tmp/qwen-server-device-admission-source-20261104')
HAND=Path('/tmp/qwen-server-device-admission-handoff-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
handoff_raw=(HAND/'handoff.json').read_bytes();handoff=json.loads(handoff_raw)
mapraw=(HAND/'sources.json').read_bytes();mapping=json.loads(mapraw)
assert sha(mapraw)=='5fdb8fc1927c3118fcb2acd65b218039ab1f1bb9b097f963fa10e92dded4c7b3'
assert len(mapping)==5 and set(mapping)=={p.name for p in ROOT.iterdir()}
raw={n:(ROOT/n).read_bytes()for n in mapping};assert{n:sha(v)for n,v in raw.items()}==mapping
archive=(HAND/'sources.tar.gz').read_bytes();assert sha(archive)=='cc93bedfbba903089e8c890ff0287bc1696b08ca5936fff19978171ce36716c9'
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
    members=tar.getmembers();assert len(members)==5 and all(m.isfile()for m in members)
    assert{m.name:sha(tar.extractfile(m).read())for m in members}==mapping
policy=ast.parse(raw['qwen_device_policy.py']);capacity=ast.parse(raw['qwen_device_capacity.py'])
constants={n.targets[0].id:ast.literal_eval(n.value)for n in policy.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in('ROLES','STAGES','HOOKS','MODULES','COVERAGE')}
assert len(constants['ROLES'])==10 and len(constants['STAGES'])==4 and len(constants['HOOKS'])==5 and len(constants['COVERAGE'])==14
cls=next(n for n in policy.body if isinstance(n,ast.ClassDef)and n.name=='DevicePolicyAdmission')
assert tuple(n.target.id for n in cls.body if isinstance(n,ast.AnnAssign))==('phase','blobs','expected_pins')
text=raw['qwen_device_capacity.py'].decode()
assert "policy=record['policy'];policy.check()"in text
assert "result=fn(record['policy'].phase,record['policy'].blobs,*args)"in text
assert "record=dict(permit=permit,policy=policy,"in text
assert "if record['state']!='busy':raise RuntimeError('nested or failed device transition is sticky')"in text
for n,v in raw.items():assert(ROOT/n).read_bytes()==v
report={
 'schema':'qwen-device-component-source-peer-review-v1','review_completed':True,'source_checks_passed':True,'no_unresolved_blockers':False,
 'source_map_sha256':sha(mapraw),'source_archive_sha256':sha(archive),'handoff_sha256':sha(handoff_raw),'source_hashes':mapping,
 'closed_documents':list(constants['ROLES']),'ordered_stages':list(constants['STAGES']),'source_fixed_hooks':list(constants['HOOKS']),
 'findings':[{
  'id':'DEVICE-ORIGINAL-POLICY-FIELDS','status':'concrete source-only local invariant gap; no execution or bypass demonstration',
  'locations':['issue_initialization record construction','_context policy.check','_hook current phase/blobs arguments'],
  'detail':'The registry retains the DevicePolicyAdmission object but not independently captured identities of its original phase, blobs and expected_pins. Frozen dataclass fields can be rebound through object.__setattr__, within the same callback-mutation boundary addressed by the other retained owners. _context then authenticates the current coherent fields, _caller derives source/hook bindings from them, and _hook supplies those current phase/blob values. Equal-content replacements can pass pure authentication; a coherent different policy could also rebase unless the external process hook rejects it.',
  'scope':'This does not establish actual authority bypass: the documented process hook must independently enforce original identities, current server hooks/policies are absent, and selected complete-capacity still denies. Nevertheless the component does not locally meet its stated original-input/one-permit invariant.',
  'minimal_repair':'Capture the original three fields before any fallible check/callback, retain them with the attempt, compare their identities before reauthentication and after callbacks, and invoke source-fixed hooks with those captured original values. Reject rather than rebase on replacement. Preserve current5 unchanged and use a declared source successor.'}],
 'source_conclusions':[
  'Ten immutable canonical bounded documents are externally hashed and joined to exact phase/source/archive/dependency/resource hashes; resource selection and allocator links, workspace request/source joins,14 review subjects and exclusive lease links are explicit. No default numerical allowance, self-issued grant or hash-only policy fallback is supplied.',
  'Source-fixed hook lookup requires five top-level Python functions with exact module/global/code origins, no defaults or closures, and retained process module/spec/function/code identities. Independent helper-source origins/hashes and the hook binding contract remain necessary; this is not arbitrary-Python/native containment.',
  'Registry insertion precedes initial callbacks and retains the policy/returned owners on failure. Second issuance during capture/busy and nested public transitions poison the attempt; _hook checks busy after the callback, so a caught nested hook transition cannot restore a permitted/observed state through that hook path.',
  'Initialization precedes ordered post_init/post_keys/pre_evaluation/terminal observations. Statistics require explicit bytes_in_use/bytes_limit, exact nonnegative integers, consistent limits and reviewed additional/workspace/reserve comparison; no missing-stat fallback or reservation is claimed.',
  'The selected device/JAX class, memory_stats descriptor/method, selection callables, original default-device identity and one-device/process checks remain explicit. Actual native semantics, exclusivity, synchronization and complete workspace bounds are independent factual/process-hook prerequisites, not proven by this component.',
  'Clean retirement removes only the guard registry after terminal observation. Hardware/default-device/array/lease retirement remains with original process/external owners; failed attempts remain retained and no retry is granted.',
 ],
 'limits':['Only declared source/hook callback boundary reviewed; no genuine backend type/memory_stats compatibility execution.','Mutable interpreter/private registry globals and arbitrary native tampering remain outside the documented trust boundary.','No actual policy instance, access path, capacity result, initialization or integration was inspected or supplied.'],
 'current_five_preserved':True,'candidate_imports':False,'collection_or_tests':False,'device_or_capacity_probes':False,'actual_policy_instances':False,'body_reads':False,'grants_or_launch':False,
 'review_script_sha256':sha(Path(__file__).read_bytes()),
}
out=HERE/'review.json'
with out.open('xb')as f:f.write((json.dumps(report,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps({'review_completed':True,'source_checks_passed':True,'unresolved_source_findings':1,'receipt':str(out),'sha256':sha(out.read_bytes())}))
