"""Bounded source/hash/AST review; never import candidates or run tests."""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/tmp/qwen-server-admission-snapshot-tests-20261104')
HANDOFF=Path('/tmp/qwen-server-admission-snapshot-test-handoff-20261104')
MAP=Path('/tmp/qwen-server-admission-snapshot-test-sources-v1-20261104.json')
OUT=Path('/tmp/qwen-snapshot-test-cnn-review-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):
    before=path.lstat();assert path.is_file()and not path.is_symlink()
    raw=path.read_bytes();after=path.lstat()
    fields=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    assert fields(before)==fields(after)and len(raw)==after.st_size
    return raw
handoff=json.loads(read(HANDOFF/'handoff.json'));map_raw=read(MAP);expected=json.loads(map_raw)
assert sha(map_raw)=='8ecdb98742f8f79dd9988cad12a0b4b79c6b33415315c022741dc65b12ff27b2'
assert len(expected)==36 and expected==handoff['source_hashes']
actual={str(p.relative_to(ROOT)):sha(read(p))for p in ROOT.rglob('*')if p.is_file()}
assert actual==expected and not any(p.is_symlink()for p in ROOT.rglob('*'))
with tarfile.open(HANDOFF/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==36 and all(m.isfile()for m in members)
    assert {m.name:sha(archive.extractfile(m).read())for m in members}==expected
assert sha(read(HANDOFF/'sources.tar.gz'))==handoff['source_archive_sha256']
runtime=json.loads(read(ROOT/'runtime_sources.json'))
assert len(runtime)==28 and {'runtime/'+n:d for n,d in runtime.items()}=={n:d for n,d in expected.items()if n.startswith('runtime/')}
parent_path=Path('/tmp/qwen-server-public-packet-sources-v1-20261104.json')
parent_raw=read(parent_path);parent=json.loads(parent_raw)
assert sha(parent_raw)=='4961e6dbe93b4905d99130a873e90bcfdddfc711ed51aaef032bb3544308edff'
assert len(parent)==27
assert {n:d for n,d in runtime.items()if n!='qwen_admission_snapshot.py'}==parent
guard=Path('/tmp/qwen-server-admission-snapshot-source-v2-20261104/qwen_admission_snapshot.py')
assert read(guard)==read(ROOT/'runtime/qwen_admission_snapshot.py')
assert sha(read(guard))=='8790711a1a402f4c21ef444ccdb76f773b4c62cd334f5be07a49d2cd4535b395'
test=ast.parse(read(ROOT/'test_snapshot_guard.py'))
cases=ast.literal_eval(next(n.value for n in test.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CASES'for t in n.targets)))
selection=json.loads(read(ROOT/'selected_tests.json'))
nodes=['test_snapshot_guard.py::test_snapshot_guard['+c+']'for c in cases]
assert len(cases)==len(set(cases))==18 and nodes==selection['static_nodeids']==handoff['static_nodeids']
fn=next(n for n in test.body if isinstance(n,ast.FunctionDef)and n.name=='test_snapshot_guard')
owned=next(n for n in fn.body if isinstance(n,ast.With))
assert isinstance(owned.body[0],ast.Assign)and isinstance(owned.body[1],ast.Try)
primary=owned.body[1].handlers[0]
cleanup=primary.body[0]
assert isinstance(cleanup,ast.Try)and cleanup.finalbody
assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='wait'for n in ast.walk(ast.Module(body=cleanup.finalbody,type_ignores=[])))
source=read(ROOT/'test_snapshot_guard.py').decode()
assert "if observed_pgid is not None:assert data['pgid']==observed_pgid"in source
assert "data['pgid']==proc.pid"not in source and "proc.pid==data['pgid']"not in source
child=ast.parse(read(ROOT/'snapshot_child.py'))
main=next(n for n in child.body if isinstance(n,ast.FunctionDef)and n.name=='main')
types=ast.literal_eval(next(n.value for n in main.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='expected_types'for t in n.targets)))
assert len(types)==13
reviewed=['TEST_SCOPE.md','conftest.py','origin.json','pytest.ini','runtime_sources.json','selected_tests.json','snapshot_child.py','test_snapshot_guard.py','runtime/qwen_admission_snapshot.py','runtime/qwen_file_contract.py','runtime/model_admission.py','runtime/qwen_server_authority_packet.py','runtime/qwen_scientific_admission.py','runtime/qwen_server_admission.py']
report=dict(schema='qwen-snapshot-test-cnn-source-review-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(ROOT),source_manifest_sha256=sha(map_raw),source_archive_sha256=handoff['source_archive_sha256'],source_count=36,runtime_count=28,parent27_exact=True,guard_v2_exact=True,static_case_count=18,ordered_static_selection_exact=True,current_archive_exact=True,
    ready_for_focused_run=False,source_only=True,tests_or_imports=False,collection=False,backend_or_body_reads=False,capacity_probes=False,
    findings=[dict(id='parent-child-pgid-binding',severity='test ownership receipt assertion gap',path='test_snapshot_guard.py',line=51,condition='Parent getpgid raises ProcessLookupError and observed_pgid remains None.',observed_source='Child self-reported pgid is compared only when observed_pgid is non-None, then used for the group-absence probe.',required='Require exact int reported PGID equal to proc.pid from start_new_session=True unconditionally, in addition to any observed PGID comparison; retain original source as unrun predecessor.',execution_reproduction=False,runtime_failure_observed=False)],
    reviewed_assertions=['Exact13 admission dataclass graph and25 proof-name markers; recording full validator seam disclosed and restored.', 'Clean capture full-check count1 versus repeated hot checks/retirement; one-issued and sticky registry failures remain.', 'Identity-equal byte/tuple replacements, dictionary replacement, unknown fields/mutable graph, helper code/export/default/descriptor mutations target actual guard checks.', 'Wrong packet/thread, caught and uncaught reentry, validator/callback failures retain original marker identity and graph fields.', 'Immediate post-Popen ownership and nested direct wait after attempted group cleanup; no healthy deadline.'],
    limitations=['These are source assertions, not executed results.','Synthetic marker bytes are intentionally invalid as scientific proofs; full validator replaced by the explicit recording seam.','No actual admission, resource sufficiency, body/private ordering, backend, capacity or HE claim.','No process was launched or reaped by this reviewer.'],
    read_targets={n:expected[n]for n in reviewed},review_script_sha256=sha(read(Path(__file__))))
for n,d in expected.items():assert sha(read(ROOT/n))==d
(OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(source_closure_checks_passed=True,ready_for_focused_run=False,confirmed_source_findings=1,review_sha256=sha(read(OUT/'review.json')),candidate_execution=False),sort_keys=True))
