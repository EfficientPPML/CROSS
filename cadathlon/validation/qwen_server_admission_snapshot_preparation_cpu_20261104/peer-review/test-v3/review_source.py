"""Static v3 fixture/receipt delta review; no imports or collection."""
import ast
import copy
import hashlib
import json
import tarfile
from pathlib import Path

H=Path('/tmp/qwen-server-admission-snapshot-test-handoff-v3-20261104')
OLD=Path('/tmp/qwen-server-admission-snapshot-tests-v2-20261104')
OUT=Path('/tmp/qwen-snapshot-v3-security-source-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def inv(r):
    paths=sorted(r.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return{p.relative_to(r).as_posix():sha(p.read_bytes())for p in paths if p.is_file()}
def dump(n):return ast.dump(n,include_attributes=False)
def tree(r,n):return ast.parse((r/n).read_bytes(),filename=n)

handoff=json.loads((H/'handoff.json').read_bytes());assert sha((H/'handoff.json').read_bytes()).startswith('4a36414d')
ROOT=Path(handoff['source_root']);expected=json.loads((H/'sources.json').read_bytes())
assert len(expected)==38 and sha((H/'sources.json').read_bytes())==handoff['source_manifest_sha256']
assert inv(ROOT)==expected==handoff['source_hashes']
assert sha((H/'sources.tar.gz').read_bytes())==handoff['source_archive_sha256']
with tarfile.open(H/'sources.tar.gz','r:gz')as archive:
    members=archive.getmembers();assert len(members)==len({m.name for m in members})==38
    assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
    assert{m.name:sha(archive.extractfile(m).read())for m in members}==expected
oldmap=json.loads(Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v1/sources-before.json').read_bytes())
assert len(oldmap)==37 and inv(OLD)==oldmap
assert set(expected)-set(oldmap)=={'successor_history_v3.json'}
assert{n for n in oldmap if oldmap[n]!=expected[n]}=={'TEST_SCOPE.md','snapshot_child.py','test_snapshot_guard.py'}
assert inv(ROOT/'runtime')==inv(OLD/'runtime')==json.loads((ROOT/'runtime_sources.json').read_bytes())
assert len(inv(ROOT/'runtime'))==28

oldchild=tree(OLD,'snapshot_child.py');newchild=tree(ROOT,'snapshot_child.py')
oldmain=next(n for n in oldchild.body if isinstance(n,ast.FunctionDef)and n.name=='main')
newmain=next(n for n in newchild.body if isinstance(n,ast.FunctionDef)and n.name=='main')
def public_index(fn):return next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='public')
i=public_index(newmain);oldi=public_index(oldmain);assert i==oldi
new_assign=ast.parse('public=object.__new__(file_module.PublicContract)').body[0]
new_set=ast.parse("object.__setattr__(public,'blobs',(('synthetic-public-marker',b'synthetic non-proof metadata'),))").body[0]
assert dump(newmain.body[i])==dump(new_assign)and dump(newmain.body[i+1])==dump(new_set)
# Normalize exactly the two-statement constructor bypass back to the old one.
# Everything else, including all guard assertions and monkeypatch restoration,
# must then be identical as an AST.
newmain.body[i:i+2]=[copy.deepcopy(oldmain.body[oldi])]
assert dump(newchild)==dump(oldchild)

oldtest=tree(OLD,'test_snapshot_guard.py');newtest=tree(ROOT,'test_snapshot_guard.py')
oldfn=next(n for n in oldtest.body if isinstance(n,ast.FunctionDef)and n.name=='test_snapshot_guard')
newfn=next(n for n in newtest.body if isinstance(n,ast.FunctionDef)and n.name=='test_snapshot_guard')
def exit_assert(fn):return next(i for i,n in enumerate(fn.body)if isinstance(n,ast.Assert)and isinstance(n.test,ast.Compare)and isinstance(n.test.left,ast.Name)and n.test.left.id=='code')
old_at=exit_assert(oldfn);new_at=exit_assert(newfn);assert new_at==old_at+3
extra=newfn.body[old_at:new_at]
assert isinstance(extra[0],ast.Try)and isinstance(extra[1],ast.With)and isinstance(extra[2],ast.Assert)
assert ast.unparse(extra[2].test)=='owned_group_absent'
assert 'parent-terminal.json'in ast.unparse(extra[1])and "'child_result_read': False"in ast.unparse(extra[1])
assert "'owned_session_pgid': proc.pid"in ast.unparse(extra[1])and "'observed_pgid': observed_pgid"in ast.unparse(extra[1])
assert "'reaped': True"in ast.unparse(extra[1])and "'exit_code': code"in ast.unparse(extra[1])
del newfn.body[old_at:new_at]
assert dump(newtest)==dump(oldtest)
cases=ast.literal_eval(next(n.value for n in oldtest.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='CASES'))
assert len(cases)==18 and ['test_snapshot_guard.py::test_snapshot_guard['+x+']'for x in cases]==handoff['static_nodeids']
assert handoff['selected_files']==['test_snapshot_guard.py']
failure=Path('/tmp/qwen-snapshot-failed-run-retention-20261104/review.json')
assert sha(failure.read_bytes())=='0bb3fb900f912c57c4cb39ebc81030d4151a8850284abc6d9fe3a70f8ad206dd'
assert inv(ROOT)==expected and inv(OLD)==oldmap
report=dict(schema='qwen-snapshot-v3-security-source-review-v1',passed=True,concrete_blockers=[],
    handoff_sha256=sha((H/'handoff.json').read_bytes()),source_manifest_sha256=handoff['source_manifest_sha256'],
    source_archive_sha256=handoff['source_archive_sha256'],source_count=38,runtime_count=28,static_cases=18,
    child_sha256=expected['snapshot_child.py'],test_sha256=expected['test_snapshot_guard.py'],
    checks=['current/archive38 exact; failed37 and runtime28 unchanged',
        'only explicit PublicContract object allocation/blobs fixture assignment changes child AST',
        'all intended guard assertions and original validator monkeypatch/restore AST unchanged',
        'only three early parent receipt statements added before exit/result assertions',
        'owned versus observed PGID distinguished; direct wait precedes receipt',
        'unchanged18 static scenarios/selection; earlier failed run retained'],
    limitations=['Explicit synthetic constructor bypass does not authenticate source blobs or scientific evidence.',
        'Early receipt covers completed direct-wait paths, not every exceptional cleanup path.',
        'A new harness must bind and join18 parent-terminal plus18 positive child-terminal records.',
        'No imports, collection, tests, child launches, probes or actual authority in this review.'],
    previous_failure_review_sha256=sha(failure.read_bytes()),candidate_imports=False,tests=False,launch_authority=False)
(OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'review':str(OUT/'review.json'),'sha256':sha((OUT/'review.json').read_bytes()),'source_count':38,'static_cases':18},sort_keys=True))
