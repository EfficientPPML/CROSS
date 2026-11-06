"""Stdlib source/tar/AST review only; never imports the candidate."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/tmp/qwen-server-snapshot-integration-tests-20261104')
HAND=Path('/tmp/qwen-server-snapshot-integration-test-handoff-20261104')
PARENT=Path('/tmp/qwen-server-snapshot-runtime-v2-20261104')
OUT=Path(__file__).parent
H=lambda raw:hashlib.sha256(raw).hexdigest()
def scan(root):
    result={}
    for p in root.rglob('*'):
        assert not p.is_symlink(),str(p)
        if p.is_file():result[p.relative_to(root).as_posix()]=H(p.read_bytes())
    return result
handoff_raw=(HAND/'handoff.json').read_bytes()
assert H(handoff_raw)=='848a375e24ee9b0244f48d1637d10df2e2ab398ffa22510465bb832ae8a19865'
handoff=json.loads(handoff_raw)
map_raw=(HAND/'sources.json').read_bytes()
assert H(map_raw)=='9fa73f2f59e6bef71a2de475a09e0aaccb25d56375f7a06452cac7b75e4afc1c'
expected=json.loads(map_raw)
assert map_raw==(json.dumps(expected,sort_keys=True,separators=(',',':'))+'\n').encode()
before=scan(ROOT);assert before==expected and len(expected)==404
parent=scan(PARENT)
assert len(parent)==391 and parent==json.loads((ROOT/'runtime_sources.json').read_bytes())
assert {n[8:]:v for n,v in expected.items()if n.startswith('runtime/')}==parent
assert H((HAND/'sources.tar.gz').read_bytes())=='bbc4c0bd3ce4bc6a1cd0efa10a7afc799e280887132869b846849fee10b2e713'
archived={}
with tarfile.open(HAND/'sources.tar.gz','r:gz')as tar:
    for member in tar.getmembers():
        assert member.isfile()and member.name not in archived
        assert not member.name.startswith('/')and '..'not in Path(member.name).parts
        with tar.extractfile(member)as stream:archived[member.name]=H(stream.read())
assert archived==expected
trees={n:ast.parse((ROOT/n).read_bytes(),filename=n)for n in handoff['test_hashes']}
assert all(expected[n]==v for n,v in handoff['test_hashes'].items())
assignment=next(n for n in trees['integration_cases.py'].body if isinstance(n,ast.Assign))
cases=ast.literal_eval(assignment.value)
assert type(cases)is tuple and len(cases)==len(set(cases))==36
nodes=['test_snapshot_integration.py::test_snapshot_integration['+c+']'for c in cases]
assert nodes==handoff['static_nodeids']
fn=next(n for n in trees['test_snapshot_integration.py'].body if isinstance(n,ast.FunctionDef)and n.name.startswith('test_'))
assert len(fn.decorator_list)==1
assert ast.unparse(fn.decorator_list[0])=="pytest.mark.parametrize('scenario', CASES, ids=CASES)"
def owned_try(tree):
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name.startswith('test_'))
    w=next(n for n in fn.body if isinstance(n,ast.With))
    assert isinstance(w.body[0],ast.Assign)and ast.unparse(w.body[0].value.func)=='subprocess.Popen'
    assert isinstance(w.body[1],ast.Try)
    return ast.dump(w.body[1],include_attributes=False)
old_test=Path('/tmp/qwen-device-component-tests-v2-20261104/test_device_component.py')
assert owned_try(trees['test_snapshot_integration.py'])==owned_try(ast.parse(old_test.read_bytes()))
entry=(ROOT/'runtime/qwen_server_entry.py').read_bytes();entry_tree=ast.parse(entry)
known=ast.parse("if __name__=='__main__':\n    raise SystemExit(main())\n").body[0]
dump=lambda node:ast.dump(node,include_attributes=False)
assert [i for i,n in enumerate(entry_tree.body)if dump(n)==dump(known)]==[len(entry_tree.body)-1]
provenance=json.loads((ROOT/'entry_fixture_provenance.json').read_bytes())
definitions=ast.Module(body=entry_tree.body[:-1],type_ignores=entry_tree.type_ignores)
assert provenance['entry_sha256']==H(entry)
assert provenance['excluded_dispatch_ast_sha256']==H(dump(known).encode())
assert provenance['definition_ast_sha256']==H(dump(definitions).encode())
assert provenance['genuine_bootstrap']is False
# Ordinary call shapes never override the private source-retention defaults.
private={'_dispatch','_require','_require_runtime','_check_runtime'}
assert not [(n.lineno,k.arg)for tree in trees.values()for n in ast.walk(tree)if isinstance(n,ast.Call)for k in n.keywords if k.arg in private]
after=scan(ROOT);assert after==before and scan(PARENT)==parent
report=dict(schema='qwen-server-snapshot-integration-test-source-peer-review-v1',
    conclusion='No concrete source blocker identified in the declared synthetic integration scope; execution remains unperformed.',
    source_root=str(ROOT),handoff_sha256=H(handoff_raw),source_manifest_sha256=H(map_raw),
    source_archive_sha256=handoff['source_archive_sha256'],source_count=404,parent_count=391,parent_exact=True,
    new_source_count=13,current_before_after_archive_equal=True,static_nodeids=nodes,
    selected_files=handoff['selected_files'],test_source_pins=handoff['test_hashes'],
    seams_sha256=expected['SEAMS.json'],scope_sha256=expected['TEST_SCOPE.md'],entry_fixture_provenance_sha256=expected['entry_fixture_provenance.json'],
    entry_definition_ast_and_single_excluded_dispatch_verified=True,owned_popen_try_ast_equals_device50=True,
    source_review=[
        'All six new Python sources read with original391 capture/control/backend/run/Finalizer paths. Actual local reader and composition module definitions retain genuine file/spec origins; only two constructor exports are recorded.',
        'Temporary source-owned numpy/jax ModuleType names have no forged source origins. Genuine discovery is denied. Constructor fixtures use the real Dependencies dataclass, keep ReadDirectory unconstructed, and fail if numerical engine callbacks run.',
        'The __main__ alias is an explicit definition-only entry fixture, excluding exactly the known terminal dispatch. Actual snapshot/origins are retained; no genuine bootstrap claim.',
        'Export, code, descriptor, metadata and state mutations use original retained call routes. Their sentinels must remain unused, and original runtime check must still reject after restoration. No private retention keyword is supplied.',
        'Clean hot checks compare finite full-validator/derivation counts; full boundary explicitly increments the full validator. Actual default capacity denial occurs after capture/derivation but before core install, evidence or key transition.',
        'Separate process recordings exercise actual run/Finalizer and attempted cleanup order. Body identity preservation, retained original owners, unretired guard on close-only failure and late exit70 are explicit. No false failed-status assertion for finalizer-only failure.',
        'Child ownership begins immediately after Popen with original nested cleanup/wait AST; early parent receipt precedes success assertions. Positive results require exact PID/PGID and main/export/profiler/temporary-name/source restoration.'],
    limitations=[
        'Source review is not collection or execution evidence and cannot establish that all intended branches will pass.',
        'Packet/phase/readiness validation and positive process admission/audit/capacity/backend/stage/output/signal operations use disclosed recordings; no real scientific/model/key authority or original reader/body execution is inferred.',
        'Terminal tests cover the real terminal_status function boundary, not actual server CLI bootstrap or native process initialization.',
        'The child audit excludes probes/protected body opens and genuine import discovery but is not claimed as an arbitrary-Python/native syscall sandbox.'],
    candidate_imports=False,collection=False,tests=False,backend_or_capacity_probes=False,borrowed_body_reads=False,actual_authority=False)
(OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'review_sha256':H((OUT/'review.json').read_bytes()),'source_count':404,'static_count':36,'parent_exact':True},sort_keys=True))
