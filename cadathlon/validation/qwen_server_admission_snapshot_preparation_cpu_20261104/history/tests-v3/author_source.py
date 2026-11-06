"""Source-only successor construction. Does not import or execute candidate code."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import tarfile

OLD=Path('/tmp/qwen-server-admission-snapshot-tests-v2-20261104')
NEW=Path('/tmp/qwen-server-admission-snapshot-tests-v3-20261104')
OUT=Path(__file__).resolve().parent
OLDMAP=Path('/tmp/qwen-server-admission-snapshot-test-sources-v2-20261104.json')
NEWMAP=Path('/tmp/qwen-server-admission-snapshot-test-sources-v3-20261104.json')
FAIL=Path('/tmp/qwen-server-admission-snapshot-root-review-20261104/failed-terminal-review.json')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def encode(value):return (json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
before=json.loads(OLDMAP.read_bytes())
assert len(before)==37
assert sha(OLDMAP.read_bytes())=='478b3411cfa54d154eb91afbb1bccab5b6ae24520c4b6f17873be9bd0d8d44e0'
assert sha(FAIL.read_bytes())=='fc08e491aa93ae6eb0961d7c0d008efd0b2cf83b00f9b47d5a06ef73b376eefd'
NEW.mkdir()
for name,digest in before.items():
    src=OLD/name;assert not src.is_symlink()
    raw=src.read_bytes();assert sha(raw)==digest
    dest=NEW/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)

child=(NEW/'snapshot_child.py').read_text()
old_line="    public=file_module.PublicContract((('synthetic-public-marker',b'synthetic non-proof metadata'),))\n"
new_lines="""    # Explicitly unadmitted synthetic fixture: PublicContract's constructor
    # authenticates real public blobs. Bypass only that constructor, without
    # replacing another validator or claiming these marker bytes are evidence.
    public=object.__new__(file_module.PublicContract)
    object.__setattr__(public,'blobs',(('synthetic-public-marker',b'synthetic non-proof metadata'),))
"""
assert child.count(old_line)==1
child=child.replace(old_line,new_lines)
assert child.replace(new_lines,old_line)==(OLD/'snapshot_child.py').read_text()
(NEW/'snapshot_child.py').write_text(child)

test=(NEW/'test_snapshot_guard.py').read_text()
anchor="    assert code==0,(scenario,(tmp_path/'child.log').read_text()[-8192:])\n"
early="""    # Publish owned process closure before exit/result assertions. A startup
    # failure can have no result.json; its PID/group/exit evidence still survives.
    try:os.killpg(proc.pid,0)
    except ProcessLookupError:owned_group_absent=True
    else:owned_group_absent=False
    with(tmp_path/'parent-terminal.json').open('x')as stream:
        json.dump({'schema':'qwen-snapshot-owned-child-terminal-v1','pid':proc.pid,
            'owned_session_pgid':proc.pid,'observed_pgid':observed_pgid,
            'exit_code':code,'reaped':True,'group_absent':owned_group_absent,
            'scenario':scenario,'child_result_read':False,
            'runtime_scope':'recording guard state only'},stream,sort_keys=True)
        stream.write('\\n')
    assert owned_group_absent
"""
assert test.count(anchor)==1
test=test.replace(anchor,early+anchor)
assert test.replace(early,'')==(OLD/'test_snapshot_guard.py').read_text()
(NEW/'test_snapshot_guard.py').write_text(test)

scope=(NEW/'TEST_SCOPE.md').read_text()
scope=scope.replace('The child constructs the exact13 frozen admission types with non-proof synthetic\nmarkers, including all25 proof names.',
'''The child constructs the exact13 frozen admission types with non-proof synthetic
markers, including all25 proof names. PublicContract alone is allocated with
object.__new__ and receives its blobs with object.__setattr__, explicitly bypassing
its real source-validating constructor. This is unadmitted fixture construction,
not an additional validator replacement.''')
scope=scope.replace('owned pytest temporary files. The parent never imports candidate modules.',
'''owned pytest temporary files. After direct wait and group observation, the parent
publishes parent-terminal.json before asserting normal exit or reading result.json.
That record separates the owned session group (start_new_session, equal to Popen
PID) from an optional actual getpgid observation. It survives a completed nonzero
child exit even when no result exists; it does not claim every exceptional cleanup
path publishes a receipt. The existing positive child-terminal.json and all guard
assertions are unchanged. The parent never imports candidate modules.''')
scope=scope.replace('and retain the18 child-terminal records.',
'''and retain and validate both the18 parent-terminal records and the18 positive
child-terminal records, joining their exact PID/group/exit/scenario fields. It must
not silently ignore the new early receipts. Retain owned pytest current aliases
without following them or copying duplicate child files.''')
scope+='''
The prior37-source run failed all18 cases in PublicContract.__post_init__, before
the recording-validator seam or snapshot guard. Its collector then rejected the
missing child-terminal records. That original run and source are preserved; it
does not establish a snapshot runtime failure. This v3 has not been imported,
collected or run, and no new launch authority is implied.
'''
(NEW/'TEST_SCOPE.md').write_text(scope)

selected=json.loads((NEW/'selected_tests.json').read_bytes())
parsed=ast.parse(test)
cases=ast.literal_eval(next(n.value for n in parsed.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='CASES'for t in n.targets)))
assert len(cases)==18
assert selected['static_nodeids']==['test_snapshot_guard.py::test_snapshot_guard['+n+']'for n in cases]
for path in NEW.rglob('*.py'):ast.parse(path.read_bytes(),filename=str(path))
overlays={n:{'before':before[n],'after':sha((NEW/n).read_bytes())}for n in before if sha((NEW/n).read_bytes())!=before[n]}
assert set(overlays)=={'snapshot_child.py','test_snapshot_guard.py','TEST_SCOPE.md'}
history=dict(schema='qwen-snapshot-test-source-successor-v3',parent_source_root=str(OLD),parent_source_manifest_sha256=sha(OLDMAP.read_bytes()),
    parent37_preserved=True,runtime28_unchanged=True,static_cases=18,overlays=overlays,
    prior_failed_root_review_path=str(FAIL),prior_failed_root_review_sha256=sha(FAIL.read_bytes()),
    prior_outcome='18 fixture-construction failures before recording seam/guard; collector missing child terminals',
    synthetic_constructor_bypass='Only PublicContract via object.__new__/object.__setattr__; no additional validator patch',
    guard_assertions_unchanged=True,positive_child_terminal_assertions_unchanged=True,
    early_parent_terminal='After successful direct wait and group observation, before exit/result assertions; records owned versus observed PGID separately',
    prospective_harness_obligation='Retain and validate18 parent-terminal records plus18 positive child-terminal records; no execution granted',
    candidate_imported=False,collection=False,tests_executed=False,actual_proofs=False,execution_authorized=False)
(NEW/'successor_history_v3.json').write_bytes(encode(history))
after={str(p.relative_to(NEW)):sha(p.read_bytes())for p in sorted(NEW.rglob('*'))if p.is_file()}
assert len(after)==38
assert all(after[n]==d for n,d in before.items()if n.startswith('runtime/'))
assert sum(n.startswith('runtime/')for n in after)==28
assert all(sha((OLD/n).read_bytes())==d for n,d in before.items())
NEWMAP.write_bytes(encode(after));(OUT/'sources.json').write_bytes(NEWMAP.read_bytes())
patch=''.join(''.join(difflib.unified_diff((OLD/n).read_text().splitlines(True),(NEW/n).read_text().splitlines(True),fromfile='v2/'+n,tofile='v3/'+n))for n in sorted(overlays))
(OUT/'source.patch').write_text(patch)
archive=OUT/'sources.tar.gz'
with tarfile.open(archive,'w:gz')as tar:
    for n in sorted(after):tar.add(NEW/n,arcname=n,recursive=False)
with tarfile.open(archive)as tar:
    assert {m.name:sha(tar.extractfile(m).read())for m in tar.getmembers()}==after
handoff=dict(schema='qwen-snapshot-tests-source-handoff-v3',source_root=str(NEW),source_manifest=str(NEWMAP),
    source_manifest_sha256=sha(NEWMAP.read_bytes()),source_archive_sha256=sha(archive.read_bytes()),source_count=38,
    source_hashes=after,static_count=18,static_nodeids=selected['static_nodeids'],selected_files=selected['selected_files'],
    runtime28_unchanged=True,parent37_preserved=True,parent_source_manifest_sha256=sha(OLDMAP.read_bytes()),
    child_source_sha256=after['snapshot_child.py'],test_source_sha256=after['test_snapshot_guard.py'],patch_sha256=sha(patch.encode()),
    prior_failed_root_review_sha256=sha(FAIL.read_bytes()),candidate_imported=False,collection=False,tests_executed=False,actual_proofs=False,execution_authorized=False,
    early_parent_receipts_required=18,positive_child_receipts_required=18)
(OUT/'handoff.json').write_bytes(encode(handoff))
print(json.dumps({k:v for k,v in handoff.items()if k not in('source_hashes','static_nodeids')},sort_keys=True))
print('handoff_sha256='+sha((OUT/'handoff.json').read_bytes()))
