"""Source-only successor: observable sticky failure after fixture restoration."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import tarfile

OLD=Path('/tmp/qwen-device-component-tests-20261104')
NEW=Path('/tmp/qwen-device-component-tests-v2-20261104')
OUT=Path(__file__).resolve().parent
OLDMAP=Path('/tmp/qwen-device-component-test-sources-v1-20261104.json')
NEWMAP=Path('/tmp/qwen-device-component-test-sources-v2-20261104.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
encode=lambda v:(json.dumps(v,sort_keys=True,indent=2)+'\n').encode()
before=json.loads(OLDMAP.read_bytes())
assert len(before)==49 and sha(OLDMAP.read_bytes())=='39ad4b3709425d86626473d38d7b30df84814019fc4794b2034c31daceb4715a'
NEW.mkdir()
for n,d in before.items():
    raw=(OLD/n).read_bytes();assert sha(raw)==d
    p=NEW/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
old_text=(OLD/'device_child.py').read_text()
anchor="""                before=len(state['log']);raises(PermissionError,lambda:capacity.issue_initialization(policy))
                assert len(state['log'])==before and row['state']=='failed'
"""
replacement="""                # Restore the original identities and benign recording behavior
                # before testing public stickiness, not merely the private flag
                # or global one-use issuance rule. Never reset guard state.
                while restores:restores.pop()()
                state['mode']='benign'
                if RECORDING_JAX is not None:
                    RECORDING_JAX.STATS={'bytes_in_use':90,'bytes_limit':100,'peak_bytes_in_use':90}
                assert policy.__dict__ is original_dict
                assert all(policy.__dict__[name]is value for name,value in zip(('phase','blobs','expected_pins'),original_fields,strict=True))
                before=(len(state['log']),state['initializations'],0 if RECORDING_JAX is None else RECORDING_JAX.CALLS)
                original_permit=row['permit']
                if row['device']is None:
                    raises(PermissionError,lambda:capacity.initialize_once(original_permit))
                else:
                    next_index=row['stage_index']+1
                    assert 0<=next_index<len(fixtures.STAGES)
                    raises(PermissionError,lambda:capacity.observe_capacity(original_permit,fixtures.STAGES[next_index]))
                assert (len(state['log']),state['initializations'],0 if RECORDING_JAX is None else RECORDING_JAX.CALLS)==before
                assert capacity._REGISTRY[id(original_permit)]is row and row['state']=='failed'
                raises(PermissionError,lambda:capacity.issue_initialization(policy))
                assert (len(state['log']),state['initializations'],0 if RECORDING_JAX is None else RECORDING_JAX.CALLS)==before
                assert row['state']=='failed'
"""
assert old_text.count(anchor)==1
text=old_text.replace(anchor,replacement)
assert text.replace(replacement,anchor)==old_text
(NEW/'device_child.py').write_text(text)
scope=(OLD/'TEST_SCOPE.md').read_text()
scope+='''
The v2 test-source successor strengthens existing failed-state assertions without
adding cases. The central retained_failed helper now runs all identity restorers,
sets recording hooks and statistics to benign values, and attempts the original
permit's next appropriate public transition. Before device admission this is
initialize_once; after admission it is observe_capacity at the next valid ordered
stage. Rejection must leave hook count, initializer count and memory-sample count
unchanged. The original retained dictionary/three fields and failed registry row
are checked again. Reissuance remains a separate one-use check, not the sole
observable evidence of sticky failure. No runtime or registry state is reset.
The exact49 predecessor and its unrun handoff remain preserved.
'''
(NEW/'TEST_SCOPE.md').write_text(scope)
for p in NEW.rglob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
overlays={n:dict(before=d,after=sha((NEW/n).read_bytes()))for n,d in before.items()if sha((NEW/n).read_bytes())!=d}
assert set(overlays)=={'device_child.py','TEST_SCOPE.md'}
history=dict(schema='qwen-device-test-sticky-successor-v2',parent_root=str(OLD),parent_source_map_sha256=sha(OLDMAP.read_bytes()),
    parent_handoff='/tmp/qwen-device-component-test-handoff-20261104/handoff.json',
    parent_handoff_sha256=sha(Path('/tmp/qwen-device-component-test-handoff-20261104/handoff.json').read_bytes()),
    parent49_preserved=True,runtime34_exact=True,device6_exact=True,selected30_unchanged=True,overlays=overlays,
    change='Restore identity mutations and benign recording behavior, then reject an appropriate transition on original permit without hook/init/sample work; keep separate reissuance check.',
    runtime_state_reset=False,candidate_imports=False,collection=False,execution=False,actual_authority=False)
(NEW/'successor_history.json').write_bytes(encode(history))
after={p.relative_to(NEW).as_posix():sha(p.read_bytes())for p in sorted(NEW.rglob('*'))if p.is_file()}
assert len(after)==50
assert all(after[n]==d for n,d in before.items()if n.startswith('runtime/')or n.startswith('device_metadata/'))
assert after['selected_tests.json']==before['selected_tests.json']
assert all(sha((OLD/n).read_bytes())==d for n,d in before.items())
NEWMAP.write_bytes(encode(after));(OUT/'sources.json').write_bytes(NEWMAP.read_bytes())
patch=''.join(''.join(difflib.unified_diff((OLD/n).read_text().splitlines(True),(NEW/n).read_text().splitlines(True),fromfile='v1/'+n,tofile='v2/'+n))for n in sorted(overlays))
(OUT/'source.patch').write_text(patch)
with tarfile.open(OUT/'sources.tar.gz','w:gz')as tar:
    for n in sorted(after):tar.add(NEW/n,arcname=n,recursive=False)
with tarfile.open(OUT/'sources.tar.gz')as tar:
    assert{m.name:sha(tar.extractfile(m).read())for m in tar.getmembers()}==after
selected=json.loads((NEW/'selected_tests.json').read_bytes())
handoff=dict(schema='qwen-device-component-focused-test-source-handoff-v2',source_root=str(NEW),source_count=50,
    source_manifest=str(NEWMAP),source_manifest_sha256=sha(NEWMAP.read_bytes()),source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
    child_sha256=after['device_child.py'],parent49_preserved=True,runtime34_exact=True,device6_exact=True,static_count=30,
    selected_files=selected['selected_files'],static_nodeids=selected['static_nodeids'],source_hashes=after,
    patch_sha256=sha(patch.encode()),candidate_imported=False,collected=False,executed=False,actual_grants=False,
    held_for_source_review_only=True)
(OUT/'handoff.json').write_bytes(encode(handoff))
print(json.dumps({k:v for k,v in handoff.items()if k not in('source_hashes','static_nodeids')},sort_keys=True))
print('handoff_sha256='+sha((OUT/'handoff.json').read_bytes()))
