"""Narrow405 source delta review only; no candidate execution."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile
P=Path('/tmp/qwen-server-snapshot-integration-test-handoff-v2-20261104')
R=Path('/tmp/qwen-server-snapshot-integration-tests-v2-20261104')
OLD=Path('/tmp/qwen-server-snapshot-integration-tests-20261104')
H=lambda raw:hashlib.sha256(raw).hexdigest()
def scan(root):
    result={}
    for path in root.rglob('*'):
        assert not path.is_symlink()
        if path.is_file():result[path.relative_to(root).as_posix()]=H(path.read_bytes())
    return result
h=json.loads((P/'handoff.json').read_bytes());m=json.loads((P/'sources.json').read_bytes())
assert H((P/'handoff.json').read_bytes())=='57498a8ea7a0b03d75f4088d173434ec4bca218094e982ed5531c183c748ffa5'
assert H((P/'sources.json').read_bytes())=='7c4f95e68dabba24ee489bfe05ddd77b5de0ca5e7ba1a3e77139ef57a410bbd4'
assert H((P/'sources.tar.gz').read_bytes())=='7814e934402820f57da5084c6cefc4d0a924cc5e68a06c219bdeacd47d03ee59'
oldmap=json.loads(Path('/tmp/qwen-server-snapshot-integration-test-handoff-20261104/sources.json').read_bytes())
assert scan(OLD)==oldmap and scan(R)==m and len(m)==405
assert set(oldmap)-set(m)==set()and set(m)-set(oldmap)=={'fixture_successor_history.json'}
assert [n for n in oldmap if oldmap[n]!=m[n]]==['integration_fixture.py']
assert len([n for n in m if n.startswith('runtime/')])==391
with tarfile.open(P/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==len({x.name for x in members})==405
    assert all(x.isfile()for x in members)
    assert {x.name:H(archive.extractfile(x).read())for x in members}==m
oldtree=ast.parse((OLD/'integration_fixture.py').read_bytes())
newtree=ast.parse((R/'integration_fixture.py').read_bytes())
fn=next(x for x in ast.walk(newtree)if isinstance(x,ast.FunctionDef)and x.name=='capture_runtime')
delegates=[x for x in fn.body if isinstance(x,ast.FunctionDef)]
assert len(delegates)==2
dump=lambda x:ast.dump(x,include_attributes=False)
for delegate,name,target in zip(delegates,('source_check','capacity_check'),('full_boundary','capacity'),strict=True):
    assert dump(delegate)==dump(ast.parse('def '+name+'():return self.'+target+'()').body[0])
call=next(x for x in ast.walk(fn)if isinstance(x,ast.Call)and ast.unparse(x.func)=='self.process._runtime')
assert ast.unparse(call.args[4])=='source_check'and ast.unparse(call.args[5])=='capacity_check'
call.args[4]=ast.parse('self.full_boundary',mode='eval').body
call.args[5]=ast.parse('self.capacity',mode='eval').body
fn.body=[x for x in fn.body if x not in delegates]
assert dump(oldtree)==dump(newtree)
oldh=json.loads(Path('/tmp/qwen-server-snapshot-integration-test-handoff-20261104/handoff.json').read_bytes())
assert h['static_nodeids']==oldh['static_nodeids']and len(h['static_nodeids'])==36
history=json.loads((R/'fixture_successor_history.json').read_bytes())
for name,digest in h['root_failed_receipts'].items():assert H(Path(name).read_bytes())==digest
pre=P/'before-root-failure-receipts'
premap=json.loads((pre/'sources.json').read_bytes())
assert len(premap)==405 and [n for n in premap if premap[n]!=m[n]]==['fixture_successor_history.json']
assert H((pre/'sources.json').read_bytes())==history['pre_receipt_source_only_checkpoint']['source_map_sha256']
assert H((pre/'sources.tar.gz').read_bytes())==history['pre_receipt_source_only_checkpoint']['source_archive_sha256']
assert scan(R)==m and scan(OLD)==oldmap
report=dict(schema='qwen-snapshot-integration-405-delta-source-peer-review-v1',
    conclusion='The two plain local delegates satisfy the unchanged exact FunctionType callback interface by construction; no additional concrete blocker found in this narrow unrun source delta.',
    source_root=str(R),handoff_sha256=H((P/'handoff.json').read_bytes()),source_manifest_sha256=H((P/'sources.json').read_bytes()),source_archive_sha256=H((P/'sources.tar.gz').read_bytes()),
    source_count=405,parent404_preserved=True,runtime391_exact=True,all36_assertion_sources_unchanged=True,
    only_code_overlay='integration_fixture.py',fixture_sha256=m['integration_fixture.py'],
    addition='fixture_successor_history.json',history_sha256=m['fixture_successor_history.json'],
    normalized_fixture_AST_equals_failed404=True,delegates=['def source_check(): return self.full_boundary()','def capacity_check(): return self.capacity()'],
    interpretation='No runtime requirements, validation exports, mutation assertions, process cases, entry provenance or owner cleanup change. The fixture now matches the actual process nested-function call category; this is not a runtime repair or a new test result.',
    pre_receipt405_checkpoint_preserved=True,original_failed36_retained=True,root_failed_receipt_pins=h['root_failed_receipts'],
    previous_static_review_limitation='The original source review missed the bound-method callback mismatch; its receipt is historical. Failed run25/11 remains separately retained and no PASS is transferred.',
    candidate_imports=False,collection=False,tests=False,probes=False,borrowed_body_reads=False,actual_authority=False)
out=Path(__file__).parent/'source-v2-review.json';out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'review_sha256':H(out.read_bytes()),'source_count':405,'runtime_exact':391,'static_cases':36},sort_keys=True))
