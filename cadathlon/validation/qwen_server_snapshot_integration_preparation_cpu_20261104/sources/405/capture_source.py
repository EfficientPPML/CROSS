"""Freeze a test-only forwarding correction; no candidate imports or tests."""
import ast
import difflib
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

H=Path(__file__).resolve().parent
N=Path('/tmp/qwen-server-snapshot-integration-tests-v2-20261104')
P=Path('/tmp/qwen-server-snapshot-integration-tests-20261104')
PH=Path('/tmp/qwen-server-snapshot-integration-test-handoff-20261104')
PROOF=Path('/tmp/qwen-server-snapshot-integration-focused-proof-20261104/preparation-v1')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def canonical(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def inventory(root):
    raw={}
    for p in sorted(root.rglob('*')):
        assert not p.is_symlink()
        if p.is_file():raw[p.relative_to(root).as_posix()]=p.read_bytes()
    return raw
def mapping(raw):return{n:sha(b)for n,b in raw.items()}
def tar_map(path):
    with tarfile.open(path,'r:gz')as t:
        rows=t.getmembers();assert all(m.isfile()for m in rows)and len(rows)==len({m.name for m in rows})
        return {m.name:sha(t.extractfile(m).read())for m in rows}
old=json.loads((PH/'sources.json').read_bytes());parent=inventory(P);current=inventory(N)
assert len(old)==404 and old==mapping(parent)==tar_map(PH/'sources.tar.gz')
assert sha((PH/'sources.json').read_bytes())=='9fa73f2f59e6bef71a2de475a09e0aaccb25d56375f7a06452cac7b75e4afc1c'
assert set(current)==set(old)and{n for n in old if sha(current[n])!=old[n]}=={'integration_fixture.py'}
before_tree=ast.parse(parent['integration_fixture.py']);after_tree=ast.parse(current['integration_fixture.py'])
def cls(tree):return next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Fixture')
before={n.name:n for n in cls(before_tree).body if isinstance(n,ast.FunctionDef)}
after={n.name:n for n in cls(after_tree).body if isinstance(n,ast.FunctionDef)}
dump=lambda n:ast.dump(n,include_attributes=False)
assert set(before)==set(after)
assert[k for k in before if dump(before[k])!=dump(after[k])]==['capture_runtime']
body=after['capture_runtime'].body
added=[n for n in body if isinstance(n,ast.FunctionDef)]
assert[n.name for n in added]==['source_check','capacity_check']
assert ast.unparse(added[0])=='def source_check():\n    return self.full_boundary()'
assert ast.unparse(added[1])=='def capacity_check():\n    return self.capacity()'
class Normalize(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        if node.name in('source_check','capacity_check'):return None
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute)and node.func.attr=='_runtime':
            assert isinstance(node.args[4],ast.Name)and node.args[4].id=='source_check'
            assert isinstance(node.args[5],ast.Name)and node.args[5].id=='capacity_check'
            node.args[4]=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='full_boundary',ctx=ast.Load())
            node.args[5]=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='capacity',ctx=ast.Load())
        return node
assert dump(Normalize().visit(after_tree))==dump(before_tree)
runtime=json.loads(current['runtime_sources.json']);assert len(runtime)==391
assert {n[8:]:sha(b)for n,b in current.items()if n.startswith('runtime/')}==runtime
assert runtime==mapping(inventory(Path('/tmp/qwen-server-snapshot-runtime-v2-20261104')))
selection=json.loads(current['selected_tests.json']);assert selection['static_count']==36
xml=ET.fromstring((PROOF/'tests.xml').read_bytes());rows=list(xml.iter('testcase'))
failed=[r.attrib['name']for r in rows if r.find('failure')is not None]
passed=[r.attrib['name']for r in rows if all(r.find(t)is None for t in('failure','error','skipped'))]
assert len(rows)==36 and len(failed)==25 and len(passed)==11
assert all(not any(part in n for part in('[process_','[terminal_'))for n in failed)
assert sum('[process_'in n for n in passed)==6 and sum('[terminal_'in n for n in passed)==5
history=dict(schema='qwen-snapshot-integration-plain-callback-source-successor-v2',
    predecessor404=dict(root=str(P),source_map_sha256=sha((PH/'sources.json').read_bytes()),source_archive_sha256=sha((PH/'sources.tar.gz').read_bytes()),handoff_sha256=sha((PH/'handoff.json').read_bytes())),
    preserved_failed_run=dict(path=str(PROOF),xml_sha256=sha((PROOF/'tests.xml').read_bytes()),tests_log_sha256=sha((PROOF/'tests.log').read_bytes()),
        tests=36,failures=25,passed=11,passed_process_cases=6,passed_terminal_cases=5,
        owner_authority='Root-reported direct execution; this author did not poll or reap its session.',owner_session=31389,launch_tool='4c43f9',terminal_tool='915588',reported_exit_code=1,reported_reaped=True,
        failure='Direct fixture passed self.full_boundary/self.capacity bound methods to runtime _function_binding, which requires FunctionType. Actual process passes nested plain functions.',
        report_absent_reason='Outer inspector rejected missing positive child records after failed tests; raw XML/log/owner artifacts remain preserved.'),
    correction='capture_runtime defines two local plain functions forwarding to the same fixture methods, then passes them to unchanged runtime._runtime.',
    code_overlay=dict(path='integration_fixture.py',before=old['integration_fixture.py'],after=sha(current['integration_fixture.py'])),
    all_runtime391_exact=True,all36_assertion_nodes_unchanged=True,FunctionType_requirement_unchanged=True,
    earlier_source_reviews_remain_source_only=True,source_scope='Unrun test-only correction; not a runtime repair or a new PASS.',
    imports=False,collection=False,execution=False,actual_authorization=False)
(N/'fixture_successor_history.json').write_bytes(canonical(history))
current=inventory(N);m=mapping(current);assert len(m)==405
assert sum(m[n]==old[n]for n in old)==403 and set(m)-set(old)=={'fixture_successor_history.json'}
for name in('integration_checks.py','integration_cases.py','integration_child.py','test_snapshot_integration.py','conftest.py','selected_tests.json','runtime_sources.json','SEAMS.json','entry_fixture_provenance.json'):
    assert m[name]==old[name]
(H/'sources.json').write_bytes(canonical(m))
patch=''.join(difflib.unified_diff(parent['integration_fixture.py'].decode().splitlines(True),current['integration_fixture.py'].decode().splitlines(True),fromfile='failed404/integration_fixture.py',tofile='unrun405/integration_fixture.py'))
(H/'source.patch').write_text(patch)
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w',format=tarfile.PAX_FORMAT)as t:
    for name,raw in current.items():
        info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.mtime=0;info.uid=info.gid=0;info.uname=info.gname='';t.addfile(info,io.BytesIO(raw))
(H/'sources.tar.gz').write_bytes(gzip.compress(buf.getvalue(),mtime=0))
assert tar_map(H/'sources.tar.gz')==m==mapping(inventory(N))and mapping(inventory(P))==old
handoff=dict(schema='qwen-snapshot-integration-held-test-handoff-v2',root=str(N),source_count=405,runtime_count=391,
    source_manifest_sha256=sha((H/'sources.json').read_bytes()),source_archive_sha256=sha((H/'sources.tar.gz').read_bytes()),
    source_archive_bytes=(H/'sources.tar.gz').stat().st_size,source_patch_sha256=sha((H/'source.patch').read_bytes()),
    static_count=36,static_nodeids=selection['static_nodeids'],selected_files=selection['selected_files'],
    fixture_sha256=m['integration_fixture.py'],history_sha256=m['fixture_successor_history.json'],
    parent404_preserved=True,runtime391_exact=True,assertion_nodes_unchanged=True,normalized_fixture_AST_unchanged=True,
    imports=False,collected=False,executed=False,actual_authority=False,source_capture_script_sha256=sha(Path(__file__).read_bytes()))
(H/'handoff.json').write_bytes(canonical(handoff))
print(json.dumps(dict(passed=True,handoff_sha256=sha((H/'handoff.json').read_bytes()),**handoff),sort_keys=True))
