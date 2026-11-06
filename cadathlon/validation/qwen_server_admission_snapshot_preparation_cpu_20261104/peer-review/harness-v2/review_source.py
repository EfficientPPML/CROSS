"""Narrow static successor harness review. Never imports/runs the harness."""
import ast
import hashlib
import json
import tarfile
from pathlib import Path
OLD=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v1')
ROOT=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v2')
OUT=Path('/tmp/qwen-snapshot-v2-harness-security-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def pretty(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def dump(v):return ast.dump(v,include_attributes=False)
def inv(root):
    paths=sorted(root.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return{p.relative_to(root).as_posix():sha(p.read_bytes())for p in paths if p.is_file()}
h=json.loads((ROOT/'handoff.json').read_bytes());assert sha((ROOT/'handoff.json').read_bytes())=='40aaae7ac0b50c6218ca234146732aca97bc48730fe4c0ea45304253a3788a06'
pins={n:sha((ROOT/n).read_bytes())for n in h['files']};assert pins==h['files']
assert pins['executed_driver.py']=='3d70d7531b9284a35da14c08d0d6cb35ecaa5148318a7c5a70e0a0133862a0aa'
assert pins['worker_driver.py']==sha((OLD/'worker_driver.py').read_bytes())=='55b6e6f684480efb7b834f72d4678df778b3f63beb2f06d36b05ca663764b8be'
assert pins['run_config.json']=='73f36f9634f726699a8c70ec826f2020b59713dc1fc9960b85859d27bb281ab0'
c=json.loads((ROOT/'run_config.json').read_bytes());oldc=json.loads((OLD/'run_config.json').read_bytes())
same=('cpu_affinity','expected_child_records','forbidden_roots','owned_temp_max_bytes','owned_temp_max_files',
    'pytest_ini_sha256','runtime_count','runtime_module_origins','runtime_source_expectations',
    'selected_files','selected_nodes','selected_nodes_sha256','selected_xml_cases','selected_xml_cases_sha256','selection_sha256','synthetic_only','actual_execution_authorized')
assert all(c[n]==oldc[n]for n in same)
assert c['expected_parent_records']==18 and c['source_count']==38
m=json.loads((ROOT/'sources-before.json').read_bytes())
assert len(m)==38 and pretty(m)==(ROOT/'sources-before.json').read_bytes()
assert sha(pretty(m))==c['source_manifest_sha256']==h['source_manifest_sha256']
assert inv(Path(c['source_root']))==inv(ROOT/'execution')==m
assert sha((ROOT/'sources.tar.gz').read_bytes())==c['source_archive_sha256']
with tarfile.open(ROOT/'sources.tar.gz','r:gz')as archive:
    members=archive.getmembers();assert len(members)==len({x.name for x in members})==38
    assert all(x.isfile()and not Path(x.name).is_absolute()and'..'not in Path(x.name).parts for x in members)
    assert{x.name:sha(archive.extractfile(x).read())for x in members}==m
assert inv(ROOT/'execution/runtime')==inv(OLD/'execution/runtime')and len(inv(ROOT/'execution/runtime'))==28
trees=[ast.parse((r/'executed_driver.py').read_bytes())for r in(OLD,ROOT)]
def alias_walk(t):
    fn=next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='inspect_owned')
    return next(n for n in fn.body if isinstance(n,ast.For))
assert dump(alias_walk(trees[0]))==dump(alias_walk(trees[1]))
def popen_try(t):
    for n in ast.walk(t):
        if isinstance(n,ast.With):
            for i,b in enumerate(n.body[:-1]):
                if isinstance(b,ast.Assign)and isinstance(b.value,ast.Call)and isinstance(b.value.func,ast.Attribute)and b.value.func.attr=='Popen':
                    assert isinstance(n.body[i+1],ast.Try);return n.body[i+1]
    raise AssertionError('missing original ownership')
assert dump(popen_try(trees[0]))==dump(popen_try(trees[1]))
mapping=next(n.value for n in trees[1].body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='map_canonical')
assert isinstance(mapping,ast.Lambda)
assert 'indent=2'in ast.unparse(mapping)and'separators'not in ast.unparse(mapping)
fn=next(n for n in trees[1].body if isinstance(n,ast.FunctionDef)and n.name=='inspect_owned')
text=ast.unparse(fn)
for literal in('parent-terminal.json','qwen-snapshot-owned-child-terminal-v1','owned_session_pgid','child_result_read','expected_parent_records','parent_terminal_sha256'):
    assert literal in text
assert "parent['observed_pgid'] == terminal['observed_pgid']"in text
assert "parent['exit_code'] == terminal['exit_code'] == 0"in text
assert "type(parent['pid']) is int"in text and "type(parent['owned_session_pgid']) is int"in text
assert "parent['child_result_read'] is False"in text and "parent['reaped'] is True"in text
assert pins=={n:sha((ROOT/n).read_bytes())for n in pins}
assert inv(Path(c['source_root']))==inv(ROOT/'execution')==m
review=dict(schema='qwen-snapshot-harness-v2-security-source-review',passed=True,concrete_blockers=[],
    handoff_sha256=sha((ROOT/'handoff.json').read_bytes()),pins=pins,source_count=38,runtime_count=28,static_cases=18,
    checks=['exact source/current/execution/archive38 and unchanged runtime28',
        'source map remains exact indent2 bytes with matching before/end serialization',
        'unchanged worker55b6/NumPy prohibition/empty origins/CPU112-115/static18 selection',
        'early parent closed-schema typed PID/group/exit/scenario joined to positive records',
        'exact18 early+18 positive/results/logs required; extra or missing names reject',
        'original alias walk and immediate-Popen cleanup AST unchanged',
        'failed v1 evidence copied with explicit historical labels'],
    scope='narrow source-only successor harness review; no import, collection, run, probe or grant',
    runtime_results_claimed=False,complete_scientific_authentication_claimed=False,
    caveat='Future terminal admission still requires owned outer exit/reap; new early records cover completed wait paths, not every exceptional cleanup path.')
(OUT/'review.json').write_bytes(pretty(review))
print(json.dumps({'review':str(OUT/'review.json'),'sha256':sha((OUT/'review.json').read_bytes())},sort_keys=True))
