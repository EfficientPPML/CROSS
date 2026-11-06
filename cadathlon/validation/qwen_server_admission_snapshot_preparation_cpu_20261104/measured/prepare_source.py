"""Author an unrun focused harness; stdlib source/evidence work only."""
import ast
import difflib
import hashlib
import io
import json
from pathlib import Path
import tarfile

OUT=Path(__file__).resolve().parent
OLD=OUT.parent/'preparation-v1'
SOURCE=Path('/tmp/qwen-server-admission-snapshot-tests-v3-20261104')
HAND=Path('/tmp/qwen-server-admission-snapshot-test-handoff-v3-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
encode=lambda x:(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
mapping_raw=(HAND/'sources.json').read_bytes();mapping=json.loads(mapping_raw)
assert sha(mapping_raw)=='963b03f0bdc4ca2f09fd03f4986c175d2a3b41b6fe82cddf42761c3b3a53c93b'
assert len(mapping)==38 and mapping_raw==encode(mapping)
archive_raw=(HAND/'sources.tar.gz').read_bytes()
assert sha(archive_raw)=='bb1eb2fb5613347f5ff2ecf9a27a1f3cdb5944261e154841ea3b19ab4ed75575'
old_driver=(OLD/'executed_driver.py').read_bytes();old_worker=(OLD/'worker_driver.py').read_bytes()
assert sha(old_driver)=='c436247368deed6d0794040e51f7044ea54de7fc354bb70645cc0116512d37d6'
assert sha(old_worker)=='55b6e6f684480efb7b834f72d4678df778b3f63beb2f06d36b05ca663764b8be'
for name,raw in [('before_executed_driver.py',old_driver),('before_worker_driver.py',old_worker),('sources-before.json',mapping_raw),('sources.tar.gz',archive_raw),('source-handoff.json',(HAND/'handoff.json').read_bytes())]:
    with(OUT/name).open('xb')as stream:stream.write(raw)
execution=OUT/'execution';execution.mkdir()
for name,digest in mapping.items():
    src=SOURCE/name;assert not src.is_symlink();raw=src.read_bytes();assert sha(raw)==digest
    p=execution/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
with tarfile.open(fileobj=io.BytesIO(archive_raw),mode='r:gz')as tar:
    assert{m.name:sha(tar.extractfile(m).read())for m in tar.getmembers()}==mapping

config=json.loads((OLD/'run_config.json').read_bytes())
config.update(schema='qwen-admission-snapshot-focused-proof-v2',source_count=38,source_root=str(SOURCE),
    proof_root=str(OUT),owned_tmpdir=str(OUT/'tmp'),pytest_basetemp=str(OUT/'pytest-owned'),
    source_manifest_encoding='canonical sorted indent=2 JSON plus newline',source_manifest_sha256=sha(mapping_raw),
    source_archive_sha256=sha(archive_raw),expected_parent_records=18,
    scope='Exactly18 fresh-child snapshot guard-state cases with exact13 types, explicit unadmitted PublicContract constructor bypass and recording full-validator seam. Validate18 early owned parent receipts plus18 positive child receipts/results/logs. No NumPy/backend, genuine scientific proof, actual role, capacity or HE/model execution.')
(OUT/'run_config.json').write_bytes(encode(config))
(OUT/'worker_driver.py').write_bytes(old_worker)
driver=old_driver.decode()
driver=driver.replace("CONFIG_SHA256 = '02d9dec6cfff9dd65b9c32f89fda52a08c47848e33bbab74237b80395b28e47d'",'CONFIG_SHA256 = '+repr(sha(encode(config))))
driver=driver.replace("map_canonical = lambda value: (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\\n').encode()", "map_canonical = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\\n').encode()")
driver=driver.replace("assert config['schema'] == 'qwen-admission-snapshot-focused-proof-v1'", "assert config['schema'] == 'qwen-admission-snapshot-focused-proof-v2'")
old="""        result_path, log_path = path.parent / 'result.json', path.parent / 'child.log'
        assert result_path.is_file() and log_path.is_file()
        result = json.loads(result_path.read_bytes())
"""
new="""        result_path, log_path = path.parent / 'result.json', path.parent / 'child.log'
        parent_path = path.parent / 'parent-terminal.json'
        assert all(p in regular for p in (result_path, log_path, parent_path))
        result = json.loads(result_path.read_bytes())
        parent = json.loads(parent_path.read_bytes())
        assert type(parent) is dict and set(parent) == {'schema', 'pid', 'owned_session_pgid', 'observed_pgid', 'exit_code', 'reaped', 'group_absent', 'scenario', 'child_result_read', 'runtime_scope'}
        assert parent['schema'] == 'qwen-snapshot-owned-child-terminal-v1'
        assert parent['child_result_read'] is False and parent['runtime_scope'] == 'recording guard state only'
"""
assert driver.count(old)==1;driver=driver.replace(old,new)
old="""        assert terminal['reaped'] is True and terminal['group_absent'] is True
"""
new=old+"""        assert parent['scenario'] == scenario
        assert type(parent['pid']) is int and parent['pid'] == pid
        assert type(parent['owned_session_pgid']) is int and parent['owned_session_pgid'] == pid
        assert parent['observed_pgid'] is None or (type(parent['observed_pgid']) is int and parent['observed_pgid'] == pid)
        assert parent['observed_pgid'] == terminal['observed_pgid']
        assert type(parent['exit_code']) is int and parent['exit_code'] == terminal['exit_code'] == 0
        assert parent['reaped'] is True and parent['group_absent'] is True
"""
assert driver.count(old)==1;driver=driver.replace(old,new)
old="""        records[scenario] = dict(terminal=terminal, result=result,
            terminal_path=path.relative_to(out).as_posix(), result_path=result_path.relative_to(out).as_posix(),
"""
new="""        records[scenario] = dict(terminal=terminal, result=result, parent_terminal=parent,
            parent_terminal_path=parent_path.relative_to(out).as_posix(), parent_terminal_sha256=sha(parent_path.read_bytes()),
            terminal_path=path.relative_to(out).as_posix(), result_path=result_path.relative_to(out).as_posix(),
"""
assert driver.count(old)==1;driver=driver.replace(old,new)
old="""    assert set(records) == set(expected) and len(records) == 18
    assert sum(p.name == 'result.json' for p in regular) == sum(p.name == 'child.log' for p in regular) == 18
"""
new="""    assert set(records) == set(expected) and len(records) == 18
    assert type(config['expected_parent_records']) is int and config['expected_parent_records'] == 18
    assert sum(p.name == 'parent-terminal.json' for p in regular) == config['expected_parent_records']
    assert sum(p.name == 'child-terminal.json' for p in regular) == 18
    assert sum(p.name == 'result.json' for p in regular) == sum(p.name == 'child.log' for p in regular) == 18
"""
assert driver.count(old)==1;driver=driver.replace(old,new)
driver=driver.replace("        child_records=child_records, retained_owned_files=len(owned_metadata),", "        child_records=child_records, early_parent_records_validated=len(child_records), retained_owned_files=len(owned_metadata),")
ast.parse(driver)
(OUT/'executed_driver.py').write_text(driver)
def ownership(raw):
    tree=ast.parse(raw)
    return ast.dump(next(n for n in ast.walk(tree)if isinstance(n,ast.With)and any(isinstance(c,ast.Call)and isinstance(c.func,ast.Attribute)and c.func.attr=='Popen'for c in ast.walk(n))),include_attributes=False)
assert ownership(old_driver)==ownership(driver)
def alias_prefix(raw):
    tree=ast.parse(raw);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='inspect_owned')
    return [ast.dump(n,include_attributes=False)for n in fn.body[:3]]
assert alias_prefix(old_driver)==alias_prefix(driver)
assert (OUT/'worker_driver.py').read_bytes()==old_worker
(OUT/'executed_driver.py.patch').write_text(''.join(difflib.unified_diff(old_driver.decode().splitlines(True),driver.splitlines(True),fromfile='preparation-v1/executed_driver.py',tofile='preparation-v2/executed_driver.py')))
(OUT/'worker_driver.py.patch').write_text('Worker byte-identical to preserved preparation-v1; no worker delta.\n')

history=OUT/'failed-v1';history.mkdir()
refs={}
for name,path in [
    ('failed-terminal-review.json',Path('/tmp/qwen-server-admission-snapshot-root-review-20261104/failed-terminal-review.json')),
    ('retention-review.json',Path('/tmp/qwen-snapshot-failed-run-retention-20261104/review.json')),
    ('retention-files.json',Path('/tmp/qwen-snapshot-failed-run-retention-20261104/files.json')),
    ('retention-owned-aliases.json',Path('/tmp/qwen-snapshot-failed-run-retention-20261104/owned-aliases.json')),
    ('retention-process-absence.json',Path('/tmp/qwen-snapshot-failed-run-retention-20261104/process-absence.json')),
    ('old-handoff.json',OLD/'handoff.json'),('old-tests.xml',OLD/'tests.xml'),('old-worker-audit.json',OLD/'worker-audit.json')]:
    raw=path.read_bytes();(history/name).write_bytes(raw);refs[name]=dict(original=str(path),sha256=sha(raw))
assert refs['failed-terminal-review.json']['sha256']=='fc08e491aa93ae6eb0961d7c0d008efd0b2cf83b00f9b47d5a06ef73b376eefd'
assert refs['retention-files.json']['sha256']=='cea9cc7683be38a068ad2008aef8fac4c7a113cb95045055b11ba4066586fdbb'
provenance=dict(schema='snapshot-focused-harness-source-provenance-v2',base=str(OLD),
    base_driver_sha256=sha(old_driver),base_worker_sha256=sha(old_worker),worker_unchanged=True,
    immediate_Popen_ownership_AST_exact=True,owned_pytest_alias_walk_AST_exact=True,
    declared_adaptations=['38-source/18-case held v3 pins and fresh output paths','Exact externally held pretty JSON map encoding, with before/after same encoding','Retain and semantically join18 early parent terminals to18 positive child terminals/results/logs; fail on missing or extra terminal counts','Preserve bounded owned pytest alias walk and original Popen wait/cleanup unchanged'],
    preserved_failure=refs,candidate_imported=False,collection=False,execution=False,actual_grant=False)
(OUT/'driver_provenance.json').write_bytes(encode(provenance))
(OUT/'README.md').write_text('''# Unrun snapshot v3 focused harness

This proposed preparation selects exactly38 source files, including unchanged28
runtime files and18 fresh-child recording guard scenarios. No source has been
imported or collected and no child has been launched by this preparation.

The previous37-source run failed18 fixture constructors before the recording seam
or guard and then lacked child terminal receipts. Its complete original proof and
bounded retention tree remain unchanged. Exact copied failure/retention receipts
under failed-v1 preserve that scope; no child process absence is retroactively
inferred for the failed run.

The new source bypasses only synthetic PublicContract construction, explicitly
without producing an admitted public contract. A new early parent-terminal.json
records direct wait and group observation before normal-exit/result assertions.
This driver requires all18 early records and all18 positive child terminals,
results and logs, validates exact typed joins, and preserves each record. It
records owned pytest current aliases without traversing or duplicating them.

The original outer Popen/direct-wait/exception-cleanup AST and worker bytes are
unchanged. CPU112-115, fresh owned TMPDIR/basetemp, explicit source pytest.ini,
empty numeric-module expectations, NumPy/backend denial and exact collection/XML
checks remain. This version binds the already held indent=2 source manifest;
the preceding version used a compact source map. Actual scientific validation,
capacity, model/backend/HE and actual role authority remain outside this suite.

Launch requires separate root authorization. No grant is included here.
''')
argv=['env','-i','PATH=/usr/bin:/bin','HOME=/home/jianming_gatech','LANG=C.UTF-8','TZ=UTC','PYTHONDONTWRITEBYTECODE=1','PYTEST_DISABLE_PLUGIN_AUTOLOAD=1','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','taskset','-c','112-115','/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(OUT/'executed_driver.py')]
proposal=dict(schema='snapshot-focused-launch-proposal-v2',held_unrun=True,ready_for_source_review_only=True,
    proof_root=str(OUT),source_root=str(SOURCE),source_count=38,runtime_count=28,static_cases=18,
    source_manifest_sha256=sha(mapping_raw),source_archive_sha256=sha(archive_raw),config_sha256=sha((OUT/'run_config.json').read_bytes()),
    driver_sha256=sha(driver.encode()),worker_sha256=sha(old_worker),argv=argv,
    early_parent_records_required=18,positive_child_records_required=18,current_execution_archive_equal=True,
    immediate_Popen_ownership_AST_exact=True,owned_pytest_alias_walk_AST_exact=True,
    candidate_imported=False,numpy_imports=False,collection=False,execution=False,capacity_observed=False,
    scientific_admission_tested=False,actual_proof_instances=False,launch_authorized=False)
(OUT/'launch-proposal.json').write_bytes(encode(proposal))
files={str(p.relative_to(OUT)):sha(p.read_bytes())for p in sorted(OUT.rglob('*'))if p.is_file()and 'execution' not in p.relative_to(OUT).parts}
handoff={**proposal,'files':files}
(OUT/'handoff.json').write_bytes(encode(handoff))
assert{str(p.relative_to(execution)):sha(p.read_bytes())for p in sorted(execution.rglob('*'))if p.is_file()}==mapping
assert all(sha((SOURCE/n).read_bytes())==d for n,d in mapping.items())
assert (OLD/'executed_driver.py').read_bytes()==old_driver and (OLD/'worker_driver.py').read_bytes()==old_worker
assert not(OUT/'tmp').exists()and not(OUT/'pytest-owned').exists()
print(json.dumps({**proposal,'handoff_sha256':sha((OUT/'handoff.json').read_bytes())},sort_keys=True))
