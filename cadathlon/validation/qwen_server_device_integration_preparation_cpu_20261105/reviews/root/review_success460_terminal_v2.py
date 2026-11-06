"""Root-owned post-terminal evidence review; fixture bodies are never read.

Only run after tool session2862 has returned a terminal exit0 and the root has
written tool-terminal460.json from that observed result. This is an evidence
reader, not a runtime import or another execution of any selected test.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import stat
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).parent
PROOF = Path('/tmp/qwen-full28-device-focused-proof-20261104/preparation-v4')
SOURCE = Path('/tmp/qwen-full28-device-focused-tests-v3-20261105')
PINS = {}
COLLECTOR = (922320, 20418229)
WORKER = (922321, 20418255)

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def state(s):
    return (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

def read(path, limit=2 << 20):
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
    assert 0 <= before.st_size <= limit
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        assert state(before) == state(os.fstat(fd))
        chunks, total = [], 0
        while total <= limit:
            block = os.read(fd, min(65536, limit + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
        assert total == before.st_size and total <= limit
        assert state(before) == state(os.fstat(fd)) == state(path.lstat())
    finally:
        os.close(fd)
    raw = b''.join(chunks)
    PINS[str(path)] = {'bytes': len(raw), 'sha256': sha(raw)}
    return raw

def document(path):
    return json.loads(read(path))

def gone(pid):
    for operation in (os.kill, os.killpg):
        try:
            operation(pid, 0)
        except ProcessLookupError:
            continue
        raise AssertionError('Original process or group remains: ' + str(pid))

def directory(path):
    s = path.lstat()
    assert stat.S_ISDIR(s.st_mode) and s.st_uid == os.geteuid()

def source_inventory(root):
    result = {}
    def walk(path):
        directory(path)
        for p in sorted(path.iterdir()):
            s = p.lstat()
            if stat.S_ISDIR(s.st_mode):
                walk(p)
            else:
                relative = p.relative_to(root).as_posix()
                assert relative in expected, 'Unselected source path: ' + relative
                result[relative] = sha(read(p, 8 << 20))
    walk(root)
    return result

# This gate precedes any candidate result or test record read.
tool = document(HERE / 'tool-terminal460.json')
assert tool['session_id'] == 2862 and tool['exit_code'] == 0
assert type(tool['terminal_tool']) is str and tool['terminal_tool']
assert tool['terminal_observed_by_root'] is True
gone(COLLECTOR[0])
gone(WORKER[0])

config_raw = read(PROOF / 'run_config.json')
assert sha(config_raw) == '30b44815308ab8521d916df3d09a61bae35bc70884f84111e6d7ffd921f945f2'
config = json.loads(config_raw)
assert config['source_root'] == str(SOURCE) and config['proof_root'] == str(PROOF)
assert config['synthetic_only'] is True and config['actual_execution_authorized'] is False
assert sha(read(PROOF / 'executed_driver.py')) == '46f94661e2ab07de22d86f740e26e569ef07ad9a8fee76678855e0b31a4945f8'
assert sha(read(PROOF / 'worker_driver.py')) == 'e2f304bfceb6bcd2bfaaaa57c2e3c78f10ca419390c3b0ef57a8236ad7027543'
assert sha(read(PROOF / 'inspect_children.py')) == '26288c3a11dbe7e2578920b070bad4cf281e85a877934af39bd97bec7597b0cf'
source_raw = read(PROOF / 'sources-before.json')
assert sha(source_raw) == 'b5697bc650e3476c45500ef7e7e1c6a68e3432fc1c5db20410bffb4f1b1b64a8'
expected = json.loads(source_raw)
assert len(expected) == 460
assert source_inventory(SOURCE) == source_inventory(PROOF / 'execution') == expected
assert document(PROOF / 'sources-after.json') == document(PROOF / 'execution-after.json') == expected
archive_raw = read(PROOF / 'sources.tar.gz', 8 << 20)
assert sha(archive_raw) == 'ae93be7bc57373d606de5957f8bf376ea8724523e360cbfdefa21e40144297dc'
import io
with tarfile.open(fileobj=io.BytesIO(archive_raw), mode='r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == len({m.name for m in members}) == 460
    assert set(m.name for m in members) == set(expected)
    for member in members:
        assert member.isfile() and 0 <= member.size <= 8 << 20
        assert sha(archive.extractfile(member).read()) == expected[member.name]

live = document(PROOF / 'live-observation.json')
root_live = document(HERE / 'live460.json')
assert PINS[str(HERE / 'live460.json')]['sha256'] == '1fa77ca8f84c32337abf5be0bd9b953e99ac6a3a5ef513fa2db4769474bcb3fb'
collector = document(PROOF / 'collector-terminal.json')
owner = document(PROOF / 'worker-owner-terminal.json')
worker = document(PROOF / 'worker-terminal.json')
early = document(PROOF / 'worker-early.json')
audit = document(PROOF / 'worker-audit.json')
assert collector['pid'] == collector['pgid'] == live['outer_pid'] == live['outer_pgid'] == COLLECTOR[0]
assert collector['start_ticks'] == live['outer_start_ticks'] == COLLECTOR[1]
assert collector['normal_main_return'] is True and collector['planned_exit_code'] == 0 and collector['error_type'] is None
assert owner['worker_pid'] == owner['owned_session_pgid'] == live['worker_pid'] == WORKER[0]
assert owner['parent_pid'] == owner['parent_pgid'] == COLLECTOR[0]
assert owner['parent_start_ticks'] == COLLECTOR[1] and owner['start_ticks'] == WORKER[1]
assert owner['exit_code'] == 0 and owner['primary_type'] is None and owner['cleanup_types'] == []
for name in ('direct_wait_completed', 'child_reaped', 'pid_absent', 'group_absent'):
    assert owner[name] is True
for record in (worker, early, audit):
    assert record['pid'] == record['pgid'] == WORKER[0] and record['start_ticks'] == WORKER[1]
assert worker['normal_main_return'] is True and worker['pytest_exit_code'] == worker['planned_exit_code'] == 0
assert early['config_sha256'] == sha(config_raw) and early['source_manifest_sha256'] == sha(source_raw)
assert audit['collected_nodes'] == config['selected_nodes'] and len(audit['collected_nodes']) == 59
assert audit['denied'] == audit['loaded_backend'] == audit['cleanup_types'] == []
assert audit['numpy_imported'] is False and audit['primary_type'] is None
assert audit['pytest_exit_code'] == audit['planned_exit_code'] == 0
assert audit['forbidden_roots'] == config['worker_forbidden_roots'] and audit['affinity'] == [112, 113, 114, 115]
assert audit['runtime_sources_before'] == audit['runtime_sources_after'] == config['runtime_source_expectations']
assert len(audit['runtime_sources_before']) == 403 and audit['runtime_module_origins'] == {}
xml = ET.fromstring(read(PROOF / 'tests.xml'))
cases = list(xml.iter('testcase'))
assert len(cases) == 59 and not any(list(xml.iter(tag)) for tag in ('failure', 'error', 'skipped'))
assert [{'classname': c.attrib['classname'], 'name': c.attrib['name']} for c in cases] == config['selected_xml_cases']
read(PROOF / 'tests.log')

base = PROOF / 'pytest-owned'
directory(base)
plan = config['artifact_plan']
records, aliases, fixtures = {}, {}, {}
total = 0
for child in sorted(base.iterdir()):
    s = child.lstat()
    if stat.S_ISLNK(s.st_mode):
        assert s.st_uid == os.geteuid() and child.name.startswith('test_device_integration_') and child.name.endswith('current')
        target = os.readlink(child)
        lexical = Path(target)
        assert 0 < len(target) <= 4096 and '\\' not in target and str(lexical) == target
        assert '.' not in lexical.parts and '..' not in lexical.parts
        if not lexical.is_absolute():
            assert lexical.name == target and '/' not in target
        named = lexical if lexical.is_absolute() else base / lexical
        assert named.parent == base and named != child
        directory(named)
        prefix = child.name[:-len('current')]
        assert named.name.startswith(prefix) and named.name[len(prefix):].isdigit()
        assert state(s) == state(child.lstat()) and os.readlink(child) == target
        aliases[child.name] = {'target': target, 'followed': False}
        continue
    directory(child)
    raw = {name: read(child / name, plan['per_record_limit_bytes']) for name in plan['records_per_child']}
    total += sum(len(b) for b in raw.values())
    assert total <= plan['total_record_limit_bytes']
    parent = json.loads(raw['parent-terminal.json'])
    terminal = json.loads(raw['child-terminal.json'])
    result = json.loads(raw['result.json'])
    case = result['case']
    assert case in config['selected_cases'] and case not in records
    assert parent['case'] == terminal['case'] == case
    pid = result['pid']
    assert type(pid) is int and pid > 0
    assert result['pgid'] == terminal['pid'] == terminal['pgid'] == parent['pid'] == parent['owned_session_pgid'] == pid
    assert parent['observed_pgid'] is None or parent['observed_pgid'] == pid
    ticks = result['starttime_ticks']
    assert type(ticks) is int and ticks > 0 and terminal['starttime_ticks'] == ticks
    assert parent['starttime_ticks'] is None or parent['starttime_ticks'] == ticks
    assert type(parent['exit_code']) is int and parent['exit_code'] == 0
    for name in ('direct_wait_completed', 'reaped', 'group_absent', 'child_report_read'):
        assert parent[name] is True
    assert parent['primary_type'] is None and terminal['normal_return'] is True and terminal['status'] == 'normal_return'
    assert result['passed'] is True and result['backend_denials'] == terminal['backend_denials'] == []
    assert terminal['result_sha256'] == parent['result_sha256'] == sha(raw['result.json'])
    assert parent['child_terminal_sha256'] == sha(raw['child-terminal.json'])
    for name in ('definition_only_entry', 'original_main_restored', 'temporary_names_removed', 'profile_restored', 'declared_exports_restored', 'source_unchanged'):
        assert result[name] is True
    for name in ('genuine_cli_bootstrap', 'model_key_ciphertext_body_reads', 'native_backend_imported', 'capacity_measured', 'actual_execution_authorized'):
        assert result[name] is False
    gone(pid)
    observed, directories, record_names = {}, [], set()
    def metadata_walk(path, prefix=''):
        for item in path.iterdir():
            info = item.lstat()
            relative = prefix + item.name
            assert info.st_uid == os.geteuid() and not stat.S_ISLNK(info.st_mode)
            if stat.S_ISDIR(info.st_mode):
                directories.append(relative)
                metadata_walk(item, relative + '/')
            elif relative in raw:
                assert prefix == '' and stat.S_ISREG(info.st_mode)
                record_names.add(relative)
            else:
                assert relative in plan['fixture_files'][case] and stat.S_ISREG(info.st_mode)
                observed[relative] = dict(bytes=info.st_size, device=info.st_dev, inode=info.st_ino, mode=stat.S_IMODE(info.st_mode))
    metadata_walk(child)
    declared = parent['fixture_metadata']
    assert declared['body_read'] is False and declared['body_copied'] is False
    assert record_names == set(raw) and set(observed) == set(plan['fixture_files'][case]) == set(declared['files'])
    assert sorted(directories) == declared['directories']
    for name, values in observed.items():
        assert all(declared['files'][name][k] == v for k, v in values.items())
        fixtures[(child / name).relative_to(PROOF).as_posix()] = dict(**values, body_read=False)
    records[case] = dict(pid=pid, pgid=pid, starttime_ticks=ticks, original_wait_completed=True,
                         independently_absent=True, child_directory=child.name,
                         pins={name: {'bytes': len(b), 'sha256': sha(b)} for name, b in raw.items()})

assert set(records) == set(config['selected_cases']) and len(records) == len({r['pid'] for r in records.values()}) == 59
assert not {r['pid'] for r in records.values()} & {COLLECTOR[0], WORKER[0]}
assert sum(len(r['pins']) for r in records.values()) == plan['expected_regular_record_count'] == 236
assert len(fixtures) == plan['fixture_regular_count'] == 14
directory(PROOF / 'tmp')
assert not list((PROOF / 'tmp').iterdir())
report = document(PROOF / 'report.json')
artifact_review = document(PROOF / 'child-artifact-review.json')
assert report['passed'] is True and report['tests'] == dict(tests=59, failures=0, errors=0, skipped=0)
assert report['source_count'] == 460 and report['runtime_count'] == 403 and report['sources_unchanged'] is True
assert report['source_manifest_sha256'] == sha(source_raw) and report['source_archive_sha256'] == sha(archive_raw)
for name in ('actual_execution_authorized', 'backend_context_construction', 'actual_key_generation', 'actual_model_execution', 'encrypted_execution', 'genuine_capacity_measured', 'fixture_bodies_copied', 'aliases_followed'):
    assert report[name] is False
assert report['artifact_review_sha256'] == PINS[str(PROOF / 'child-artifact-review.json')]['sha256']
assert artifact_review['passed'] is True and artifact_review['validated_children'] == 59
assert artifact_review['regular_files'] == 236 and artifact_review['fixture_metadata_files'] == 14
assert artifact_review['fixture_bodies_read'] is False and artifact_review['fixture_bodies_copied'] is False and artifact_review['aliases_followed'] is False
assert [r['case'] for r in artifact_review['records']] == config['selected_cases']
for row in artifact_review['records']:
    original = records[row['case']]
    assert all(row[name] == original[name] for name in ('pid', 'pgid', 'starttime_ticks'))
gone(COLLECTOR[0])
gone(WORKER[0])
assert source_inventory(SOURCE) == source_inventory(PROOF / 'execution') == expected
result = dict(schema='full28-focused460-root-success-terminal-v1', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              scope='59 focused recording tests only; no genuine backend, capacity, model/key/ciphertext body or HE proof',
              tool_terminal=tool, source_count=460, runtime_count=403, tests_passed=59,
              source_current_execution_archive_exact=True, all61_original_pids_groups_absent=True,
              child_records=records, fixture_metadata=fixtures, aliases=aliases,
              fixture_bodies_read=False, fixture_bodies_copied=False, aliases_followed=False,
              elapsed_s=report['elapsed_s'], public_pins=PINS, root_live_receipt_sha256=PINS[str(HERE / 'live460.json')]['sha256'])
destination = HERE / 'success460-terminal-review.json'
assert not destination.exists()
payload = (json.dumps(result, sort_keys=True, indent=2) + '\n').encode()
destination.write_bytes(payload)
print(json.dumps({'path': str(destination), 'sha256': sha(payload), 'passed_tests': 59, 'regular_child_records': 236, 'fixtures_metadata_only': 14}))
