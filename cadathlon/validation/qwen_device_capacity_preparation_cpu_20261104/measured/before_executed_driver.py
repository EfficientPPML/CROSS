"""Held unrun public/recording snapshot proof driver; never auto-discovers tests.

Exact config and worker bytes are pinned. Source/test/driver review and an
explicit launch authorization remain external; preparation has not executed.
"""
from pathlib import Path
import hashlib, json, os, signal, stat, subprocess, sys, tarfile, time
import xml.etree.ElementTree as ET

CONFIG_SHA256 = '73f36f9634f726699a8c70ec826f2020b59713dc1fc9960b85859d27bb281ab0'
WORKER_SHA256 = '55b6e6f684480efb7b834f72d4678df778b3f63beb2f06d36b05ca663764b8be'
HERE = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
canonical = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
map_canonical = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()

def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {p.relative_to(root).as_posix(): sha(p.read_bytes())
            for p in sorted(root.rglob('*')) if p.is_file()}

def inspect_owned(out, config):
    metadata = {}
    total = 0
    for leaf in (out / 'tmp', out / 'pytest-owned'):
        assert leaf.is_dir() and not leaf.is_symlink()
        for base, dirs, files in os.walk(leaf, followlinks=False):
            for name in list(dirs):
                alias = Path(base) / name
                if not alias.is_symlink(): continue
                # Pytest's numbered tmp_path directories have owned *current
                # aliases. Record them once without traversing duplicate data.
                assert leaf == out / 'pytest-owned' and alias.parent == leaf and name.startswith('test_snapshot_guard_') and name.endswith('current')
                before = alias.lstat(); target_raw = os.readlink(alias)
                assert stat.S_ISLNK(before.st_mode) and before.st_uid == os.geteuid() and len(target_raw) <= 4096
                named_target = alias.parent / target_raw
                assert '..' not in Path(target_raw).parts
                target = named_target.resolve(strict=True)
                assert named_target == target
                prefix = name[:-len('current')]
                assert target.name.startswith(prefix) and target.name[len(prefix):].isdigit()
                assert target.parent == leaf and target.name != name and target.is_dir() and not target.is_symlink()
                target_stat = target.lstat(); assert target_stat.st_uid == os.geteuid()
                after = alias.lstat()
                assert (before.st_dev, before.st_ino, before.st_mode, before.st_uid) == (after.st_dev, after.st_ino, after.st_mode, after.st_uid)
                assert os.readlink(alias) == target_raw
                assert len(metadata) < config['owned_temp_max_files']
                metadata[alias.relative_to(out).as_posix()] = dict(kind='owned_pytest_alias', target=target_raw, resolved_target=target.relative_to(out).as_posix())
                dirs.remove(name)
            for name in files:
                path = Path(base) / name
                assert path.is_file() and not path.is_symlink()
                size = path.stat().st_size
                total += size
                assert len(metadata) < config['owned_temp_max_files'] and total <= config['owned_temp_max_bytes']
                metadata[path.relative_to(out).as_posix()] = dict(kind='regular', bytes=size, sha256=sha(path.read_bytes()))
    expected = [n.split('[', 1)[1][:-1] for n in config['selected_nodes']]
    records = {}
    regular = [out / name for name, item in metadata.items() if item['kind'] == 'regular' and name.startswith('pytest-owned/')]
    for path in sorted(p for p in regular if p.name == 'child-terminal.json'):
        terminal = json.loads(path.read_bytes())
        result_path, log_path = path.parent / 'result.json', path.parent / 'child.log'
        parent_path = path.parent / 'parent-terminal.json'
        assert all(p in regular for p in (result_path, log_path, parent_path))
        result = json.loads(result_path.read_bytes())
        parent = json.loads(parent_path.read_bytes())
        assert type(parent) is dict and set(parent) == {'schema', 'pid', 'owned_session_pgid', 'observed_pgid', 'exit_code', 'reaped', 'group_absent', 'scenario', 'child_result_read', 'runtime_scope'}
        assert parent['schema'] == 'qwen-snapshot-owned-child-terminal-v1'
        assert parent['child_result_read'] is False and parent['runtime_scope'] == 'recording guard state only'
        scenario = terminal['scenario']
        assert scenario in expected and scenario not in records and result['scenario'] == scenario
        pid, pgid = result['pid'], result['pgid']
        assert type(pid) is int and pid > 0 and type(pgid) is int and pgid == pid
        assert type(terminal['pid']) is int and terminal['pid'] == pid
        assert type(terminal['child_reported_pgid']) is int and terminal['child_reported_pgid'] == pid
        assert terminal['observed_pgid'] is None or (type(terminal['observed_pgid']) is int and terminal['observed_pgid'] == pid)
        assert type(terminal['exit_code']) is int and terminal['exit_code'] == 0
        assert terminal['reaped'] is True and terminal['group_absent'] is True
        assert parent['scenario'] == scenario
        assert type(parent['pid']) is int and parent['pid'] == pid
        assert type(parent['owned_session_pgid']) is int and parent['owned_session_pgid'] == pid
        assert parent['observed_pgid'] is None or (type(parent['observed_pgid']) is int and parent['observed_pgid'] == pid)
        assert parent['observed_pgid'] == terminal['observed_pgid']
        assert type(parent['exit_code']) is int and parent['exit_code'] == terminal['exit_code'] == 0
        assert parent['reaped'] is True and parent['group_absent'] is True
        assert result['assertions_completed'] is True and result['original_validator_identities_restored'] is True
        assert result['recording_full_validation_seam'] is True and result['scientific_admission_tested'] is False
        assert result['actual_proof_instances'] is False and result['backend_imported'] is False and result['backend_denials'] == []
        assert type(result['graph_type_count']) is int and result['graph_type_count'] == 13
        for group in (False, True):
            try: (os.killpg if group else os.kill)(pid, 0)
            except ProcessLookupError: pass
            else: raise ValueError('owned snapshot child or group remains live')
        records[scenario] = dict(terminal=terminal, result=result, parent_terminal=parent,
            parent_terminal_path=parent_path.relative_to(out).as_posix(), parent_terminal_sha256=sha(parent_path.read_bytes()),
            terminal_path=path.relative_to(out).as_posix(), result_path=result_path.relative_to(out).as_posix(),
            log_path=log_path.relative_to(out).as_posix(), log_sha256=sha(log_path.read_bytes()))
    assert set(records) == set(expected) and len(records) == 18
    assert type(config['expected_parent_records']) is int and config['expected_parent_records'] == 18
    assert sum(p.name == 'parent-terminal.json' for p in regular) == config['expected_parent_records']
    assert sum(p.name == 'child-terminal.json' for p in regular) == 18
    assert sum(p.name == 'result.json' for p in regular) == sum(p.name == 'child.log' for p in regular) == 18
    return metadata, [records[name] for name in expected]


def main():
    # Reject the unfinalized template before any source, evidence or child work.
    if not isinstance(CONFIG_SHA256, str) or not isinstance(WORKER_SHA256, str):
        raise ValueError('snapshot proof pins remain unset; launch disabled')
    raw = (HERE / 'run_config.json').read_bytes()
    assert sha(raw) == CONFIG_SHA256
    config = json.loads(raw)
    assert config['schema'] == 'qwen-admission-snapshot-focused-proof-v2'
    assert config['synthetic_only'] is True and config['actual_execution_authorized'] is False
    source = Path(config['source_root'])
    out = Path(config['proof_root'])
    execution = out / 'execution'
    assert source.is_absolute() and out.is_absolute()
    assert type(config['source_count']) is int and config['source_count'] > 0
    nodes, xml_cases = config['selected_nodes'], config['selected_xml_cases']
    assert type(nodes) is list and nodes and len(nodes) == len(set(nodes))
    assert type(xml_cases) is list and len(xml_cases) == len(nodes)
    assert sha(canonical(nodes)) == config['selected_nodes_sha256']
    assert sha(canonical(xml_cases)) == config['selected_xml_cases_sha256']
    assert os.sched_getaffinity(0) == {112, 113, 114, 115}
    def dump(name, value): (out / name).write_bytes(canonical(value))
    manifest = (out / 'sources-before.json').read_bytes()
    assert sha(manifest) == config['source_manifest_sha256']
    expected = json.loads(manifest)
    assert manifest == map_canonical(expected)
    assert len(expected) == config['source_count']
    assert inventory(source) == inventory(execution) == expected
    assert sha((out / 'sources.tar.gz').read_bytes()) == config['source_archive_sha256']
    def archived():
        with tarfile.open(out / 'sources.tar.gz') as archive:
            members = archive.getmembers()
            assert len(members) == len({m.name for m in members}) == len(expected)
            assert all(m.isfile() and not Path(m.name).is_absolute()
                       and '..' not in Path(m.name).parts for m in members)
            return {m.name: sha(archive.extractfile(m).read()) for m in members}
    assert archived() == expected
    assert Path(__file__).read_bytes() == (out / 'executed_driver.py').read_bytes()
    assert sha((out / 'worker_driver.py').read_bytes()) == WORKER_SHA256
    assert (out / 'run_config.json').read_bytes() == raw
    temporary = Path(config['owned_tmpdir'])
    basetemp = Path(config['pytest_basetemp'])
    assert temporary == out / 'tmp' and basetemp == out / 'pytest-owned'
    assert temporary.parent.resolve() == basetemp.parent.resolve() == out.resolve()
    assert not os.path.lexists(temporary) and not os.path.lexists(basetemp)
    temporary.mkdir(mode=0o700)
    env = dict(PATH='/usr/bin:/bin', HOME='/home/jianming_gatech', LANG='C.UTF-8', TZ='UTC', TMPDIR=str(temporary),
               PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
               OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    command = ['taskset', '-c', '112-115', '/tmp/cross-pretrained-validation-env/bin/python',
               '-I', '-B', str(out / 'worker_driver.py'), str(out / 'run_config.json'), CONFIG_SHA256]
    start = time.monotonic()
    with (out / 'tests.log').open('xb') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            # Ownership starts immediately after Popen, before any fallible
            # identity observation or evidence publication.
            child_pgid = os.getpgid(child.pid)
            if child_pgid != child.pid:
                raise ValueError('owned child did not retain its fresh session group')
            dump('live-observation.json', dict(outer_pid=os.getpid(), outer_pgid=os.getpgrp(),
                 worker_pid=child.pid, worker_pgid=child_pgid, command=command,
                 environment=env, config_sha256=CONFIG_SHA256))
            code = child.wait()  # No healthy-execution deadline.
        except BaseException as primary:
            # start_new_session establishes the owned group even if getpgid or
            # live publication failed. Always attempt direct wait after cleanup.
            try:
                try: os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError: pass
                except BaseException as cleanup:
                    if cleanup is not primary: primary.add_note('owned group cleanup: ' + repr(cleanup))
            finally:
                try: child.wait()
                except BaseException as cleanup:
                    if cleanup is not primary: primary.add_note('owned direct wait: ' + repr(cleanup))
            raise
    try: os.killpg(child.pid, 0)
    except ProcessLookupError: group_absent = True
    else: group_absent = False
    current, after = inventory(source), inventory(execution)
    (out / 'sources-after.json').write_bytes(map_canonical(current))
    (out / 'execution-after.json').write_bytes(map_canonical(after))
    archive_after = archived()
    owned_metadata, child_records = inspect_owned(out, config)
    dump('owned-pytest-metadata.json', owned_metadata)
    dump('test-child-terminals.json', child_records)
    xml = ET.parse(out / 'tests.xml').getroot()
    counts = {name: len(list(xml.iter(tag))) for name, tag in
              [('tests', 'testcase'), ('failures', 'failure'), ('errors', 'error'), ('skipped', 'skipped')]}
    observed_xml = [dict(classname=c.attrib['classname'], name=c.attrib['name'])
                    for c in xml.iter('testcase')]
    audit = json.loads((out / 'worker-audit.json').read_bytes())
    passed = (code == 0 and group_absent and current == after == archive_after == expected
              and counts == dict(tests=len(nodes), failures=0, errors=0, skipped=0)
              and observed_xml == xml_cases and audit['collected_nodes'] == nodes
              and audit['denied'] == audit['loaded_backend'] == []
              and audit['forbidden_roots'] == config['forbidden_roots']
              and audit['numpy_imported'] is False
              and type(audit['pytest_exit_code']) is int and audit['pytest_exit_code'] == 0
              and type(audit['pid']) is int and audit['pid'] == child.pid
              and type(audit['pgid']) is int and audit['pgid'] == child_pgid
              and audit['affinity'] == [112, 113, 114, 115]
              and audit['temporary_root'] == str(temporary) and audit['scientific_admission_tested'] is False
              and len(child_records) == config['expected_child_records']
              and audit['runtime_sources_before'] == audit['runtime_sources_after'] == config['runtime_source_expectations']
              and audit['runtime_module_origins'] == {n: dict(file=p, spec=p) for n,p in config['runtime_module_origins'].items()})
    report = dict(passed=passed, tests=counts, selected_nodes=nodes, selected_xml_cases=observed_xml,
        source_count=len(expected), source_manifest_sha256=config['source_manifest_sha256'],
        source_archive_sha256=config['source_archive_sha256'],
        sources_unchanged=current == after == archive_after == expected,
        child_returncode=code, child_reaped=True, child_group_absent=group_absent,
        elapsed_s=time.monotonic()-start, synthetic_only=True,
        actual_execution_authorized=False, numpy_imported=audit['numpy_imported'], model_execution=False, encrypted_execution=False,
        backend_context_construction=False, actual_key_generation=False, actual_client_role_execution=False,
        driver_sha256=sha(Path(__file__).read_bytes()), worker_driver_sha256=WORKER_SHA256,
        config_sha256=CONFIG_SHA256, scope=config['scope'],
        child_records=child_records, early_parent_records_validated=len(child_records), retained_owned_files=len(owned_metadata),
        snapshot_guard_recording_only=True, full_scientific_admission_tested=False,
        actual_proofs_present=False, temporary_root=str(temporary), pytest_basetemp=str(basetemp))
    dump('report.json', report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if passed else 1

if __name__ == '__main__': raise SystemExit(main())
