"""Finalized public/synthetic proc-self proof driver; never auto-discovers tests.

Exact config and worker bytes are pinned. Source/test/driver review and an
explicit launch authorization remain external; preparation has not executed.
"""
from pathlib import Path
import hashlib, json, os, signal, subprocess, sys, tarfile, time
import xml.etree.ElementTree as ET

CONFIG_SHA256 = '1b918582e25198b221911c85f3c582672d12c50821dcccf91f110f65b1f27aa9'
WORKER_SHA256 = 'e151f7706e792e01c4cddc10029ac03a10dd95e4417042edbd0b23e425d0a8f3'
HERE = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
canonical = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
compact_map = lambda value: (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()

def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {p.relative_to(root).as_posix(): sha(p.read_bytes())
            for p in sorted(root.rglob('*')) if p.is_file()}

def main():
    # Reject the unfinalized template before any source, evidence or child work.
    if not isinstance(CONFIG_SHA256, str) or not isinstance(WORKER_SHA256, str):
        raise ValueError('proc-self proof pins remain unset; launch disabled')
    raw = (HERE / 'run_config.json').read_bytes()
    assert sha(raw) == CONFIG_SHA256
    config = json.loads(raw)
    assert config['schema'] == 'qwen-keygen-proc-self-focused-proof-v1'
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
    assert config['source_manifest_encoding'] == 'canonical sorted compact JSON plus newline'
    assert manifest == compact_map(expected)
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
    env = dict(PATH='/usr/bin:/bin', HOME='/home/jianming_gatech', LANG='C.UTF-8', TZ='UTC',
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
    (out / 'sources-after.json').write_bytes(compact_map(current))
    (out / 'execution-after.json').write_bytes(compact_map(after))
    archive_after = archived()
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
              and audit['capacity_evaluated'] is False
              and audit['own_process_metadata_scope'] == config['own_process_metadata_scope']
              and type(audit['pytest_exit_code']) is int and audit['pytest_exit_code'] == 0
              and type(audit['pid']) is int and audit['pid'] == child.pid
              and type(audit['pgid']) is int and audit['pgid'] == child_pgid
              and audit['affinity'] == [112, 113, 114, 115]
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
        own_process_metadata_scope=config['own_process_metadata_scope'],
        process_metadata_contents_recorded=False, resource_capacity_evaluated=False,
        genuine_backend_imports_authorized=False, temporary_module_name_doubles=False)
    dump('report.json', report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if passed else 1

if __name__ == '__main__': raise SystemExit(main())
