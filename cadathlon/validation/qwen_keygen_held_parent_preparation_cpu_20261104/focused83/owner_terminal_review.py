"""Public owner terminal readback. No tested module imports or test reruns."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
pretty = lambda value: (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def inventory(root):
    paths = sorted(root.rglob('*'))
    assert all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in paths)
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in paths if p.is_file()}


def absent(pid, group=False):
    try:
        (os.killpg if group else os.kill)(pid, 0)
    except ProcessLookupError:
        return True
    return False


def main():
    config_raw = (HERE/'run_config.json').read_bytes()
    assert sha(config_raw) == '3609af883f663629cac19012ed1f745b5711a8c79da00b4717bde96b23387dc7'
    config = json.loads(config_raw)
    report_raw = (HERE/'report.json').read_bytes()
    report = json.loads(report_raw)
    audit = json.loads((HERE/'worker-audit.json').read_bytes())
    live = json.loads((HERE/'live-observation.json').read_bytes())
    raw = (HERE/'sources-before.json').read_bytes()
    expected = json.loads(raw)
    assert sha(raw) == '50625b443ad26a88907e37ed901618b55fe3f5370ae96504b49dbaa0ca6974a1'
    assert raw == (json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    assert len(expected) == 83
    assert inventory(Path(config['source_root'])) == inventory(HERE/'execution') == expected
    assert json.loads((HERE/'sources-after.json').read_bytes()) == expected
    assert json.loads((HERE/'execution-after.json').read_bytes()) == expected
    archive = HERE/'sources.tar.gz'
    assert sha(archive.read_bytes()) == '93961105185d6c22566f8fbd315f392a75e6bdfc114dc7937c3721279d17ccbc'
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert len(members) == len({m.name for m in members}) == 83
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert {m.name: sha(tar.extractfile(m).read()) for m in members} == expected
    parent_raw = Path('/tmp/qwen-keygen-held-parent-sources-v1-20261104.json').read_bytes()
    assert sha(parent_raw) == '7ffa419573da6100462fbc7d8ab778975c11d37fba4e551bd86c3b92d448a4b9'
    parent = json.loads(parent_raw)
    assert len(parent) == 74
    assert all(expected['runtime/'+name] == digest for name, digest in parent.items())
    assert inventory(Path('/tmp/qwen-key-generation-held-parent-20261104')) == parent
    xml = ET.parse(HERE/'tests.xml').getroot()
    cases = [dict(classname=n.attrib['classname'], name=n.attrib['name']) for n in xml.iter('testcase')]
    counts = {name: len(list(xml.iter(tag))) for name, tag in [('tests', 'testcase'), ('failures', 'failure'), ('errors', 'error'), ('skipped', 'skipped')]}
    assert counts == dict(tests=34, failures=0, errors=0, skipped=0) == report['tests']
    assert cases == config['selected_xml_cases'] == report['selected_xml_cases']
    assert audit['collected_nodes'] == config['selected_nodes'] == report['selected_nodes']
    assert len(audit['collected_nodes']) == len(set(audit['collected_nodes'])) == 34
    assert audit['denied'] == audit['loaded_backend'] == []
    assert audit['numpy_imported'] is True and report['numpy_imported'] is True
    assert audit['runtime_sources_before'] == audit['runtime_sources_after'] == config['runtime_source_expectations']
    assert len(config['runtime_source_expectations']) == len(config['runtime_module_origins']) == 3
    assert {n: sha(Path(n).read_bytes()) for n in config['runtime_source_expectations']} == config['runtime_source_expectations']
    assert audit['runtime_module_origins'] == {n: dict(file=p, spec=p) for n, p in config['runtime_module_origins'].items()}
    assert audit['forbidden_roots'] == config['forbidden_roots'] and 'numpy' not in audit['forbidden_roots']
    assert audit['affinity'] == [112, 113, 114, 115]
    assert audit['pid'] == live['worker_pid'] == 81342 and audit['pgid'] == live['worker_pgid'] == 81342
    assert live['outer_pid'] == live['outer_pgid'] == 81341
    assert type(audit['pytest_exit_code']) is int and audit['pytest_exit_code'] == 0
    assert report['passed'] is True and report['sources_unchanged'] is True
    assert type(report['child_returncode']) is int and report['child_returncode'] == 0
    assert report['child_reaped'] is True and report['child_group_absent'] is True
    assert all(report[n] is False for n in ('actual_execution_authorized', 'actual_key_generation', 'backend_context_construction', 'model_execution', 'encrypted_execution'))
    assert report['synthetic_only'] is True
    assert sha((HERE/'executed_driver.py').read_bytes()) == report['driver_sha256'] == '724311947123ea8a174d03324ef287db4e607307f567d181dfb40bad4865eeb0'
    assert sha((HERE/'worker_driver.py').read_bytes()) == report['worker_driver_sha256'] == '3d17fc29050faa6177634ef7be52c643c6d9fa6b5824ebb17cd94c657adef89e'
    proposal_raw = (HERE/'launch-proposal.json').read_bytes()
    assert sha(proposal_raw) == 'a2b411a038eb35cb4e4fbd4a12e9dfcc6b56ef886cd44709fa5e7130a9f74019'
    grant = Path('/tmp/qwen-key-generation-actual-root-review-20261104/held-parent83-one-run-authorization.json')
    assert sha(grant.read_bytes()) == '9440e50fd0e974e5e424002b207254737bd12a4bf224ee3ee441a7b54bfe1e15'
    states = {str(pid): dict(pid_absent=absent(pid), group_absent=absent(pid, True)) for pid in (81341, 81342)}
    assert all(all(row.values()) for row in states.values())
    evidence = ['report.json', 'tests.xml', 'tests.log', 'worker-audit.json', 'live-observation.json',
        'sources-before.json', 'sources-after.json', 'execution-after.json', 'sources.tar.gz',
        'executed_driver.py', 'worker_driver.py', 'run_config.json', 'launch-proposal.json',
        'source-review.json', 'owner_terminal_review.py']
    result = dict(schema='qwen-held-parent-owner-terminal-v1', passed=True,
        checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(), owner_session=None,
        authoritative_direct_tool_chunk='0a780d', owner_tool_exit_code=0, owner_reaped=True,
        persistent_session_created=False, independent_live_observation=False,
        live_identity_scope='Driver captured its own and owned child identities before direct wait; no independent external live observation because the run finished in the initial tool call.',
        process_states=states, child_direct_returncode=0, child_reaped=True, tests=counts,
        elapsed_s=report['elapsed_s'], source_count=83, all_parent74_unchanged=True,
        all_current_execution_start_end_archive_equal=True, exact_collection_and_xml_order=True,
        numpy_three_sources_and_origins_exact=True, numpy_genuinely_loaded=True,
        no_backend_import=True, audit_denials=[], actual_authority=False,
        scope=config['scope'], authorization_sha256=sha(grant.read_bytes()),
        files={name: sha((HERE/name).read_bytes()) for name in evidence})
    receipt_raw = pretty(result)
    with (HERE/'owner-terminal.json').open('xb') as f:
        f.write(receipt_raw)
    handoff = dict(schema='qwen-held-parent-terminal-handoff-v1', proof_root=str(HERE),
        source_manifest_sha256=sha(raw), source_archive_sha256=sha(archive.read_bytes()),
        report_sha256=sha(report_raw), owner_receipt_sha256=sha(receipt_raw),
        owner_receipt='owner-terminal.json', owner_script_sha256=sha(Path(__file__).read_bytes()),
        source_count=83, tests=34, passed=True, owner_session=None,
        authoritative_direct_tool_chunk='0a780d', exit_code=0, reaped=True,
        independent_live_observation=False, no_actual_role_authorized=True,
        root_independent_terminal_review_pending=True)
    handoff_raw = pretty(handoff)
    with (HERE/'terminal-handoff.json').open('xb') as f:
        f.write(handoff_raw)
    print(json.dumps(dict(report_sha256=sha(report_raw), owner_sha256=sha(receipt_raw),
        handoff_sha256=sha(handoff_raw), source_count=83, tests=counts,
        elapsed_s=report['elapsed_s'], process_states=states), sort_keys=True))


if __name__ == '__main__':
    main()
