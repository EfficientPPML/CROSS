"""Portable public evidence verification; never imports candidate/runtime code."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
load = lambda name: json.loads((HERE/name).read_bytes())


def inventory(root):
    files = sorted(root.rglob('*'))
    assert all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in files)
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in files if p.is_file()}


def archive(prefix, count):
    raw = (HERE/prefix/'sources.json').read_bytes()
    expected = json.loads(raw)
    assert raw == (json.dumps(expected, sort_keys=True, separators=(',', ':'))+'\n').encode()
    assert len(expected) == count
    with tarfile.open(HERE/prefix/'sources.tar.gz') as tar:
        members = tar.getmembers()
        assert len(members) == len({m.name for m in members}) == count
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        content = {m.name: tar.extractfile(m).read() for m in members}
    assert {name: sha(raw) for name, raw in content.items()} == expected
    return expected, content


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', action='store_true', help='Additionally compare original source/NumPy files; no imports or bodies.')
    args = parser.parse_args()
    expected = load('index.json')
    current = inventory(HERE)
    assert set(current) == set(expected) | {'index.json'}
    assert all(current[name] == digest for name, digest in expected.items())
    whitelist = load('public_whitelist.json')
    assert all(current[row['destination']] == row['sha256'] for row in whitelist['copied_files'])
    map74, content74 = archive(Path('source74'), 74)
    map83, content83 = archive(Path('focused83'), 83)
    assert all(map83['runtime/'+name] == digest for name, digest in map74.items())
    parent71 = json.loads(content74['parent71_sources.json'])
    assert len(parent71) == 71
    assert [name for name in parent71 if parent71[name] != map74[name]] == ['qwen_key_store.py']
    assert sha((HERE/'source74/before_qwen_key_store.py').read_bytes()) == parent71['qwen_key_store.py']
    assert map74['qwen_key_store.py'] == '96a20067b0034a4f9ecd394e2b56e29fe5389c996505230701d9cbd883a2d67d'
    config = load('focused83/run_config.json'); report = load('focused83/report.json')
    audit = load('focused83/worker-audit.json'); owner = load('focused83/owner-terminal.json')
    terminal = load('root/held-parent83-terminal-review.json')
    handoff = load('focused83/terminal-handoff.json')
    assert sha((HERE/'focused83/sources.json').read_bytes()) == report['source_manifest_sha256'] == config['source_manifest_sha256']
    assert sha((HERE/'focused83/sources.tar.gz').read_bytes()) == report['source_archive_sha256'] == config['source_archive_sha256']
    assert load('focused83/sources-after.json') == load('focused83/execution-after.json') == map83
    for name, key in [('executed_driver.py', 'driver_sha256'), ('worker_driver.py', 'worker_driver_sha256'), ('run_config.json', 'config_sha256')]:
        assert sha((HERE/'focused83'/name).read_bytes()) == report[key]
    xml = ET.parse(HERE/'focused83/tests.xml').getroot()
    counts = {name: len(list(xml.iter(tag))) for name, tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
    cases = [dict(classname=n.attrib['classname'], name=n.attrib['name']) for n in xml.iter('testcase')]
    assert counts == report['tests'] == owner['tests'] == dict(tests=34, failures=0, errors=0, skipped=0)
    assert cases == config['selected_xml_cases'] == report['selected_xml_cases']
    assert audit['collected_nodes'] == config['selected_nodes'] == report['selected_nodes']
    assert len(set(audit['collected_nodes'])) == 34
    assert audit['denied'] == audit['loaded_backend'] == []
    assert audit['numpy_imported'] is True and report['numpy_imported'] is True
    assert audit['runtime_sources_before'] == audit['runtime_sources_after'] == config['runtime_source_expectations']
    assert len(config['runtime_source_expectations']) == len(config['runtime_module_origins']) == 3
    assert audit['runtime_module_origins'] == {n: dict(file=p,spec=p) for n,p in config['runtime_module_origins'].items()}
    assert audit['forbidden_roots'] == config['forbidden_roots'] and 'numpy' not in audit['forbidden_roots']
    assert audit['affinity'] == [112,113,114,115]
    assert report['passed'] is True and report['sources_unchanged'] is True and report['synthetic_only'] is True
    assert type(report['child_returncode']) is int and report['child_returncode'] == 0
    assert report['child_reaped'] is True and report['child_group_absent'] is True
    assert all(report[n] is False for n in ['actual_execution_authorized','actual_key_generation','backend_context_construction','model_execution','encrypted_execution'])
    assert owner['passed'] is True and owner['owner_tool_exit_code'] == 0 and owner['owner_reaped'] is True
    assert owner['authoritative_direct_tool_chunk'] == '0a780d' and owner['owner_session'] is None
    assert owner['independent_live_observation'] is False
    assert all(all(row.values()) for row in owner['process_states'].values())
    for name, digest in owner['files'].items():
        local = 'sources.json' if name == 'sources-before.json' else name
        assert sha((HERE/'focused83'/local).read_bytes()) == digest
    assert sha((HERE/'focused83/report.json').read_bytes()) == terminal['report_sha256'] == handoff['report_sha256']
    assert sha((HERE/'focused83/owner-terminal.json').read_bytes()) == terminal['owner_receipt_sha256'] == handoff['owner_receipt_sha256']
    assert terminal['passed'] is True and terminal['normal_zero_reaped'] is True
    assert terminal['root_direct_reaping_claim'] is False and terminal['root_live_observation'] is False
    resource_index = load('resource/index.json')
    assert all(sha((HERE/'resource'/n).read_bytes()) == h for n,h in resource_index['files'].items())
    authority_map, authority_content = archive(Path('authority28'), 28)
    old28 = load('authority28/parent28_sources.json')
    assert [n for n in old28 if old28[n] != authority_map[n]] == ['keygen_boundary_pins.py']
    new_pin = authority_content['keygen_boundary_pins.py']
    old_pin = (HERE/'authority28/before_keygen_boundary_pins.py').read_bytes()
    assert sha(old_pin) == old28['keygen_boundary_pins.py']
    assert new_pin.replace(b'96a20067b0034a4f9ecd394e2b56e29fe5389c996505230701d9cbd883a2d67d',
        b'2e60908c9773debaa85a5d325df1367013ffb99c6ddead2f587dafca78efb0bd') == old_pin
    pins = next(ast.literal_eval(n.value) for n in ast.parse(new_pin).body
        if isinstance(n, ast.Assign) and n.targets[0].id == 'REQUIRED_ACTUAL_SOURCES')
    assert len(pins) == 58 and pins['qwen_key_store.py'] == map74['qwen_key_store.py']
    if args.current:
        assert inventory(Path(config['source_root'])) == map83
        assert inventory(Path('/tmp/qwen-key-generation-held-parent-20261104')) == map74
        assert {n: sha(Path(n).read_bytes()) for n in config['runtime_source_expectations']} == config['runtime_source_expectations']
    print(json.dumps(dict(passed=True, indexed_files=len(expected), source74=74, tested_source83=83,
        tests=counts, current_sources_checked=args.current, candidate_imports=False,
        actual_generation_authorized=False), sort_keys=True))


if __name__ == '__main__':
    main()
