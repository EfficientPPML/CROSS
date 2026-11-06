"""Portable public evidence readback; no producer imports or test execution."""
from pathlib import Path
import argparse, ast, hashlib, json, tarfile
import xml.etree.ElementTree as ET

P = Path(__file__).resolve().parent
SOURCE = Path('/tmp/qwen-key-generation-producer-v2-20261104')
HELD = Path('/tmp/qwen-key-generation-producer-20261104')
EXECUTION = Path('/tmp/qwen-key-producer-boundary-proof-20261104/preparation-v1/execution')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(name): return json.loads((P / name).read_bytes())
def canonical(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {p.relative_to(root).as_posix(): sha(p.read_bytes())
            for p in root.rglob('*') if p.is_file()}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--current', action='store_true')
    args = parser.parse_args()
    index = read('files.sha256.json'); actual = inventory(P)
    assert actual.pop('files.sha256.json') and actual == index
    whitelist = read('public-whitelist.json')
    assert set(whitelist['files']) == set(index) | {'files.sha256.json'}
    assert whitelist['private_or_synthetic_payload_bodies_included'] is False
    origins = read('copy-origins.json')
    for name, row in origins.items():
        raw = (P / name).read_bytes()
        assert sha(raw) == row['sha256'] and len(raw) == row['bytes']
    expected = read('measured/sources-before.json')
    assert len(expected) == 71
    assert expected == read('measured/sources-after.json') == read('measured/execution-after.json')
    assert sha((P / 'measured/sources-before.json').read_bytes()) == '6b8a842e72694d691ea2f4a2d9831331b14e929ad521d701fe445e85f939d874'
    with tarfile.open(P / 'measured/sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 71
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        blobs = {m.name: archive.extractfile(m).read() for m in members}
    assert {n: sha(b) for n, b in blobs.items()} == expected
    assert all(Path(n).suffix in ('.py', '.md', '.json') for n in blobs)
    for p in (P / 'source').rglob('*'):
        if p.is_file(): assert p.read_bytes() == blobs[p.relative_to(P / 'source').as_posix()]
    held = read('history/held64/source_manifest.json'); assert len(held) == 64
    changes = read('source/writer_successor.json')
    overlays = changes['overlays']
    assert set(overlays) == {'DESIGN.md', 'implementation_origin.json', 'qwen_key_store.py'}
    old = {n: ((P / 'history/held64' / n).read_bytes() if n in overlays else blobs[n]) for n in held}
    assert {n: sha(b) for n, b in old.items()} == held
    assert len(set(expected) - set(held)) == 7
    for name, row in overlays.items(): assert held[name] == row['before'] and expected[name] == row['after']
    assert {n for n in held if held[n] != expected[n]} == set(overlays)
    selected = read('source/selected_sources.json')['files']; assert len(selected) == 57
    assert all(expected[n] == held[n] == row['sha256'] for n, row in selected.items())
    assert expected['qwen_key_producer.py'] == held['qwen_key_producer.py']
    for name, digest in read('history/writer-plan/files.sha256.json').items():
        assert sha((P / 'history/writer-plan' / name).read_bytes()) == digest
    refs = read('history/writer-plan/source_refs.json')
    assert sha((P / 'history/held64/source_manifest.json').read_bytes()) == refs['held64_map_sha256'] == changes['held64_manifest_sha256']
    assert sha((P / 'history/writer-plan/proposed.patch.txt').read_bytes()) == refs['patch_sha256'] == changes['accepted_patch_sha256']
    literal = {}
    for node in ast.parse((P / 'history/writer-plan/make_proposal.py').read_bytes()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ('helper', 'anchor', 'before', 'after'):
                literal[node.targets[0].id] = ast.literal_eval(node.value)
    raw = old['qwen_key_store.py'].decode()
    assert raw.count(literal['anchor']) == raw.count(literal['before']) == 1
    reconstructed = raw.replace(literal['anchor'], literal['helper']+literal['anchor']).replace(literal['before'], literal['after']).encode()
    assert reconstructed == blobs['qwen_key_store.py']
    assert sha(reconstructed) == refs['proposed_store_sha256']
    for name, digest in read('plan/files.sha256.json').items():
        assert sha((P / 'plan' / name).read_bytes()) == digest
    plan = read('plan/public_plan.json')
    keyplan = read('source/public_inputs/inputs/actual_key_plan.json')
    assert plan['execution_authorized'] is False
    assert len(plan['public_array_inventory']) == 335
    assert plan['rotations'] == keyplan['combined_keys'] and len(plan['rotations']) == keyplan['combined_key_count'] == 166
    assert 65535 in plan['rotations'] and 262143 in plan['rotations']
    assert plan['rotation_plan']['sha256'] == sha(blobs['public_inputs/inputs/actual_key_plan.json'])
    auth = read('source/execution_authorization.json')
    assert all(auth[n] is False for n in ('actual_execution_enabled', 'backend_context_enabled', 'key_generation_enabled'))
    for filename, entry in [('qwen_key_store.py', 'create_actual_key_store'), ('qwen_key_producer.py', 'create_actual_key_producer')]:
        fn = next(n for n in ast.parse(blobs[filename]).body if isinstance(n, ast.FunctionDef) and n.name == entry)
        assert len(fn.body) == 1 and isinstance(fn.body[0], ast.Raise) and fn.body[0].exc.func.id == 'PermissionError'

    report, config, audit = read('measured/report.json'), read('measured/run_config.json'), read('measured/worker-audit.json')
    owner, live, handoff = read('measured/owner-terminal.json'), read('measured/live-observation.json'), read('measured/handoff.json')
    selection = read('source/selected_tests.json'); assert selection['expected_nodes'] == 30
    assert report['passed'] is True and report['sources_unchanged'] is True and report['source_count'] == 71
    assert report['tests'] == dict(tests=30, failures=0, errors=0, skipped=0)
    assert report['child_returncode'] == 0 and report['child_reaped'] is True and report['child_group_absent'] is True
    assert all(report[n] is False for n in ('actual_execution_authorized', 'actual_key_generation', 'backend_context_construction', 'model_execution', 'encrypted_execution'))
    assert owner['session_id'] == 72366 and owner['terminal_chunk'] == '903bc1'
    assert owner['exit_code'] == 0 and owner['reaped'] is True and all(owner['observed_absence'].values())
    assert owner['observed_processes'] == live
    for field, name in [('report_sha256', 'report.json'), ('source_manifest_sha256', 'sources-before.json'), ('source_archive_sha256', 'sources.tar.gz'), ('tests_xml_sha256', 'tests.xml'), ('audit_sha256', 'worker-audit.json'), ('driver_sha256', 'executed_driver.py'), ('worker_sha256', 'worker_driver.py'), ('config_sha256', 'run_config.json')]:
        assert owner[field] == handoff[field] == sha((P / 'measured' / name).read_bytes())
    assert handoff['owner_receipt_sha256'] == sha((P / 'measured/owner-terminal.json').read_bytes())
    assert report['source_manifest_sha256'] == config['source_manifest_sha256'] == owner['source_manifest_sha256']
    assert report['source_archive_sha256'] == config['source_archive_sha256'] == owner['source_archive_sha256']
    assert audit['denied'] == audit['loaded_backend'] == [] and audit['pytest_exit_code'] == 0
    assert audit['affinity'] == [108,109,110,111] and audit['pid'] == live['worker_pid'] and audit['pgid'] == live['worker_pgid']
    assert audit['collected_nodes'] == report['selected_nodes'] == config['selected_nodes'] == selection['nodes']
    assert sha(canonical(config['selected_nodes'])) == config['selected_nodes_sha256']
    assert audit['runtime_sources_before'] == audit['runtime_sources_after'] == config['runtime_source_expectations']
    assert config['runtime_source_expectations'] == read('source/harness_provenance.json')['runtime_source_expectations']
    assert len(config['runtime_source_expectations']) == 3
    assert audit['runtime_module_origins'] == {n: dict(file=p,spec=p) for n,p in config['runtime_module_origins'].items()}
    xml = ET.parse(P / 'measured/tests.xml').getroot()
    cases = [dict(classname=c.attrib['classname'],name=c.attrib['name']) for c in xml.iter('testcase')]
    assert len(cases) == 30 and cases == config['selected_xml_cases'] == report['selected_xml_cases']
    assert sha(canonical(cases)) == config['selected_xml_cases_sha256']
    assert all(not list(xml.iter(n)) for n in ('failure','error','skipped'))
    assert {name: sum(c['classname'] == name[:-3] for c in cases) for name in selection['files']} == selection['per_file']

    root = read('reviews/root/terminal-review.json')
    assert root['passed'] is True and root['source_count'] == 71 and root['tests'] == 30
    for name, digest in root['pins'].items(): assert sha((P / 'measured' / name).read_bytes()) == digest
    for field, name in [('launch_review_sha256','launch-review.json'), ('source_review_sha256','successor-source-review.json'), ('root_live_sha256','live-observation.json'), ('script_sha256','verify_terminal.py')]:
        assert root[field] == sha((P / 'reviews/root' / name).read_bytes())
    launch = read('reviews/root/launch-review.json')
    assert launch['passed'] and launch['synthetic_launch_authorized'] and launch['actual_execution_authorized'] is False
    assert launch['security_review_sha256'] == sha((P / 'reviews/security/successor71-source-review.json').read_bytes())
    for name, digest in launch['pins'].items(): assert sha((P / 'measured' / name).read_bytes()) == digest
    for label, oldmap in [('held64-source-review.json',held),('successor71-source-review.json',expected)]:
        for name,digest in read('reviews/security/'+label)['reviewed_sources'].items(): assert oldmap[name] == digest
    if args.current:
        assert inventory(SOURCE) == inventory(EXECUTION) == expected and inventory(HELD) == held
        for row in origins.values(): assert sha(Path(row['source']).read_bytes()) == row['sha256']
        for row in selected.values(): assert sha(Path(row['path']).read_bytes()) == row['sha256']
        for name, digest in config['runtime_source_expectations'].items(): assert sha(Path(name).read_bytes()) == digest
    print(json.dumps(dict(passed=True,indexed_files=len(index),source_count=71,tests=30,
        writer_tests=15,recording_producer_tests=15,current=args.current,
        actual_execution_authorized=False,scope='Public evidence verification only; no test/backend/actual body execution')))

if __name__ == '__main__': main()
