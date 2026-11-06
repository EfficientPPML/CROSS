"""Read public capsule evidence only; never import the candidate or run tests."""
from pathlib import Path
import argparse, ast, difflib, hashlib, json, tarfile
import xml.etree.ElementTree as ET

P = Path(__file__).resolve().parent
SOURCE = Path('/tmp/qwen-model-stage-aggregate-tests-20261104')
PARENT = Path('/tmp/qwen-model-stage-aggregate-20261104')
EXECUTION = Path('/tmp/qwen-model-stage-boundary-proof-20261104/preparation-v1/execution')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(name): return json.loads((P/name).read_bytes())
def canonical(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {p.relative_to(root).as_posix(): sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
def digest(name): return sha((P/name).read_bytes())
def plan_bytes(version, name):
    direct = P/'plans'/version/name
    if direct.is_file(): return direct.read_bytes()
    assert version == 'v2'
    row = read('plans/v2/shared-files.json')[name]
    assert row['capsule_path'] == 'plans/original/'+name
    raw = (P/row['capsule_path']).read_bytes()
    assert sha(raw) == row['sha256']
    return raw

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
        raw = (P/name).read_bytes()
        assert sha(raw) == row['sha256'] and len(raw) == row['bytes']
    expected = read('measured/sources-before.json')
    assert len(expected) == 38
    assert expected == read('measured/sources-after.json') == read('measured/execution-after.json')
    assert digest('measured/sources-before.json') == '30ab709ada68471e65767357808398990cbb36329bb29650d91fc0b1a75a3f1a'
    assert digest('measured/sources.tar.gz') == 'b456d1148ee8d2abe7e12444e5f9e3cdf2a806df7809771cf8b90c2d62780286'
    with tarfile.open(P/'measured/sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 38
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        blobs = {m.name: archive.extractfile(m).read() for m in members}
    assert {n: sha(b) for n,b in blobs.items()} == expected
    assert all(Path(n).suffix in ('.py','.json','.txt') for n in blobs)
    for p in (P/'source').iterdir():
        assert p.is_file() and p.read_bytes() == blobs[p.name]
    parent = read('history/parent29_sources.json')
    assert len(parent) == 29 and all(expected[n] == h for n,h in parent.items())
    assert len(set(expected)-set(parent)) == 9
    assert json.loads(blobs['parent29_sources.json']) == parent
    assert digest('history/parent29_sources.json') == '613be1bd3e69ea5284882a2241913c02bbbbcdf11665e04fc42db0801d30ee6a'
    origin = read('source/source_origin.json'); overlay = origin['declared_overlay']
    assert sha(blobs[overlay['path']]) == overlay['after_sha256']
    assert digest('history/selected366_qwen_public_parameters.py') == overlay['before_sha256']
    assert digest('source/parameter_identity.patch.txt') == overlay['patch_sha256']
    reconstructed_patch = ''.join(difflib.unified_diff(
        (P/'history/selected366_qwen_public_parameters.py').read_text().splitlines(True),
        blobs['qwen_public_parameters.py'].decode().splitlines(True),
        fromfile='selected366/qwen_public_parameters.py', tofile='model-stage-successor/qwen_public_parameters.py'))
    assert reconstructed_patch.encode() == blobs['parameter_identity.patch.txt']
    for name,row in origin['base_files'].items():
        assert (overlay['before_sha256'] if name == overlay['path'] else expected[name]) == row['sha256']
    assert all(expected[n] == h for n,h in origin['new_modules'].items())
    for version in ('original','v2'):
        for name,h in read('plans/'+version+'/files.json')['files'].items(): assert sha(plan_bytes(version,name)) == h
    assert sha(plan_bytes('v2','PLAN.md')) == origin['accepted_plan']['sha256']
    for name,h in origin['schema_inputs'].items(): assert sha(plan_bytes('v2',name)) == sha(blobs[name]) == h
    note = read('history/static-count-note.json')
    assert note['source_count'] == 29 and note['tests'] == note['candidate_imports'] == note['bodies'] == 0
    auth = read('source/execution_authorization.json')
    assert all(auth[n] is False for n in ('actual_model_validation_enabled','actual_parameter_factory_enabled','backend_context_enabled','model_body_reads_authorized','model_forward_enabled'))
    for file,name in [('model_admission.py','create_actual_model_admission'),('model_store_validation.py','create_actual_model_validator'),('qwen_public_parameters.py','open_actual_admitted_parameters')]:
        fn = next(n for n in ast.parse(blobs[file]).body if isinstance(n,ast.FunctionDef) and n.name == name)
        assert len(fn.body) == 1 and isinstance(fn.body[0],ast.Raise) and fn.body[0].exc.func.id == 'PermissionError'

    report, config, audit = read('measured/report.json'), read('measured/run_config.json'), read('measured/worker-audit.json')
    owner, live, handoff = read('measured/owner-terminal.json'), read('measured/live-observation.json'), read('measured/handoff.json')
    selection = read('source/selected_model_tests.json')
    assert selection['expected_nodes'] == 48 and selection['per_file'] == {'test_model_admission.py':28,'test_held_identity.py':10,'test_model_recording.py':10}
    assert report['passed'] is True and report['sources_unchanged'] is True and report['source_count'] == 38
    assert report['tests'] == dict(tests=48,failures=0,errors=0,skipped=0)
    assert report['child_returncode'] == 0 and report['child_reaped'] is True and report['child_group_absent'] is True
    assert all(report[n] is False for n in ('actual_execution_authorized','actual_key_generation','backend_context_construction','model_execution','encrypted_execution'))
    assert owner['session_id'] == 75704 and owner['terminal_chunk'] == 'f20b6d'
    assert owner['exit_code'] == 0 and owner['reaped'] is True and all(owner['observed_absence'].values())
    assert owner['observed_processes'] == live
    for field,name in [('report_sha256','report.json'),('source_manifest_sha256','sources-before.json'),('source_archive_sha256','sources.tar.gz'),('tests_xml_sha256','tests.xml'),('audit_sha256','worker-audit.json'),('driver_sha256','executed_driver.py'),('worker_sha256','worker_driver.py'),('config_sha256','run_config.json')]:
        assert owner[field] == handoff[field] == digest('measured/'+name)
    assert handoff['owner_receipt_sha256'] == digest('measured/owner-terminal.json')
    assert report['source_manifest_sha256'] == config['source_manifest_sha256'] == owner['source_manifest_sha256']
    assert report['source_archive_sha256'] == config['source_archive_sha256'] == owner['source_archive_sha256']
    assert audit['denied'] == audit['loaded_backend'] == [] and type(audit['pytest_exit_code']) is int and audit['pytest_exit_code'] == 0
    assert audit['affinity'] == [108,109,110,111] and audit['pid'] == live['worker_pid'] and audit['pgid'] == live['worker_pgid']
    assert audit['collected_nodes'] == report['selected_nodes'] == config['selected_nodes'] == selection['nodes']
    assert len(set(config['selected_nodes'])) == 48 and sha(canonical(config['selected_nodes'])) == config['selected_nodes_sha256']
    refs = read('source/numpy_source_refs.json')
    runtime = {row['path']:row['sha256'] for row in refs['expected_loaded_modules'].values()}
    runtime[refs['additional_source']['path']] = refs['additional_source']['sha256']
    assert len(runtime) == 4 and audit['runtime_sources_before'] == audit['runtime_sources_after'] == config['runtime_source_expectations'] == runtime
    assert len(config['runtime_module_origins']) == 3
    assert config['runtime_module_origins'] == {n:r['path'] for n,r in refs['expected_loaded_modules'].items()}
    assert audit['runtime_module_origins'] == {n:dict(file=p,spec=p) for n,p in config['runtime_module_origins'].items()}
    xml = ET.parse(P/'measured/tests.xml').getroot()
    cases = [dict(classname=c.attrib['classname'],name=c.attrib['name']) for c in xml.iter('testcase')]
    assert len(cases) == 48 and cases == config['selected_xml_cases'] == report['selected_xml_cases']
    assert sha(canonical(cases)) == config['selected_xml_cases_sha256']
    assert all(not list(xml.iter(n)) for n in ('failure','error','skipped'))
    assert {name:sum(c['classname']==name[:-3] for c in cases) for name in selection['files']} == selection['per_file']
    assert read('measured/source-handoff.json')['selection'] == selection

    root = read('reviews/root/terminal-review.json')
    assert digest('reviews/root/terminal-review.json') == '5c1341990f86b2535eced6f267d3ee7fff8cdfc7abbab797006ce4de063ec3c6'
    assert root['passed'] is True and root['source_count'] == 38 and root['tests'] == 48 and root['unchanged_parent_sources'] == 29
    for name,h in root['pins'].items(): assert digest('measured/'+name) == h
    for field,name in [('launch_review_sha256','focused-launch-review.json'),('source_review_sha256','test-source-review.json'),('root_live_sha256','live-observation.json'),('script_sha256','verify_terminal.py')]:
        assert root[field] == digest('reviews/root/'+name)
    launch = read('reviews/root/focused-launch-review.json')
    assert launch['passed'] and launch['synthetic_launch_authorized'] and launch['actual_execution_authorized'] is False
    assert launch['peer_review_sha256'] == digest('reviews/transformers/test-source-review.json')
    for name,h in launch['pins'].items(): assert digest('measured/'+name) == h
    rootlive = read('reviews/root/live-observation.json')
    assert rootlive['passed'] and rootlive['live'] == live and rootlive['live_sha256'] == digest('measured/live-observation.json')
    assert owner['root_live_observation_sha256'] == root['root_live_sha256']
    assert owner['root_launch_review_sha256'] == root['launch_review_sha256']
    assert set(rootlive['processes']) == {str(live['outer_pid']),str(live['worker_pid'])}
    for name,script in [('review.json','verify_plan.py'),('clarification-review.json','verify_clarification.py'),('implementation-review.json','verify_implementation.py'),('test-source-review.json','verify_test_source.py'),('focused-launch-review.json','verify_focused_launch.py')]:
        assert read('reviews/root/'+name)['script_sha256'] == digest('reviews/root/'+script)
    for name,script,field,m in [('review.json','review_source.py','source_hashes',parent),('test-source-review.json','review_test_source.py','source_hashes_after',expected)]:
        peer = read('reviews/transformers/'+name)
        assert peer['passed'] and peer[field] == m and peer['review_script_sha256'] == digest('reviews/transformers/'+script)
    assert read('reviews/transformers/test-source-review.json')['static_nodes'] == selection['nodes']
    assert read('reviews/root/review.json')['plan_index_sha256'] == digest('plans/original/files.json')
    assert read('reviews/root/clarification-review.json')['plan_index_sha256'] == digest('plans/v2/files.json')
    if args.current:
        assert inventory(SOURCE) == inventory(EXECUTION) == expected and inventory(PARENT) == parent
        for row in origins.values(): assert sha(Path(row['source']).read_bytes()) == row['sha256']
        for row in origin['base_files'].values(): assert sha(Path(row['path']).read_bytes()) == row['sha256']
        for name,h in runtime.items(): assert sha(Path(name).read_bytes()) == h
    print(json.dumps(dict(passed=True,indexed_files=len(index),source_count=38,unchanged_parent_sources=29,
        tests=48,admission_tests=28,small_file_identity_tests=10,recording_tests=10,current=args.current,
        actual_execution_authorized=False,body_integrity_admitted=False,
        scope='Public evidence verification only; no tests, producer imports or body descriptors followed.')))

if __name__ == '__main__': main()
