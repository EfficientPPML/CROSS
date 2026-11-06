"""Read public evidence only; never import producer code or follow data descriptors."""
from pathlib import Path
import argparse, ast, hashlib, json, tarfile
import xml.etree.ElementTree as ET

P = Path(__file__).resolve().parent
SOURCE = Path('/tmp/qwen-client-stage-glue-v2-20261104')
EXECUTION = Path('/tmp/qwen-client-stage-boundary-proof-20261104/development-v1/execution')
HELD = Path('/tmp/qwen-client-stage-glue-20261104')
PARENT = Path('/tmp/qwen-organizer-file-engine-integration-v2-20261104')
def sha(b): return hashlib.sha256(b).hexdigest()
def read(n): return json.loads((P / n).read_bytes())
def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {p.relative_to(root).as_posix(): sha(p.read_bytes())
            for p in root.rglob('*') if p.is_file()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', action='store_true')
    args = parser.parse_args()
    index = read('files.sha256.json')
    observed = inventory(P)
    assert observed.pop('files.sha256.json')
    assert observed == index, 'capsule inventory mismatch'
    whitelist = read('public-whitelist.json')
    assert set(whitelist['files']) == set(index) | {'files.sha256.json'}
    assert whitelist['private_or_synthetic_payload_bodies_included'] is False
    origins = read('copy-origins.json')
    for name, row in origins.items():
        raw = (P / name).read_bytes()
        assert sha(raw) == row['sha256'] and len(raw) == row['bytes']

    expected = read('measured/sources-before.json')
    assert len(expected) == 373
    assert expected == read('measured/sources-after.json') == read('measured/execution-after.json')
    assert sha((P / 'measured/sources-before.json').read_bytes()) == '7837d7c4d2a3d23ca77c7d3ff58052b45b4cda309f5cb6c234afb08fa5772f1b'
    with tarfile.open(P / 'measured/sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 373
        assert all(m.isfile() and not Path(m.name).is_absolute()
                   and '..' not in Path(m.name).parts for m in members)
        blobs = {m.name: archive.extractfile(m).read() for m in members}
    assert {n: sha(b) for n, b in blobs.items()} == expected
    assert all(Path(n).suffix in ('.py', '.md', '.json') for n in expected)
    for p in (P / 'source').iterdir():
        assert p.read_bytes() == blobs[p.name]
    parent = read('source/client_parent366_manifest.json')
    held = read('source/client_parent370_manifest.json')
    assert len(parent) == 366 and len(held) == 370
    assert all(expected[n] == v for n, v in held.items())
    assert all(held[n] == v for n, v in parent.items())
    assert set(held)-set(parent) == {'client_stage.py', 'CLIENT_STAGE_DESIGN.md',
                                   'client_stage_origin.json', 'client_parent366_manifest.json'}
    assert set(expected)-set(held) == {'test_client_stage.py', 'client_parent370_manifest.json',
                                     'client_test_origin.json'}
    origin = read('source/client_stage_origin.json')
    assert origin['overlays'] == {} and origin['actual_execution_authorized'] is False
    assert origin['parent_manifest_sha256'] == sha(blobs['client_parent366_manifest.json'])
    assert expected['runtime/jaxite_word/ckks_ctx.py'] == origin['selected_default_decoder_sha256']
    assert expected['cleanup_origin.py'] == origin['shared_cleanup_origin_sha256']
    assert expected['qwen_finalizer.py'] == origin['shared_finalizer_sha256']
    definitions = {n.name: n for n in ast.parse(blobs['client_stage.py']).body
                   if isinstance(n, ast.FunctionDef)}
    for name in ('create_client_input', 'create_client_output'):
        body = definitions[name].body
        assert len(body) == 1 and isinstance(body[0], ast.Raise)
        assert isinstance(body[0].exc, ast.Call) and body[0].exc.func.id == 'PermissionError'

    report = read('measured/report.json')
    owner = read('measured/owner-terminal.json')
    handoff = read('measured/handoff.json')
    audit = read('measured/worker-audit.json')
    live = read('measured/live-observation.json')
    assert report['passed'] is True and report['sources_unchanged'] is True
    assert report['tests'] == dict(tests=19, failures=0, errors=0, skipped=0)
    assert report['child_returncode'] == 0 and report['child_reaped'] is True
    assert report['child_group_absent'] is True and report['source_count'] == 373
    assert report['synthetic_only'] is True
    assert all(report[n] is False for n in ('model_execution', 'encrypted_execution', 'actual_role_execution'))
    assert owner['session_id'] == 30266 and owner['terminal_chunk'] == 'a84cf9'
    assert owner['exit_code'] == 0 and owner['reaped'] is True
    assert all(v is True for v in owner['observed_absence'].values())
    assert owner['observed_processes'] == live
    for k, path in [('report_sha256', 'measured/report.json'),
                    ('source_manifest_sha256', 'measured/sources-before.json'),
                    ('source_archive_sha256', 'measured/sources.tar.gz')]:
        assert owner[k] == handoff[k] == sha((P / path).read_bytes())
        if k != 'report_sha256': assert report[k] == owner[k]
    assert handoff['owner_receipt_sha256'] == sha((P / 'measured/owner-terminal.json').read_bytes())
    for key, name in [('driver_sha256', 'executed_driver.py'), ('worker_driver_sha256', 'worker_driver.py')]:
        assert report[key] == handoff[key] == sha((P / 'measured' / name).read_bytes())
    assert audit['denied'] == audit['loaded_backend'] == []
    assert audit['pytest_exit_code'] == 0 and audit['affinity'] == [108, 109, 110, 111]
    assert audit['pid'] == live['worker_pid'] and audit['pgid'] == live['worker_pgid']
    xml = ET.parse(P / 'measured/tests.xml').getroot()
    assert len(list(xml.iter('testcase'))) == 19
    assert all(not list(xml.iter(n)) for n in ('failure', 'error', 'skipped'))
    nodes = ['test_client_stage.py::' + c.attrib['name'] for c in xml.iter('testcase')]
    assert nodes == report['selected_nodes'] == read('source/client_test_origin.json')['selected_nodes']

    root = read('reviews/root/terminal-review.json')
    assert root['passed'] is True and root['tests'] == 19 and root['source_count'] == 373
    assert root['script_sha256'] == sha((P / 'reviews/root/verify_terminal.py').read_bytes())
    for path, digest in root['pins'].items():
        base = Path(path)
        local = ('reviews/root/' if base.parent.name == 'qwen-client-stage-root-review-20261104'
                 else 'measured/') + base.name
        assert sha((P / local).read_bytes()) == digest
    source_review = read('reviews/root/source-review.json')
    assert source_review['passed'] is True and source_review['runtime_sha256'] == expected['client_stage.py']
    assert source_review['security_review_sha256'] == sha((P / 'reviews/security/review.json').read_bytes())
    assert source_review['source_manifest_sha256'] == sha(blobs['client_parent370_manifest.json'])
    assert source_review['script_sha256'] == sha((P / 'reviews/root/verify_source.py').read_bytes())
    refs = read('parent-reference.json')
    assert refs['source_count'] == 366 and refs['tests'] == 12
    assert refs['files']['measured/source_manifest.json']['sha256'] == sha(blobs['client_parent366_manifest.json'])
    assert refs['files']['reviews/root/stage-successor-review.json']['sha256'] == origin['parent_review_sha256']
    if args.current:
        assert inventory(SOURCE) == inventory(EXECUTION) == expected
        assert inventory(HELD) == held and inventory(PARENT) == parent
        for row in origins.values(): assert sha(Path(row['source']).read_bytes()) == row['sha256']
        for row in refs['files'].values():
            assert sha((P / row['relative_capsule_path']).read_bytes()) == row['sha256']
    print(json.dumps(dict(passed=True, indexed_files=len(index), source_count=373,
                         unchanged_parent370=True, tests=19, backend_imports=0,
                         current=args.current, actual_execution_authorized=False,
                         scope='Public evidence verification only; no tests/backend or data descriptors executed')))

if __name__ == '__main__': main()
