"""Verify public readiness and launch evidence only; never execute the grant."""
from pathlib import Path
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
read = lambda n: json.loads((ROOT / n).read_bytes())
index = read('files.sha256.json')
assert set(index) == {p.name for p in ROOT.iterdir() if p.is_file()} - {'files.sha256.json'}
for name, pin in index.items():
    assert Path(name).name == name and not (ROOT / name).is_symlink()
    assert sha((ROOT / name).read_bytes()) == pin, name
assert index['preflight-report.json'] == '75d8bf2225337ae9dea63e59cd6d1fdd6ff59a7d024b96c988ae2c0185cecad7'
assert index['sources.tar.gz'] == 'ccefaa4c2ec41371660aecbbb618f67951b6a144b898a20fe04a743db15e4654'
assert index['root-terminal-review.json'] == '578cca4a259ad432c012708d0cbd35ca40cfbc4e5f958ae6bc34a28511e2bb9c'
assert index['root-launch.json'] == 'c2493b48d5e38462b652004f7b7e856b4f20af6c9b83840dacce43843caecaa0'
assert index['owner-launch.json'] == 'df7a72e1747384a9d35a8b9f7966214a92bb6ef07cbf41eb1d5c0373ef20368d'
m, old = read('source_manifest.json'), read('parent185_source_manifest.json')
assert read('execution_source_manifest.json') == m
assert len(m) == 187 and len(old) == 185 and set(old) <= set(m)
assert index['source_manifest.json'] == 'e0bfcee1ab85f8750e576de24bdbdf13faf4f0b8abe66ecef55b3403a084cce0'
assert index['parent185_source_manifest.json'] == 'aa5e823a116a9d64353b9112e9e09ba03641cd2acd420f4ed8d7b8e3fac2fa8c'
review = read('root-source-review.json')
assert {n for n in old if m[n] != old[n]} == set(review['metadata_changed'] + review['test_changed'])
assert set(m) - set(old) == set(review['added'])
assert review['runtime_changes'] == [] and review['legacy_fixture_server_enabled'] is False
with tarfile.open(ROOT / 'sources.tar.gz', 'r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == len({v.name for v in members}) == 187
    assert all(v.isfile() and v.name in m for v in members)
    assert {v.name: sha(archive.extractfile(v).read()) for v in members} == m
    legacy = json.loads(archive.extractfile('execution_authorization.json').read())
    ready = json.loads(archive.extractfile('validator_authorization.json').read())
    pins = json.loads(archive.extractfile('producer_bindings.json').read())
    assert legacy == {'ready': False}
    assert ready == dict(schema='gain4-validator-only-authorization-v1', ready=True,
                         stage='validate', execution_authorized=True)
    assert pins['server_report_sha256'] == '897fa13c99005dff41d10e192d87417a6d31a155995c3e65805922fc5326d68a'
    assert pins['server_root_review_sha256'] == m['producer_server_root_review.json'] == '727345f7d182d5b65fafa73d057c594ba69c1b427a960a5169676aaf3e679a98'
r = read('preflight-report.json')
assert r['execute'] is False and r['source_hashes_start'] == r['source_hashes_end'] == m
for key in ('passed', 'source_files_unchanged', 'source_archive_verified',
            'all_direct_children_reaped', 'all_owned_groups_empty',
            'process_cleanup_completed', 'terminal_scope_completed',
            'termination_scope_completed', 'no_encrypted_execution',
            'no_private_or_ciphertext_payload_read'):
    assert r[key] is True, key
assert r['cleanup_errors'] == []
for name, count in [('mock_tests', 543), ('installed_denials', 10)]:
    assert r[name] == dict(tests=count, errors=0, failures=0, skipped=0, exit_code=0)
    xml = ET.parse(ROOT / (name + '.xml'))
    assert len(xml.findall('.//testcase')) == count
    assert not any(xml.findall('.//' + tag) for tag in ('error', 'failure', 'skipped'))
guard = r['fresh_source_guard']
assert guard['passed'] is True and guard['fresh'] is True and guard['source_count'] == 187
assert guard['manifest_sha256'] == index['source_manifest.json']
owner, root = read('owner-terminal.json'), read('root-terminal-review.json')
assert (owner['outer_session'], owner['outer_tool_chunk'], owner['outer_exit_code'], owner['outer_reaped']) == (31093, '510590', 0, True)
assert root['passed'] is True and root['owner_terminal_sha256'] == index['owner-terminal.json']
assert root['report_sha256'] == index['preflight-report.json']
grant = read('root-launch.json')
assert grant['authorized'] is True and grant['phase'] == 'validate' and grant['source_count'] == 187
assert grant['source_manifest_sha256'] == index['source_manifest.json']
assert grant['readiness_review_sha256'] == index['root-terminal-review.json']
assert grant['healthy_deadline'] is None and grant['cpu_affinity'] == [124, 125, 126, 127]
print(json.dumps({'passed': True, 'sources': 187, 'synthetic_tests': 543,
                  'installed_checks': 10, 'scope': 'Readiness and launch only; numerical result outside this capsule.'}))
