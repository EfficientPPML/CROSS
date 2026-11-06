"""Offline public capsule verification; no runtime or external payload reads."""
from pathlib import Path
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def read(name):
    return json.loads((ROOT / name).read_bytes())


index = read('files.sha256.json')
assert set(index) == {p.name for p in ROOT.iterdir() if p.is_file()} - {'files.sha256.json'}
for name, pin in index.items():
    assert Path(name).name == name and not (ROOT / name).is_symlink()
    assert sha((ROOT / name).read_bytes()) == pin, name

assert index['report.json'] == '897fa13c99005dff41d10e192d87417a6d31a155995c3e65805922fc5326d68a'
assert index['worker_report.json'] == 'e3eb179778ac962b624842010391865c77de56a513ca1b5de3fdc97fdf9d22d1'
assert index['sources.tar.gz'] == '4ab405b63a86998082acbacfd0d642b2a6371e286688939ba3c88131985f41ea'
assert index['root-review.json'] == '727345f7d182d5b65fafa73d057c594ba69c1b427a960a5169676aaf3e679a98'
assert index['owner-terminal.json'] == '21bf021debeb2dc23321b16fc978e5e7d7b3f0ebd1ef9bfd3a6cff5f54156ca1'

r, w = read('report.json'), read('worker_report.json')
m = read('source_manifest.json')
assert len(m) == 163
assert r['source_hashes_start'] == r['source_hashes_end'] == m
assert w['source_hashes_start'] == w['source_hashes_end'] == m
with tarfile.open(ROOT / 'sources.tar.gz', 'r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == len({p.name for p in members}) == 163
    assert all(p.isfile() and p.name in m for p in members)
    assert {p.name: sha(archive.extractfile(p).read()) for p in members} == m
assert r['worker_report_sha256'] == index['worker_report.json']
assert r['source_archive_sha256'] == index['sources.tar.gz']
assert r['stage'] == w['stage'] == 'server'
assert r['execute'] is True and type(r['worker_exit_code']) is int and r['worker_exit_code'] == 0
for key in ('passed', 'source_archive_verified', 'source_files_unchanged',
            'terminal_scope_completed', 'termination_scope_completed',
            'process_cleanup_completed', 'all_direct_children_reaped',
            'all_owned_groups_empty', 'worker_reaped', 'worker_group_empty_verified'):
    assert r[key] is True, key
for key in ('passed', 'phase_complete', 'source_files_unchanged', 'loader_entered',
            'loader_cleanup_completed', 'terminal_scope_completed',
            'termination_scope_completed', 'all_inputs_authenticated_before_server'):
    assert w[key] is True, key
assert r['cleanup_errors'] == [] and w['forbidden_attempts'] == []
assert r['request_contract'] == w['request_contract']
assert r['readback'] == w['readback']
assert r['readback']['tile_count'] == w['completed_ciphertexts'] == 54
assert [v['index'] for v in r['readback']['manifest']['tiles']] == list(range(54))
for field in ('loaded_local_modules_start', 'loaded_local_modules_end'):
    for item in w[field].values():
        assert item['sha256'] == m[item['path']]
a = w['arithmetic']
assert a['checkpoints_completed'] == 54 and a['refreshes'] == 0
assert a['codecs_called'] is False and a['keygen'] is False
assert a['synchronous_outputs'] is True and a['input_metadata_unchanged'] is True
assert len(a['cases']) == 2
for case, (name, keys, updates) in zip(a['cases'], [('K128_rows16', 128, 13), ('K15_rows9', 15, 10)]):
    assert (case['name'], case['key_count'], case['updates']) == (name, keys, updates)
    assert case['level_reduction_calls'] == case['diagnostic_ciphertext_products'] == case['diagnostic_rescales'] == 1
    assert case['production_counts'] == dict(ciphertext_additions=14, ciphertext_products=2 * updates,
                                             plaintext_products=3, rotations=14)
root, owner = read('root-review.json'), read('owner-terminal.json')
assert root['passed'] is True and root['fixture_server_terminal_gates_passed'] is True
assert root['numerical_validation_performed'] is False and root['actual_validator_launch_authorized'] is False
assert root['server_report_sha256'] == owner['persisted_parent_report_sha256'] == index['report.json']
assert root['owner_terminal_sha256'] == index['owner-terminal.json']
assert (owner['session_id'], owner['tool_chunk'], owner['exit_code'], owner['reaped']) == (27546, 'ab88e7', 0, True)
assert owner['numerical_validation_performed'] is False
assert root['owned_worker_group_absent'] is True and owner['owned_worker_group_absent'] is True
print(json.dumps({'passed': True, 'sources': 163, 'server_checkpoints': 54,
                  'scope': 'Offline server completion evidence only; numerical validation not performed.'}))
