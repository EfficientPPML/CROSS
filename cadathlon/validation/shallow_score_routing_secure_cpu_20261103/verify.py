"""Verify archived source bytes and recorded numerical gates without HE execution."""
from pathlib import Path
import hashlib
import json
import math
import tarfile

HERE = Path(__file__).resolve().parent
ARCHIVE_SHA = 'bed0a8d807f7be54b2bed8d0f79286def5de810e847d819550a56d209999ada9'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    manifest = json.loads((HERE / 'artifacts.json').read_text())
    for name, expected in manifest.items():
        assert sha((HERE / name).read_bytes()) == expected, name
    r = json.loads((HERE / 'report.json').read_text())
    assert r['passed'] and r['fixed_absolute_error_gate'] == 1e-6
    before = r['source_hashes_before']
    assert len(before) == 99 and before == r['source_hashes_after']
    assert r['source_files_unchanged'] and r['source_archive_members_verified']
    assert sha((HERE / 'sources.tar.gz').read_bytes()) == ARCHIVE_SHA == r['source_archive_sha256']
    with tarfile.open(HERE / 'sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len(before) == len({m.name for m in members})
        assert all(m.isfile() for m in members)
        assert {m.name: sha(archive.extractfile(m).read()) for m in members} == before
    levels = {'head compaction': 4, 'key placement': 3, 'key extraction': 2, 'head expansion': 1}
    assert set(r['checks']) == {f'{gain}:{name}' for gain in ('1.0', '0.125') for name in levels}
    for name, c in r['checks'].items():
        assert c['passed'] and c['slots_checked'] == 65536 and c['metadata_ok']
        assert math.isfinite(c['max_abs_error']) and 0 <= c['max_abs_error'] <= 1e-6
        assert type(c['nsd']) is int and c['nsd'] == 1 and c['level'] == levels[name.split(':')[1]]
        assert c['scale'] == r['ring']['scale']
        assert len(c['execution_rings']) == 1
        security = c['execution_rings'][0]
        assert security['degree'] == 131072 and security['security_bits'] == 128
        assert security['q_towers'] == r['ring']['q_towers'] and security['p_towers'] == r['ring']['p_towers']
    ring = r['ring']
    q, p = ring['q_towers'], ring['p_towers']
    assert (ring['degree'], ring['depth'], ring['dnum'], len(q), len(p)) == (131072, 5, 4, 12, 3)
    assert math.prod(q + p).bit_length() <= 3523
    alpha = math.ceil(len(q) / ring['dnum'])
    assert all(math.prod(q[i:i + alpha]) <= math.prod(p) for i in range(0, len(q), alpha))
    assert len(ring['rotation_indices']) == 54
    assert set(r['key_generation_guards']) == {'jaxite_word.key_gen', 'key_gen'}
    assert set(r['class_codec_guards']) == {'encrypt', 'decrypt', 'encrypt_slots', 'decrypt_slots', 'encrypt_slots_batch', 'decrypt_slots_batch'}
    assert r['server_complete_before_client_validation'] and r['keyless_server']
    assert r['input_payload_unchanged'] and r['input_metadata_unchanged']
    assert not r['cleanup_errors'] and r['public_key_store_removed']
    terminal = json.loads((HERE / 'terminal.json').read_text())
    assert terminal['exit_code'] == 0 and terminal['session_reaped']
    assert terminal['process_absent'] and terminal['owned_key_directory_absent']
    return {'passed': True, 'source_files': len(before), 'slot_checks': 8 * 65536,
            'worst_max_abs_error': max(c['max_abs_error'] for c in r['checks'].values()),
            'scope': 'Offline artifact and recorded-gate verification; no encrypted rerun.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
