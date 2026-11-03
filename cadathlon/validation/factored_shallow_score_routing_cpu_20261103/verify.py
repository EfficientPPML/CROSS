"""Offline integrity, recorded-test and independent public key-plan checks."""
from pathlib import Path
import hashlib
import json
import math
import tarfile

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify():
    for name, expected in json.loads((HERE / 'artifacts.json').read_text()).items():
        assert sha((HERE / name).read_bytes()) == expected, name
    r = json.loads((HERE / 'report.json').read_text())
    assert r['passed'] and r['tests_exit_code'] == r['plan_exit_code'] == 0
    assert r['source_files_unchanged'] and r['source_archive_verified']
    h = r['source_hashes_before']
    assert len(h) == 10 and h == r['source_hashes_after']
    assert sha((HERE / 'sources.tar.gz').read_bytes()) == r['source_archive_sha256']
    with tarfile.open(HERE / 'sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 10
        assert all(m.isfile() for m in members)
        assert {m.name: sha(archive.extractfile(m).read()) for m in members} == h
    assert '421 passed' in (HERE / 'tests.log').read_text()
    assert sha((HERE / 'public_plan.json').read_bytes()) == r['public_plan_sha256']
    p = json.loads((HERE / 'public_plan.json').read_text())
    head = {sign*63*i % 65536 for i in range(1, 12) for sign in (-1, 1)}
    direct = head | {sign*16*i % 65536 for i in range(1, 128) for sign in (-1, 1)}
    factored = head | {sign*16*i % 65536 for i in range(1, 16) for sign in (-1, 1)}
    factored |= {sign*256*i % 65536 for i in range(1, 8) for sign in (-1, 1)}
    assert len(direct) == 276 and len(factored) == 66
    assert set(p['key_groups']['direct_routing_only']['keys']) == direct
    assert set(p['key_groups']['factored_routing_only']['keys']) == factored
    q, auxiliary = p['profile']['q'], p['profile']['p']
    assert (len(q), len(auxiliary)) == (100, 14) and math.prod(q+auxiliary).bit_length() <= 3523
    assert all(math.prod(q[i:i+13]) <= math.prod(auxiliary) for i in range(0, len(q), 13))
    pair = 2*8*114*131072*4
    assert p['profile']['raw_rotation_pair_bytes'] == pair
    for row in p['key_groups'].values():
        assert row['count'] == len(set(row['keys']))
        assert row['raw_rotation_bytes'] == row['count']*pair
        assert row['raw_rotation_GiB'] == row['raw_rotation_bytes']/2**30
    assert len(p['mappings']) == 128
    terminal = json.loads((HERE / 'terminal.json').read_text())
    assert terminal['exit_code'] == 0 and terminal['session_reaped']
    return dict(passed=True, source_files=10, tests=421, direct_keys=276, factored_keys=66,
                encrypted_execution=False, main_changed=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
