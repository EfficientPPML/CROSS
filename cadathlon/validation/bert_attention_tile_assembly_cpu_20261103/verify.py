"""Verify exact archived inputs and the recorded ideal-arithmetic result."""
from pathlib import Path
import hashlib
import json
import tarfile

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def verify():
    for name, expected in json.loads((HERE/'artifacts.json').read_text()).items():
        assert sha((HERE/name).read_bytes()) == expected, name
    report = json.loads((HERE/'report.json').read_text())
    assert report['passed'] and report['tests_exit_code'] == 0
    assert report['source_files_unchanged'] and report['source_archive_verified']
    before = report['source_hashes_before']
    assert len(before) == 9 and before == report['source_hashes_after']
    assert sha((HERE/'sources.tar.gz').read_bytes()) == report['source_archive_sha256']
    with tarfile.open(HERE/'sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 9
        assert all(m.isfile() for m in members)
        assert {m.name: sha(archive.extractfile(m).read()) for m in members} == before
    assert '198 passed' in (HERE/'tests.log').read_text()
    head = {sign*63*i % 65536 for i in range(1, 12) for sign in (-1, 1)}
    local = {sign*16*i % 65536 for i in range(1, 16) for sign in (-1, 1)}
    tiles = {sign*256*i % 65536 for i in range(1, 8) for sign in (-1, 1)}
    row = {4096*i for i in range(1, 16)}
    broadcast = {sign*2**i % 65536 for i in range(6) for sign in (-1, 1)}
    assert len(head | local | tiles | row | broadcast) == 89
    assert 120 + 128*6 + 16*26 + 112*27 == 4328
    assert 89*(2*8*114*131072*4)/2**30 == 79.265625
    return dict(passed=True, source_files=9, ideal_tests=198,
                rotation_keys=89, rotations_per_direction=4328,
                encrypted_execution=False, full_model=False)


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
