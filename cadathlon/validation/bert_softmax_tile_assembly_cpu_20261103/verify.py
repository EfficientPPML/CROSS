"""Offline integrity and public schedule checks; no model or HE imports."""
from pathlib import Path
import hashlib
import json
import tarfile

HERE = Path(__file__).resolve().parent


def sha(raw): return hashlib.sha256(raw).hexdigest()


def verify():
    for name, digest in json.loads((HERE/'artifacts.json').read_text()).items():
        assert sha((HERE/name).read_bytes()) == digest, name
    report = json.loads((HERE/'report.json').read_text())
    terminal = json.loads((HERE/'terminal.json').read_text())
    assert report['passed'] and report['tests_exit_code'] == 0
    assert report['source_files_unchanged'] and report['source_archive_verified']
    before = report['source_hashes_before']
    assert len(before) == 109 and before == report['source_hashes_after']
    assert sha((HERE/'sources.tar.gz').read_bytes()) == report['source_archive_sha256']
    with tarfile.open(HERE/'sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 109
        assert all(m.isfile() for m in members)
        raw = {m.name: archive.extractfile(m).read() for m in members}
    assert {name: sha(data) for name, data in raw.items()} == before
    assert '476 passed' in (HERE/'tests.log').read_text()
    assert terminal['observed_exit_code'] == 0 and terminal['session_reaped']
    manifest = json.loads(raw['candidate_manifest.json'])
    schedules = 0
    for key_count in range(1, 129):
        updates = manifest['normalization_reciprocal_by_public_K'][str(key_count)]['iterations']
        for continuation in (2, 17):
            level, refreshes = 23, 0
            for _ in range(5):
                if level < updates+7: level = 25; refreshes += 1
                level -= updates+5
                assert level >= 2
            if level < continuation: level = 24; refreshes += 1
            assert level >= continuation
            if key_count == 128:
                assert level == (7 if continuation == 2 else 24)
                assert refreshes == (4 if continuation == 2 else 5)
            schedules += 1
    return dict(passed=True, source_files=109, recorded_tests=476,
                independent_public_schedules=schedules, encrypted_execution=False,
                repeated_refresh_error_verified=False)


if __name__ == '__main__': print(json.dumps(verify(), indent=2))
