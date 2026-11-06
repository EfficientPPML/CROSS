"""Verify public ideal evidence only; no backend or private workspace reads."""
from pathlib import Path
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name, expected in json.loads((ROOT/'artifact_hashes.json').read_text()).items():
    assert sha(ROOT/name) == expected, name


def archive(path, expected, digest=None):
    if digest is not None:
        assert sha(path) == digest
    with tarfile.open(path) as tar:
        members = tar.getmembers()
        assert len(members) == len({m.name for m in members}) == len(expected)
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert {m.name: hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members} == expected


r = json.loads((ROOT/'report.json').read_text())
assert all(r[n] is True for n in ('passed', 'terminal_scope_completed', 'cleanup_completed',
                                'process_reaped', 'source_files_unchanged', 'source_archive_verified'))
assert r['returncode'] == 0 and r['encrypted_execution'] is False
assert r['source_count'] == 28 and r['source_hashes_before'] == r['source_hashes_after'] == r['snapshot_hashes']
assert r['test_counts'] == dict(tests=606, failures=0, errors=0, skipped=0)
assert r['full_slot_comparisons'] == 864 and r['compared_slots'] == 56623104
archive(ROOT/'sources.tar.gz', r['source_hashes_before'], r['source_archive_sha256'])
suites = list(ET.parse(ROOT/'tests.xml').getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in suites) == 606
assert all(int(s.get(n, 0)) == 0 for s in suites for n in ('errors', 'failures', 'skipped'))
t = json.loads((ROOT/'terminal_readback.json').read_text())
assert t['observed_exit_code'] == 0 and t['reaped'] and t['owned_pids_absent'] and t['current_snapshot_archive_equal']
meta = ROOT/'independent_metadata'
m = json.loads((meta/'report.json').read_text())
assert m['passed'] and m['tests_passed'] == 46 and m['source_count'] == 14 and m['extracted_actual_method_count'] == 21
assert m['source_hashes_before'] == m['source_hashes_after']
archive(meta/'sources.tar.gz', m['source_hashes_before'], m['source_archive_sha256'])
for folder in (ROOT/'history').iterdir():
    if folder.is_dir():
        h = json.loads((folder/'report.json').read_text())
        archive(folder/'sources.tar.gz', h['archive_time_hashes'], h['source_archive_sha256'])
review = json.loads((ROOT/'independent_review/final_readonly_review.json').read_text())
assert review['reviewed_owner_sha256'] == r['source_hashes_before']['encrypted_bsgs_score_routing.py']
negative = ROOT/'independent_review/readback-v1'
n = json.loads((negative/'source_hashes_before.json').read_text())
archive(negative/'sources.tar.gz', n)
assert len(n) == 8
negative_suites = list(ET.parse(negative/'tests.xml').getroot().iter('testsuite'))
assert sum(int(s.get('tests', 0)) for s in negative_suites) == 3
assert sum(int(s.get('failures', 0)) for s in negative_suites) == 3
assert sum(int(s.get('errors', 0)) for s in negative_suites) == 0
print('PASS:606 ideal/metadata checks;28 exact sources;864 full-slot comparisons; no HE or speed claim')
