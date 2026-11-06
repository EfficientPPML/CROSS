"""Portable verification of a preserved numerical failure; no HE dependencies."""
from pathlib import Path
import hashlib
import json
import tarfile

P = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
r = json.loads((P / 'run-v1/report.json').read_bytes())
s = json.loads((P / 'failure_summary.json').read_bytes())
receipt = json.loads((P / 'terminal_readback.json').read_bytes())
assert r['passed'] is False and r['complete'] is True and r['preflight_only'] is False
assert r['errors'] == [] and r['fixed_absolute_error_gate'] == 1e-6
assert len(r['checks']) == 54
failed = {n: v['max_abs_error'] for n, v in r['checks'].items() if not v['passed']}
assert set(failed) == {f'K128_rows16:y{i:02d}' for i in range(8, 14)}
assert failed == s['checkpoint_failures'] and max(failed.values()) == 1.9912012305667304e-6
assert all(v['metadata_ok'] and v['shape_and_finite'] for v in r['checks'].values())
derived = {c: {n: v['max_abs_error'] for n, v in rows.items() if not v['passed']}
           for c, rows in r['derived_checks'].items()}
assert derived == s['derived_failures'] == {
    'K128_rows16': {'final_true_inverse': 1.9912010884581832e-6}, 'K15_rows9': {}}
assert sum(map(len, r['derived_checks'].values())) == 52
assert all(v['final_true_normalized_quotient']['passed'] for v in r['derived_checks'].values())
assert r['server_complete_before_any_decode'] is True
assert r['input_payload_and_metadata_unchanged'] is True
assert r['refresh_performed'] is False and r['full_model_evaluated'] is False
assert r['cleanup_completed'] is True and r['owned_key_directory_removed'] is True
assert r['cleanup_errors'] == r['collector_cleanup_errors'] == []
assert len(r['codec_guards']) == 7 and len(r['key_generation_guards']) == 2
assert r['source_hashes_before'] == r['source_hashes_after'] and len(r['source_hashes_before']) == 114
assert r['source_files_unchanged'] is True and r['source_archive_verified'] is True
assert sha(P / 'run-v1/report.json') == receipt['report_sha256'] == 'f97f839d23a2d91e258f8937a92bc8173ebd1bc77483db803a100531c8985b45'
assert sha(P / 'run-v1/sources.tar.gz') == r['source_archive_sha256'] == receipt['source_archive_sha256']
with tarfile.open(P / 'run-v1/sources.tar.gz') as t:
    members = t.getmembers()
    assert len(members) == len({m.name for m in members}) == 114 and all(m.isfile() for m in members)
    assert {m.name: hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members} == r['source_hashes_before']
assert receipt['session'] == 17163 and receipt['exit_code'] == 1 and receipt['reaped'] is True
assert all(receipt[n] is True for n in ('worker_absent', 'owned_group_absent',
    'owned_key_directory_absent', 'all114_current_sources_equal_start_end_archive'))
assert 'Exit status: 1' in (P / 'run-v1.time.txt').read_text()
inventory = json.loads((P / 'artifact_hashes.json').read_bytes())
assert set(inventory) == {str(p.relative_to(P)) for p in P.rglob('*') if p.is_file()
                        and p.name != 'artifact_hashes.json' and '__pycache__' not in p.parts}
for name, digest in inventory.items():
    assert sha(P / name) == digest, name
print('VERIFIED PRESERVED FAILURE: 114 exact sources; 6/54 checkpoint and 1/52 derived failures; exit1/reaped, cleanup complete.')
