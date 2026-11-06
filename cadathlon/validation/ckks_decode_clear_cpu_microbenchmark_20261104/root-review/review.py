"""Independent source/data/timing readback. Does not execute arithmetic."""
import hashlib
import json
import os
import statistics
import tarfile
from pathlib import Path

P = Path('/tmp/ckks-decode-clear-cpu-microbenchmark-20261104')
OLD_PROOF = Path('/tmp/ckks-scaled-crt-decoder-candidate-v2-20261104/source-bound-v1')
OUT = Path('/tmp/ckks-decode-microbenchmark-root-review-20261104')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
def read(p, pin=None):
    raw = p.read_bytes()
    if pin is not None:
        assert sha(raw) == pin
    return raw
def absent(pid, group=False):
    try:
        (os.killpg if group else os.kill)(pid, 0)
    except ProcessLookupError:
        return True
    return False

manifest_raw = read(P/'sources-before.json', '8d30d2eb685de9080aba631f3344cfa1104da578cacbc3b55effe3c53df88013')
manifest = json.loads(manifest_raw)
assert len(manifest) == 210
assert json.loads(read(P/'sources-after.json')) == manifest
assert all(sha(read(P/n)) == h for n, h in manifest.items())
read(P/'sources.tar.gz', 'cc0bf3e01d80e1d94bdd269098e302aac83062c72f0f6f2548bb50ff9f6c4d26')
with tarfile.open(P/'sources.tar.gz') as archive:
    members = archive.getmembers()
    assert len(members) == 210 and all(m.isfile() for m in members)
    assert {m.name for m in members} == set(manifest)
    assert all(sha(archive.extractfile(m).read()) == manifest[m.name] for m in members)
original = json.loads(read(OLD_PROOF/'sources-before.json', '9d9190b2a3c65f831f69fe30af6ece7b0b94f05bf83e9a21310ca32f99fda2aa'))
assert len(original) == 207
assert all(h == manifest['runtime207/'+n] == sha(read(OLD_PROOF/'execution'/n)) for n, h in original.items())
assert set(manifest) - {'runtime207/'+n for n in original} == {'benchmark.py', 'plan.json', 'public_p60_profile.json'}
plan = json.loads(read(P/'plan.json'))
assert plan['original_runtime207_sources'] == original
r = json.loads(read(P/'run-v1/report.json', '88bb46090d6546849cdb1d5d1c6892fc19e0efc7400fa4b186f407b36ff4a3bc'))
assert r['source_hashes_start'] == r['source_hashes_end'] == manifest
assert all(r[k] is True for k in ('passed', 'complete', 'source_unchanged'))
assert all(r[k] is False for k in ('ciphertext_or_key_or_model_operation', 'actual_toy5_recovery', 'new_optimization', 'tpu_plugin_loaded_at_end'))
assert r['jax_backend'] == 'cpu' and r['device_platforms'] == ['cpu']
assert r['cpu_affinity'] == [112, 113, 114, 115]
assert r['import_guard_denials'] == ['libtpu'] and r['device_access_denials'] == []
assert all(row['sha256'] == manifest[row['path']] for row in r['loaded_local_modules'].values())
assert len(r['dataset_files']) == 6
for name, row in r['dataset_files'].items():
    raw = read(P/'run-v1/synthetic_datasets'/name, row['sha256'])
    assert len(raw) == row['bytes']
summary = []
assert len(r['cases']) == 2
for case, specification in zip(r['cases'], plan['cases'], strict=True):
    assert case['name'] == specification['name']
    assert case['q'] == specification['q'] and case['scale'] == specification['scale']
    assert case['residues_shape'] == [131072, len(case['q'])]
    assert case['residues_unchanged'] is True
    assert len(case['calls']) == 10
    schedule = [('warmup', None, name) for name in ('old', 'candidate')]
    schedule += [('measured', pair, name) for pair in range(4) for name in (('old', 'candidate') if pair%2 == 0 else ('candidate', 'old'))]
    assert [(c['phase'], c['pair'], c['decoder']) for c in case['calls']] == schedule
    assert len({c['output_sha256'] for c in case['calls']}) == 1
    assert all(c['exact_binary64_agreement'] is True and c['elapsed_ns'] > 0 and c['elapsed_seconds'] == c['elapsed_ns']/1e9 for c in case['calls'])
    measured = {name: [c['elapsed_seconds'] for c in case['calls'] if c['phase'] == 'measured' and c['decoder'] == name] for name in ('old', 'candidate')}
    ratios = [new/old for old, new in zip(measured['old'], measured['candidate'], strict=True)]
    summary.append({'case': case['name'], 'old_seconds': measured['old'], 'candidate_seconds': measured['candidate'], 'old_median_seconds': statistics.median(measured['old']), 'candidate_median_seconds': statistics.median(measured['candidate']), 'paired_candidate_over_old_ratios': ratios, 'median_paired_ratio': statistics.median(ratios)})
assert (r['pid'], r['pgid']) == (3011310, 3011309)
assert absent(r['pid']) and absent(r['pgid'], group=True)
OUT.mkdir(exist_ok=False)
result = {'passed': True, 'scope': 'clear CPU decoder microbenchmark only; source/hash/data/timing readback, no arithmetic rerun', 'source_count': 210, 'unchanged_original_sources': 207, 'all_current_archive_start_end_exact': True, 'all20_calls_report_exact_binary64_agreement': True, 'synthetic_dataset_files': 6, 'synthetic_dataset_bytes': sum(v['bytes'] for v in r['dataset_files'].values()), 'report_sha256': sha(read(P/'run-v1/report.json')), 'owner_session': 28130, 'owner_reported_exit0_reaped': True, 'owner_terminal_chunk': '41e03a', 'recorded_pid_pgid': [r['pid'], r['pgid']], 'independently_absent': True, 'summary': summary, 'statistical_speed_claim': False, 'full_HE_or_calibration': False, 'new_optimization': False}
(OUT/'review.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
(OUT/'review.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result, sort_keys=True))
