"""Offline public-only evidence verification; never opens private workspaces."""
from pathlib import Path
import hashlib,json,tarfile,re
ROOT=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
for name,h in json.loads((ROOT/'artifact_hashes.json').read_text()).items():assert sha(ROOT/name)==h,name
r=json.loads((ROOT/'report.json').read_text());v=json.loads((ROOT/'terminal_readback.json').read_text())
assert r['passed'] and r['execute'] and r['actual_native_started'] and r['official_evaluation_complete']
assert r['exact_reference_parity_passed'] and r['approximation_validation_passed']
assert all(r.get(k)is True for k in r['reference_acceptance_requires'])
assert r['progress']==dict(completed_unique=408,exact_parity_failures=0,approximation_failures=0)
assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==41
assert r['asset_hashes_before']==r['asset_hashes_after'] and len(r['asset_hashes_before'])==6
assert sha(ROOT/'sources.tar.gz')==r['source_archive_sha256']==v['source_archive_sha256']
with tarfile.open(ROOT/'sources.tar.gz') as tar:
    members=tar.getmembers();assert len(members)==41==len({m.name for m in members})
    assert all(m.isfile() for m in members)
    assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members}==r['source_hashes_before']
assert [(c['size'],c['seed'],c['count']) for c in r['cases']]==[(s,seed,count) for s,count in ((1,100),(2,200),(3,408)) for seed in (42,43,44)]
assert sum(c['count'] for c in r['cases'])==2124
assert [c['official_correct'] for c in r['cases']]==[82,79,84,162,172,166,345,345,345]
for c in r['cases']:
    assert c['official_correct']==c['approximate_correct'] and c['official_accuracy']==c['approximate_accuracy']==c['official_correct']/c['count']
    assert c['domain_failure_count']==0
    assert all(c[k]is True for k in ('approximation_complete','approximate_numeric_gate_passed','prediction_agreement','ranked_accuracy_floor_passed','ranked_accuracy_drop_passed','exact_reference_parity_passed'))
assert len(r['private_reference_files'])==11 and r['native_policy']['official_attention']=='sdpa' and r['native_policy']['diagnostic_attention']=='eager'
assert r['cpu_affinity']==[124,125,126,127] and r['native_policy']['torch_threads']==4
assert v['observed_exit_code']==0 and v['all_owned_pids_absent'] and v['current_before_after_archive_equal'] and v['report_sha256']==sha(ROOT/'report.json')
assert re.search(r'Exit status: 0\s*$',(ROOT/'actual-v1.time').read_text())
review=json.loads((ROOT/'root_private_readback.json').read_text())
assert review['passed'] and review['report_sha256']==sha(ROOT/'report.json')
assert review['unique_rows']==408 and review['selected_rows']==2124 and review['domain_observations']==63648
assert review['private_reference_files']==11 and review['original_fixture_files']==54 and review['private_payloads_archived']is False
assert review['max_exact_native_diagnostic_error']<=1e-12
assert [(x['count'],x['seed'],x['official_correct'],x['approximate_correct']) for x in review['cases']]==[(x['count'],x['seed'],x['official_correct'],x['approximate_correct']) for x in r['cases']]
terminal=json.loads((ROOT/'root_terminal_observation.json').read_text());assert terminal['session_id']==9977 and terminal['reaped'] and terminal['observed_exit_code']==0
assert terminal['verifier_archive_time_sha256']==sha(ROOT/'root_private_verifier.py')
successor=json.loads((ROOT/'root_readback_successor/report.json').read_text())
assert {k:v for k,v in successor.items() if k!='verified_utc'}=={k:v for k,v in review.items() if k!='verified_utc'}
terminal2=json.loads((ROOT/'root_readback_successor/terminal.json').read_text());assert terminal2['session_id']==47924 and terminal2['observed_exit_code']==0 and terminal2['reaped']
assert terminal2['verifier_archive_time_sha256']==sha(ROOT/'root_readback_successor/verify_private.py')
print('PASS:408 unique plaintext references;9 ranked cases;41 exact sources; two independent private readbacks; no HE or benchmark claim')
