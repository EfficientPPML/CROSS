"""Offline source/report verification; no checkpoint imports or model execution."""
from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r=json.loads((P/'report.json').read_text());s=json.loads((P/'summary.json').read_text())
assert r['passed'] is True and r['test_exit_code']==0 and r['source_files_unchanged'] is True
assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==53
assert r['source_archive_sha256']==s['source_archive_sha256']==sha(P/'sources.tar.gz')
with tarfile.open(P/'sources.tar.gz') as archive:
    members=archive.getmembers();assert len(members)==53 and len({m.name for m in members})==53 and all(m.isfile() for m in members)
    data={m.name:archive.extractfile(m).read() for m in members}
assert {n:hashlib.sha256(b).hexdigest() for n,b in data.items()}==r['source_hashes_before']
for n in ['check_approximate.py','candidate_gate.py','collector_requirement.json','origin.json','test_collector.py','test_prerequisites.py','preflight_collector.py']:assert (P/n).read_bytes()==data[n]
req=json.loads(data['collector_requirement.json']);m=json.loads(data['candidate-v2/manifest.json'])
assert req['manifest_sha256']==s['candidate_manifest_sha256']==hashlib.sha256(data['candidate-v2/manifest.json']).hexdigest()
assert m['source_input_sha256']==req['selection_input_sha256'] and len(req['selection_input_sha256'])==20
for group in ('selection_artifact_sha256','selection_input_sha256','calibration_input_sha256','runtime_dependency_sha256'):
    assert all(hashlib.sha256(data[n]).hexdigest()==h for n,h in req[group].items())
assert req['calibration_input_sha256']['calibration/protocol.json']!=req['selection_input_sha256']['protocol.json']
assert r['checkpoint_model_evaluated'] is False and r['validation_outputs_read'] is False
assert s['observed_terminal_exit_code']==0 and s['session_reaped'] is True
print('PASS: 53 exact sources, 20 selected inputs, separate original calibration contract, no model evaluation')
