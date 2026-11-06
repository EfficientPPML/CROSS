"""Offline exact source/preflight/identity audit; no model imports."""
from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
summary=json.loads((P/'summary.json').read_text())
for kind in ('fresh','published'):
 p=P/kind;r=json.loads((p/'report.json').read_text());s=summary[kind]
 assert r['passed'] and r['test_exit_code']==0 and r['source_files_unchanged'] and r['calibrated_runtime_sources_identical']
 assert len(r['source_hashes_before'])==63 and r['source_hashes_before']==r['source_hashes_after']
 assert sha((p/'sources.tar.gz').read_bytes())==r['source_archive_sha256']==s['source_archive_sha256']
 with tarfile.open(p/'sources.tar.gz') as t:
  members=t.getmembers();assert len(members)==63==len({m.name for m in members}) and all(m.isfile() for m in members)
  data={m.name:t.extractfile(m).read() for m in members}
 assert {n:sha(b) for n,b in data.items()}==r['source_hashes_before']
 req=json.loads(data['validation_requirement.json']);proto=json.loads(data['validation_protocol.json'])
 assert proto['workload']==kind and proto['context_count']==112
 assert len(req['expected_runtime_source_hashes'])==53
 assert all(sha(data[n])==h for n,h in req['expected_runtime_source_hashes'].items())
 assert sha(data['calibration_prerequisite/report.json'])==req['full_report_sha256']=='8bcc7b6e7275914b85245e4ee65481ca87b37fed565dda8bac59fe97563b6641'
 assert sha(data['calibration_prerequisite/runtime_sources.tar.gz'])==req['full_source_archive_sha256']=='42cf71780d0f8d84f29791f8e18ed13f133589bb5c66fed9a45790b4d3d1578b'
 assert sha(data['validation_inputs/tokenized_fixtures.json'])==req['validation_input_sha256']==proto['input_sha256']
 assert sha(data['validation_protocol.json'])==req['validation_protocol_sha256']
 assert all(sha(data[n])==h for n,h in req['validation_code_sha256'].items())
 assert req['candidate_manifest_sha256']=='0af45a8fc6d13b9d429ee1af2311b2cb6ac0aa4cde02eca56478472762482489'
 for n in ['validate_heldout.py','calibration_gate.py','validation_gate.py','validation_requirement.json','validation_protocol.json','test_validation_gate.py','preflight_validation.py']:assert (p/n).read_bytes()==data[n]
 assert not r['checkpoint_model_evaluated'] and not r['validation_outputs_read']
 assert s['observed_terminal_exit_code']==0 and s['session_reaped']
old=P/'fresh/historical-preflight-v1';r=json.loads((old/'report.json').read_text());assert not r['passed'] and r['test_exit_code']==1 and r['source_files_unchanged']
assert sha((old/'sources.tar.gz').read_bytes())==r['source_archive_sha256']
print('PASS: both63-source preflights, same53-source calibrated runtime, distinct frozen112 inputs; historical test failure retained; no model execution')
