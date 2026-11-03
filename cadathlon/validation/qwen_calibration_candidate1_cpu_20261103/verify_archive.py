"""Offline candidate byte/certificate-inventory verification; no model imports."""
from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((P/'verification.json').read_text())
for n,h in v['artifact_sha256'].items():assert sha(P/n)==h,n
for run in ('candidate-v1','candidate-validation'):
    r=json.loads((P/run/'report.json').read_text());assert r['passed'] and r['source_hashes_before']==r['source_hashes_after']
    assert sha(P/run/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(P/run/'sources.tar.gz') as t:actual={i.name:hashlib.sha256(t.extractfile(i).read()).hexdigest() for i in t.getmembers()}
    assert actual==r['source_hashes_before']
    for n,h in actual.items():assert sha(P/n)==h,n
m=json.loads((P/'candidate-v1/manifest.json').read_text());assert sha(P/'candidate-v1/manifest.json')==v['manifest_sha256']
assert len(m['nonlinear'])==141 and set(m['reciprocal_by_keys'])=={str(k) for k in range(1,33)}
assert {k:x['final_depth'] for k,x in m['depth_by_sequence'].items()}=={'8':9588,'16':10026,'19':10464,'32':10464}
assert not v['model_or_approximate_trajectory_evaluated'] and not v['validation_data_used_for_selection']
print(json.dumps({'passed':True,'candidate_index':1,'nonlinear_sites':141,'manifest_sha256':v['manifest_sha256'],'approximate_full_model_evaluated':False}))
