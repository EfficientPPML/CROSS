"""Offline public proposal byte verification; no backend/model imports."""
from pathlib import Path
import hashlib,json,tarfile
p=Path(__file__).resolve().parent;v=json.loads((p/'verification.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
for name,expected in v['artifact_sha256'].items():assert sha((p/name).read_bytes())==expected,name
r=json.loads((p/'final-preflight/report.json').read_text());assert r['passed'] and r['source_hashes_before']==r['source_hashes_after']
with tarfile.open(p/'final-preflight/sources.tar.gz') as t:
    actual={i.name:sha(t.extractfile(i).read()) for i in t.getmembers()};assert actual==r['source_hashes_before']
for name,expected in actual.items():assert sha((p/name).read_bytes())==expected,name
pre=json.loads((p/'input_preflight.json').read_text());assert pre['passed'] and pre['validation_token_intersection']==0 and pre['no_model_instantiated'] and pre['no_jax_imported']
assert pre['unique_per_bucket']=={'8':96,'16':96,'32':96}
print(json.dumps({'passed':True,'selected_source_count':len(actual),'proposal_only':True,'model_measurement_performed':False}))
