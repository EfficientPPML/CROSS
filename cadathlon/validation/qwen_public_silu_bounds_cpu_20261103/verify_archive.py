from pathlib import Path
from fractions import Fraction
import hashlib,json,tarfile
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((P/'verification.json').read_text());r=json.loads((P/'public-v1/report.json').read_text())
assert all(sha(P/n)==h for n,h in v['artifact_sha256'].items())
with tarfile.open(P/'public-v1/sources.tar.gz') as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==11 and all(m.isfile() for m in ms)
 assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==r['source_hashes_before']==r['source_hashes_after']
assert sha(P/'public-v1/sources.tar.gz')==r['source_archive_sha256']==v['source_archive_sha256']
assert r['passed'] and r['complete'] and r['rows_computed']==86016 and len(r['layers'])==28
assert not r['model_evaluated'] and not r['coefficient_selection'] and not r['encrypted_execution'] and r['no_jax_imported']
assert r['asset_sha256_before']==r['asset_sha256_after']==json.loads((P/'model_pins.json').read_text())['asset_sha256']
for i in range(28):
 layer=r['layers'][str(i)];assert layer['row_count']==len(layer['rows'])==3072
 for row in layer['rows']:
  exact=Fraction(*map(int,row['squared_bound_fraction']));assert exact>=0 and Fraction.from_float(row['native_exact_real_upper'])**2>=exact
 assert layer['maximum_row_upper']==max(x['native_exact_real_upper'] for x in layer['rows'])
 for name,shape in [('gamma',[1024]),('gate_weight',[3072,1024])]:
  d=layer[name];assert d['shape']==shape and d['source_roundtrip_exact'] and d['source_matches_direct_checkpoint_bytes'] and d['source_header_dtype'] in ('BF16','F32') and d['promoted_dtype']=='<f4'
new=json.loads((P/'public-v1/tokenized_fresh_inputs.json').read_text());raw=json.loads((P/'fresh_validation_proposal.json').read_text());old=[]
for name in ('old_calibration_inputs.json','old_validation_inputs.json'):old+=json.loads((P/'inputs'/name).read_text())
assert len(new)==len({tuple(x['token_ids']) for x in new})==56 and len(old)==344
assert not ({tuple(x['token_ids']) for x in new}&{tuple(x['token_ids']) for x in old})
for x,y in zip(new,raw):
 assert all(x[k]==value for k,value in y.items()) and len(x['token_ids'])==x['sequence_length'] and x['raw_token_length']>=32
freeze=json.loads((P/'fresh_input_freeze.json').read_text())
assert sha(P/'public-v1/tokenized_fresh_inputs.json')==r['fresh_validation']['tokenized_sha256']==freeze['tokenized_manifest_sha256']
assert sha(P/'fresh_validation_proposal.json')==freeze['source_raw_sha256']
print(json.dumps({'passed':True,'row_enclosures_verified':86016,'fresh_inputs':56,'old_overlap_count':0,'model_evaluated':False,'candidate_selected':False}))
