from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((P/'verification.json').read_text());r=json.loads((P/'public-tests-v1/report.json').read_text())
assert all(sha(P/n)==h for n,h in v['artifact_sha256'].items())
with tarfile.open(P/'public-tests-v1/sources.tar.gz') as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==7 and all(m.isfile() for m in ms)
 assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==r['source_hashes_before']==r['source_hashes_after']
assert r['passed'] and r['test_exit_code']==0 and not r['model_evaluated'] and not r['checkpoint_weights_read'] and not r['coefficient_selection']
p=json.loads((P/'proposal.json').read_text());rows=json.loads((P/'fresh_validation_proposal.json').read_text())
assert sha(P/'fresh_validation_proposal.json')==p['fresh_validation']['sha256'] and len(rows)==len({x['text'] for x in rows})==56
print(json.dumps({'passed':True,'synthetic_tests':8,'fixed_weights_computed':False,'model_evaluated':False,'candidate_selected':False}))
