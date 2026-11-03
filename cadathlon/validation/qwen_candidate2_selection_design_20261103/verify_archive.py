from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((P/'verification.json').read_text());r=json.loads((P/'preflight-v1/report.json').read_text());p=json.loads((P/'proposal.json').read_text())
assert all(sha(P/n)==h for n,h in v['artifact_sha256'].items())
with tarfile.open(P/'preflight-v1/sources.tar.gz') as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==6 and all(m.isfile() for m in ms)
 assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==r['source_hashes_before']==r['source_hashes_after']
assert r['passed'] and r['test_exit_code']==0 and not r['coefficient_selection'] and not r['model_evaluated']
assert p['frozen_base']['unchanged_norm_rule_count']==113 and p['original_published_regression']['required_after_fresh_validation']
assert p['fresh_validation']['sha256']=='af93ae2a580bc4ff51d1463e8ab50ab054e295d0e4932340ae81d18185696988'
print(json.dumps({'passed':True,'tests':9,'source_count':6,'selection_performed':False,'model_evaluated':False}))
