from pathlib import Path
from fractions import Fraction as F
import hashlib,json,tarfile
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads((P/'verification.json').read_text());r=json.loads((P/'report.json').read_text());c=json.loads((P/'certificate_report.json').read_text())
assert all(sha(P/n)==h for n,h in v['artifact_sha256'].items())
assert sha(P/'certificate_sources.tar.gz')==c['source_archive_sha256']==r['certificate_source_archive_sha256']==v['certificate_source_archive_sha256']
assert sha(P/'certificate_report.json')==r['certificate_report_sha256']
with tarfile.open(P/'certificate_sources.tar.gz') as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==4 and all(m.isfile() for m in ms)
 assert {'/'+m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==c['source_hashes_before']==c['source_hashes_after']
assert c['passed'] and r['passed'] and len(r['results'])==len(c['results'])==8
for name,x in c['results'].items():
 d=x['exact_H_coefficient_L1'];delta=F(int(d['numerator_hex'],16),1<<d['denominator_power_of_two']);assert 0<=delta<1
 e=delta/(2-delta);saved=x['exact_relative_bound'];assert e==F(int(saved['numerator_hex'],16),int(saved['denominator_hex'],16))
 lo,hi=x['normalized_multiplier_bounds'];assert F(lo)<=1-e and F(hi)>=1+e
 assert F(x['uniform_relative_error_upper'])>=e and x['uniform_relative_error_upper']<1e-6 and r['results'][name]['verified']
print(json.dumps({'passed':True,'sites':8,'candidate_selected':False,'model_evaluated':False}))
