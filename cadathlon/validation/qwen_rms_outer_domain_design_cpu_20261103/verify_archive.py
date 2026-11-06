from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
r=json.loads((P/'report.json').read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
assert r['passed'] and r['test_exit_code']==0 and r['source_hashes_before']==r['source_hashes_after']
assert len(r['source_hashes_before'])==7 and sha((P/'sources.tar.gz').read_bytes())==r['source_archive_sha256']
with tarfile.open(P/'sources.tar.gz') as t:
 m=t.getmembers();assert len(m)==7==len({x.name for x in m}) and all(x.isfile() for x in m)
 data={x.name:t.extractfile(x).read() for x in m}
assert {n:sha(b) for n,b in data.items()}==r['source_hashes_before']
assert all((P/n).read_bytes()==b for n,b in data.items())
assert r['plan']['rms_sites']['total']==113 and r['plan']['extraction_not_performed']
assert not r['plan']['model_evaluated'] and not r['plan']['coefficients_selected'] and not r['plan']['validation_outputs_used']
print('PASS: seven exact design/reference sources,11tests; no tensor extraction/model/coefficient selection')
