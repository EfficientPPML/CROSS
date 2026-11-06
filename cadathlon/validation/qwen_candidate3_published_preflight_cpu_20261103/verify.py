from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
sha=lambda x:hashlib.sha256(x).hexdigest()
for n,h in json.loads((ROOT/'artifact_manifest.json').read_bytes()).items(): assert sha((ROOT/n).read_bytes())==h,n
r=json.loads((ROOT/'preflight-v1/report.json').read_bytes()); b=r['source_hashes_before']
assert r['passed'] and r['test_exit_code']==0 and len(b)==70 and b==r['source_hashes_after']
assert r['tests']==dict(tests=129,errors=0,failures=0,skipped=0)
assert r['source_files_unchanged'] and r['source_archive_verified']
a=ROOT/'preflight-v1/sources.tar.gz';assert sha(a.read_bytes())==r['source_archive_sha256']
with tarfile.open(a) as t:
 ms=t.getmembers();assert len(ms)==70==len({m.name for m in ms}) and all(m.isfile() for m in ms)
 assert {m.name:sha(t.extractfile(m).read()) for m in ms}==b
assert not any(n.startswith('validation_inputs/') for n in b)
assert r['read_guard']['forbidden_reads_blocked']==[] and not r['fresh_input_bodies_read'] and not r['checkpoint_model_evaluated']
req=json.loads((ROOT/'validation_requirement.json').read_bytes());assert len(req['expected_runtime_source_hashes'])==55
assert all(b[n]==h for n,h in req['expected_runtime_source_hashes'].items())
assert req['candidate_manifest_sha256']=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
assert req['validation_input_sha256']=='5caad37526dfda5a379f029ddfc75fb10879b3313d3533ec4cbf60d4192ef359'
for p in ROOT.glob('*.py'):
 if p.name!='verify.py':assert sha(p.read_bytes())==b[p.name]
print('PASS: 110 tests + 19 subtests, 70 exact sources; published body unread in preflight.')
