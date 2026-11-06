from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as E
p=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
manifest=json.loads((p/'artifact_hashes.json').read_bytes())
assert all(sha((p/n).read_bytes())==h for n,h in manifest.items())
r=json.loads((p/'report.json').read_bytes()); review=json.loads((p/'root-review/failed-full-v1.json').read_bytes())
assert r['passed']is False and r['complete']is False
assert r['tests']==dict(tests=80,errors=0,failures=2,skipped=0)
assert r['source_hashes_before']==r['source_hashes_after']==r['snapshot_source_hashes_after'] and len(r['source_hashes_before'])==150
assert r['source_files_unchanged']and r['source_archive_verified']and r['asset_files_unchanged']
assert r['all_direct_children_reaped']and r['all_owned_groups_empty']and r['process']['direct_returncode']==1
assert not r['cleanup_errors']and r['process']['cleanup_completed']
assert review['outer_exit_code']==1 and review['outer_reaped']and review['execution_passed']is False
assert sha((p/'report.json').read_bytes())==review['report_sha256']
assert sha((p/'sources.tar.gz').read_bytes())==r['source_archive_sha256']==review['archive_sha256']
with tarfile.open(p/'sources.tar.gz')as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==150 and all(m.isfile()for m in ms)
 assert {m.name:sha(t.extractfile(m).read())for m in ms}==r['source_hashes_before']
suites=list(E.parse(p/'tests.xml').getroot().iter('testsuite'))
assert {k:sum(int(s.attrib.get(k,0))for s in suites)for k in ('tests','errors','failures','skipped')}==r['tests']
failed=[c.attrib['name']for s in suites for c in s.findall('testcase')if c.find('failure')is not None]
assert len(failed)==2 and all('joined_all12'in n for n in failed)
print('PASS: preserved ideal fixture failure;150 sources,80 tests,2 failed full cases; no numerical pass.')
