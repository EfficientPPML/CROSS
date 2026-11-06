"""Portable public-only evidence verifier; never executes producer code."""
import argparse,hashlib,io,json,tarfile,xml.etree.ElementTree as ET
from pathlib import Path

def sha(raw):return hashlib.sha256(raw).hexdigest()
def j(path):return json.loads(path.read_bytes())
def archive(path):
 with tarfile.open(fileobj=io.BytesIO(path.read_bytes()),mode='r:gz')as t:
  rows=t.getmembers();assert len({r.name for r in rows})==len(rows)
  assert all(r.isfile()and not Path(r.name).is_absolute()and '..'not in Path(r.name).parts and 0<=r.size<=8<<20 for r in rows)
  return {r.name:t.extractfile(r).read()for r in rows}
def xml(path,n,fail):
 t=ET.parse(path);assert len(t.findall('.//testcase'))==n and len(t.findall('.//failure'))==fail
 assert not t.findall('.//error')and not t.findall('.//skipped')
def main():
 p=argparse.ArgumentParser();p.add_argument('--current',type=Path);a=p.parse_args();root=Path(__file__).resolve().parent
 manifest=j(root/'artifact_hashes.json');actual={str(q.relative_to(root))for q in root.rglob('*')if q.is_file()and q.name!='artifact_hashes.json'}
 assert actual==set(manifest)
 for name,h in manifest.items():assert sha((root/name).read_bytes())==h
 b=root/'source-bound-v2';r=j(b/'report.json');t=j(b/'root_terminal.json');s=j(b/'source_manifest.json')
 assert sha((b/'report.json').read_bytes())=='600de45d83a350aad241f2cdf9874b080d2890117c8cb1680c71a87b039e3844'
 assert sha((b/'sources.tar.gz').read_bytes())=='da1291bcef3b9d2ac58379e12d493015c57c0ed7f120687c95e2522807a00453'
 assert len(s)==6 and s==r['source_hashes_before']==r['source_hashes_after']
 source=archive(b/'sources.tar.gz');assert {n:sha(raw)for n,raw in source.items()}==s
 assert r['passed']is True and r['tests']==88 and r['pytest_returncode']==0 and r['xml_bad']==0
 assert r['audit_denials']==r['model_backend_modules']==[]
 for key in ('actual_freeze_published','actual_model_or_private_body_read','execution_authorized'):assert r[key]is False and t[key]is False
 assert r['actual_candidate_outcome_read']is False
 assert t['passed']is True and t['exit_code']==0 and t['reaped']is True and t['process_group_absent']is True
 assert t['current_execution_archive_start_end_equal']is True and t['sources']==6 and t['tests']==88
 assert t['report_sha256']==sha((b/'report.json').read_bytes())
 assert t['source_archive_sha256']==sha((b/'sources.tar.gz').read_bytes())
 assert t['source_manifest_sha256']==r['source_manifest_sha256']==sha((b/'source_manifest.json').read_bytes())
 assert t['review_script_sha256']==sha((b/'root_review.py').read_bytes())
 xml(b/'tests.xml',88,0)
 h=root/'history';old=j(h/'original79/reported_terminal.json')
 assert {n:sha(raw)for n,raw in archive(h/'original79/sources-post-terminal.tar.gz').items()}==old['sources_post_terminal']
 assert old['independent_process_absence_proof']is False and old['original_log_or_xml_recorded']is False
 for label,n,fail in [('negative-v1',2,1),('corrected-negative-v1',1,1),('successor-v2',1,0)]:xml(h/label/'tests.xml',n,fail)
 old=archive(h/'corrected-negative-v1/sources.tar.gz');new=archive(h/'successor-v2/sources.tar.gz')
 for name in ('legacy_bundle.json','test_legacy_closure.py'):assert old[name]==new[name]
 for name in ('freeze_issuer.py','freeze_admission.py'):assert new[name]==source[name]
 if a.current:
  for name,pin in s.items():assert sha((a.current/name).read_bytes())==pin
 print(json.dumps(dict(passed=True,sources=6,tests=88,public_files=len(manifest)+1,current_checked=bool(a.current),actual_freeze_published=False)))
if __name__=='__main__':main()
