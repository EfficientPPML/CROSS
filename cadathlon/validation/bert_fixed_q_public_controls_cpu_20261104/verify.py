from pathlib import Path
import argparse,hashlib,json,tarfile,xml.etree.ElementTree as E
p=argparse.ArgumentParser();p.add_argument('--current',type=Path);a=p.parse_args();root=Path(__file__).resolve().parent
h=lambda b:hashlib.sha256(b).hexdigest();manifest=json.loads((root/'sha256.json').read_bytes())
assert all(h((root/n).read_bytes())==s for n,s in manifest.items())
for label,passed in [('success',True),('historical-failure',False)]:
 d=root/label;r=json.loads((d/'report.json').read_bytes());expected=r['source_hashes_before']
 assert r['passed']is passed and len(expected)==49 and r['source_hashes_after']==expected
 assert h((d/'sources.tar.gz').read_bytes())==r['source_archive_sha256']
 with tarfile.open(d/'sources.tar.gz')as t:
  rows=t.getmembers();assert len(rows)==49 and all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in rows)
  assert {m.name:h(t.extractfile(m).read())for m in rows}==expected
 t=json.loads((d/'terminal.json').read_bytes());assert t['session_reaped']and t['pid_absent']and t['exit_code']==(0 if passed else 1)
 if passed:
  assert sum(len(x['levels'])for x in r['profiles'])==150 and all(not x['registered_accepted']and x['actual_QP_secure']and x['generated_P_exact']for x in r['profiles'])
  assert r['forbidden_during_controls']==0 and r['guard_count']==r['installed_guard_denials']==37 and r['guards_restored']
  x=E.parse(d/'tests.xml');s=list(x.getroot().iter('testsuite'));assert sum(int(v.attrib['tests'])for v in s)==16
  assert all(int(v.attrib.get(k,0))==0 for v in s for k in ('failures','errors','skipped'))
  assert t['tests']['source_before']==t['tests']['source_after']==expected
  if a.current:assert {n:h((a.current/n).read_bytes())for n in expected}==expected
print(json.dumps(dict(passed=True,files=len(manifest),sources=49,prefixes=150,tests=16,current=a.current is not None)))
