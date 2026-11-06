from pathlib import Path,PurePosixPath
import argparse,hashlib,json,tarfile,io,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
h=lambda b:hashlib.sha256(b).hexdigest()
j=lambda p:json.loads(p.read_bytes())
a=argparse.ArgumentParser();a.add_argument('--current',action='store_true');args=a.parse_args()
files={f.relative_to(P).as_posix():f for f in P.rglob('*')if f.is_file()and f.name!='files.sha256.json'and '__pycache__'not in f.parts}
index=j(P/'files.sha256.json');assert set(files)==set(index)and all(h(f.read_bytes())==index[n]for n,f in files.items())
def tar_bytes(path):
 raw=path.read_bytes();out={};total=0
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as t:
  for m in t:
   total+=m.size;n=PurePosixPath(m.name)
   assert m.isfile()and not n.is_absolute()and '..'not in n.parts and m.name not in out and 0<=m.size<=16<<20 and total<=32<<20
   out[m.name]=t.extractfile(m).read()
 return out
r=j(P/'preflight/report.json');assert h((P/'preflight/report.json').read_bytes())=='804fea6b0fbacaaaa226d50962e3e535ce6bdff62e0348c9febc896bfa120c3d'
assert all(r[k]is True for k in('passed','complete','source_files_unchanged','source_archive_verified','child_reaped','child_group_absent','synthetic_selection_exercised','synthetic_actual_path_exercised'))
assert all(r[k]is False for k in('actual_observer_admission','actual_observer_evidence_read','actual_coefficient_selection','model_or_HE_executed','root_grant_issued'))
assert r['tests']==150 and r['xml_bad']==0 and r['child_returncode']==0 and r['candidate_source_count']==36 and r['proof_source_count']==37
before=r['source_hashes_before'];assert len(before)==36 and before==r['source_hashes_after']==r['execution_hashes']==j(P/'preflight/source_manifest.json')==j(P/'reviewed_source_manifest.json')
assert h((P/'reviewed_source_manifest.json').read_bytes())=='f3d0bf68c503c2baba2cc34b7806c152efae153bbc180a4ebb85aaeae2d833f0'
raw=tar_bytes(P/'preflight/sources.tar.gz');assert len(raw)==37
assert h((P/'preflight/sources.tar.gz').read_bytes())==r['source_archive_sha256']=='4e76d85e6a3ce0f8f2570a225f86a63f22a1b101e7039b2fb900988f248e0a13'
assert {n:h(b)for n,b in raw.items()if n!='external_preflight.py'}==before and raw['external_preflight.py']==(P/'preflight_driver.py.txt').read_bytes()
parent=json.loads(raw['parent_sources.json']);assert len(parent)==18 and all(before[n]==pin for n,pin in parent.items())
assert json.loads(raw['selection_readiness.json'])['ready']is False
assert h(raw['inputs/reconstruction_script.py.txt'])=='3690b5710a8f019f529062303289d415fad99d85bb579ff426e66ec9633ff2e3'
assert h(raw['inputs/observer_sources.json'])=='8131c5ac045f1cf274a493bbc14278e330c0a8e057079edc4f10f9a23dd9bf91'
assert h(raw['inputs/candidate3_manifest.json'])=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
assert h(raw['inputs/norm_multipliers.json'])=='761a07e232e0bba981ba014eb6d202cce96a9471616743594fecdc6a57d67840'
audit=j(P/'preflight/audit.json');assert audit==r['audit']and audit['passed']is True and audit['denials']==[]and audit['model_backend_modules']==[]and len(audit['loaded_local_modules'])==18 and len(audit['synthetic_child_launches'])==8
assert all(before[v['path']]==v['sha256']for v in audit['loaded_local_modules'].values())
cases=ET.parse(P/'preflight/tests.xml').findall('.//testcase');assert len(cases)==150 and not any(c.find(k)is not None for c in cases for k in('failure','error','skipped'))
t=j(P/'preflight/terminal_observation.json');assert t['session_id']==61520 and t['observed_exit_code']==0 and all(t[k]is True for k in('authoritatively_reaped','collector_pid_absent','child_pid_absent','child_group_absent'))
old=j(P/'history/preflight-v1/report.json');assert old['passed']is False and old['tests']==150 and old['xml_bad']==11 and old['source_files_unchanged']is True and old['source_archive_verified']is True and old['source_hashes_before']==before
oldraw=tar_bytes(P/'history/preflight-v1/sources.tar.gz');assert {n:h(b)for n,b in oldraw.items()if n!='external_preflight.py'}==before
assert oldraw['external_preflight.py']==(P/'history/preflight_driver_v1.py.txt').read_bytes()
for name,failed in(('negative-v1',3),('successor-v1',0)):
 q=P/'review'/('root-cleanup-'+name);review=j(q/'review.json');m=tar_bytes(q/'sources.tar.gz')
 assert h((q/'sources.tar.gz').read_bytes())==review['source_archive_sha256']
 assert {n:h(b)for n,b in m.items()}==review['source_hashes_before']==review['source_hashes_after']
 counts=j(q/'test_counts.json');assert counts['tests']==4 and counts['failures']==failed and counts['errors']==0
assert j(P/'review/transformer/reviewed_sources.json')==before
for n,pin in j(P/'review/security.json')['sources'].items():assert before[n]==pin
handoff=tar_bytes(P/'pure-proof-handoff/sources.tar.gz');assert len(handoff)==26 and {n:h(b)for n,b in handoff.items()}==j(P/'pure-proof-handoff/source_manifest.json')
assert all(before[n]==h(b)for n,b in handoff.items())
if args.current:
 root=Path('/tmp/qwen-expanded-prefix-selector-admission-v2-20261104')
 assert {f.relative_to(root).as_posix():h(f.read_bytes())for f in root.rglob('*')if f.is_file()}==before
 assert Path('/tmp/check_qwen_selector_admission_preflight_v2_20261104.py').read_bytes()==raw['external_preflight.py']
print(json.dumps(dict(passed=True,indexed_files=len(files),candidate_sources=36,proof_sources=37,synthetic_tests=150,current=args.current,actual_selection_grant_issued=False)))
