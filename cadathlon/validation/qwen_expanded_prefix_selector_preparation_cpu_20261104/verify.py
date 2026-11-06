from pathlib import Path,PurePosixPath
import argparse,hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
h=lambda b:hashlib.sha256(b).hexdigest()
j=lambda p:json.loads(p.read_bytes())
a=argparse.ArgumentParser();a.add_argument('--current',action='store_true');args=a.parse_args()
files={f.relative_to(P).as_posix():f for f in P.rglob('*')if f.is_file()and f.name!='files.sha256.json'and '__pycache__'not in f.parts};index=j(P/'files.sha256.json');assert set(files)==set(index)and all(h(f.read_bytes())==index[n]for n,f in files.items())
r=j(P/'preflight/report.json');assert h((P/'preflight/report.json').read_bytes())=='2d40fd917432437b5ae7931e74425a5199fa852dd8739e783755fd0a0be8cada'
assert all(r[k]is True for k in('passed','complete','source_files_unchanged','source_archive_verified','child_reaped','child_group_absent','synthetic_selection_exercised'))and r['child_returncode']==0
assert all(r[k]is False for k in('actual_observer_admission','actual_coefficient_selection','model_or_HE_executed'))
assert r['tests']==66 and r['xml_bad']==0 and r['candidate_source_count']==18 and r['proof_source_count']==19
before=r['source_hashes_before'];assert len(before)==18 and before==r['source_hashes_after']==r['execution_hashes']==j(P/'preflight/source_manifest.json')
archive=P/'preflight/sources.tar.gz';assert h(archive.read_bytes())==r['source_archive_sha256']=='d32da07f6942e06225124867490459d85e05bafceba7b8101d2c30fd9f06d1cb'
with tarfile.open(archive)as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==19 and sum(m.size for m in ms)<16<<20;raw={}
 for m in ms:
  n=PurePosixPath(m.name);assert m.isfile()and not n.is_absolute()and '..'not in n.parts and m.size<8<<20;raw[m.name]=t.extractfile(m).read()
assert {n:h(b)for n,b in raw.items()if n!='external_preflight.py'}==before and raw['external_preflight.py']==(P/'preflight_driver.py.txt').read_bytes()
assert json.loads(raw['execution_authorization.json'])=={'actual_coefficient_selection':False,'actual_observer_admission':False}
origins=json.loads(raw['source_origins.json']);assert all(h(raw[n])==v['sha256']for n,v in origins.items())
assert h(raw['inputs/candidate3_manifest.json'])=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
audit=j(P/'preflight/audit.json');assert audit==r['audit']and audit['passed']is True and audit['denials']==[]and audit['model_backend_modules']==[]and len(audit['loaded_local_modules'])==8
assert all(before[v['path']]==v['sha256']for v in audit['loaded_local_modules'].values())
cases=ET.parse(P/'preflight/tests.xml').findall('.//testcase');assert len(cases)==66 and not any(c.find(k)is not None for c in cases for k in('failure','error','skipped'))
terminal=j(P/'preflight/terminal_observation.json');assert terminal['session_id']==43452 and terminal['observed_exit_code']==0 and terminal['authoritatively_reaped']is True and terminal['child_pid_absent']is True
for v in(1,2,3):
 old=j(P/f'history/preflight-v{v}/report.json');assert old['passed']is False and old['source_files_unchanged']is True and old['source_archive_verified']is True
for v,failed in((1,1),(4,1),(5,0)):
 cases=ET.parse(P/f'history/development-v{v}/tests.xml').findall('.//testcase');assert sum(c.find('failure')is not None for c in cases)==failed
if args.current:
 root=Path('/tmp/qwen-expanded-prefix-selector-preparation-20261104');assert {n:h((root/n).read_bytes())for n in before}==before
 assert Path('/tmp/check_qwen_selector_preflight_v4_20261104.py').read_bytes()==raw['external_preflight.py']
print(json.dumps(dict(passed=True,indexed_files=len(files),candidate_sources=18,proof_sources=19,synthetic_tests=66,current=args.current,actual_selection=False)))
