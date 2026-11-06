"""Portable stdlib integrity/record verifier. Never imports archived candidates."""
import argparse,hashlib,json,os,stat,tarfile
import xml.etree.ElementTree as ET
from pathlib import Path,PurePosixPath
HERE=Path(__file__).absolute().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def safe(name):
 p=PurePosixPath(name)
 assert isinstance(name,str)and str(p)==name and not p.is_absolute()and '..'not in p.parts
 return HERE/name
def read(path,limit=128<<20):
 before=path.lstat();assert stat.S_ISREG(before.st_mode)and 0<=before.st_size<=limit
 state=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mode,s.st_mtime_ns,s.st_ctime_ns)
 fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
 try:
  assert state(os.fstat(fd))==state(before);blocks=[];size=0
  while size<=limit:
   b=os.read(fd,min(65536,limit+1-size))
   if not b:break
   blocks.append(b);size+=len(b)
  assert size==before.st_size and state(os.fstat(fd))==state(before)==state(path.lstat())
  return b''.join(blocks)
 finally:os.close(fd)
def doc(name):return json.loads(read(safe(name)))
def actual_files(root):
 result={}
 for path in sorted(root.rglob('*')):
  s=path.lstat();assert not stat.S_ISLNK(s.st_mode),'capsule/source aliases forbidden'
  if stat.S_ISREG(s.st_mode):result[path.relative_to(root).as_posix()]=path
  else:assert stat.S_ISDIR(s.st_mode)
 return result
def main(current=False):
 whitelist=doc('WHITELIST.json');assert whitelist==sorted(set(whitelist))
 assert set(actual_files(HERE))==set(whitelist)
 indexed={}
 for line in read(safe('SHA256SUMS')).decode().splitlines():
  pin,name=line.split('  ',1);assert len(pin)==64 and name not in indexed
  indexed[name]=pin;assert sha(read(safe(name)))==pin,name
 assert set(indexed)==set(whitelist)-{'SHA256SUMS'}
 origins=doc('COPY_ORIGINS.json')
 for name,row in origins.items():
  data=read(safe(name));assert sha(data)==row['sha256']and len(data)==row['bytes']
 assert len(origins)==doc('BUILD_PLAN.json')['copied_files']
 lineage=doc('source/lineage.json');base={};total=0
 with tarfile.open(safe(lineage['base_archive']))as archive:
  members=archive.getmembers();assert len(members)==len({m.name for m in members})==460
  for m in members:
   safe(m.name);assert m.isfile()and 0<=m.size<=64<<20
   total+=m.size;assert total<=128<<20
   base[m.name]=archive.extractfile(m).read();assert len(base[m.name])==m.size
 maps={};objects={}
 for stage,row in lineage['stages'].items():
  wanted=doc(row['map']);assert len(wanted)==int(stage)
  assert sha(read(safe(row['map'])))==row['manifest_sha256']
  reconstructed={n:b for n,b in base.items()if n not in row['remove']}
  for name,path in row['overrides'].items():reconstructed[name]=read(safe(path))
  assert {n:sha(b)for n,b in reconstructed.items()}==wanted
  objects[stage]=reconstructed;maps[stage]=wanted
 assert sha(read(safe(lineage['base_archive'])))==lineage['stages']['460']['historical_archive_sha256']
 runtime={n[8:]:pin for n,pin in maps['460'].items()if n.startswith('runtime/')};assert len(runtime)==403
 for stage in maps:
  assert {n[8:]:pin for n,pin in maps[stage].items()if n.startswith('runtime/')}==runtime
  assert objects[stage]['selected_tests.json']==objects['460']['selected_tests.json']
 assert len(maps['459'])-len(maps['458'])==1 and len(maps['460'])-len(maps['459'])==1
 assert {n for n in maps['458']if maps['458'][n]!=maps['459'][n]}=={'owned_child.py'}
 assert {n for n in maps['459']if maps['459'][n]!=maps['460'][n]}=={'SEAMS.json','TEST_SCOPE.md','cases_loader_engine.py','cases_process.py','fixture_base.py','owned_child.py'}
 runs=doc('RUNS.json');seen_stages=[]
 for run in runs:
  stage=run['stage'];seen_stages.append(stage);prefix=run['destination'];d=lambda n:doc(prefix+'/'+n)
  config=d('run_config.json');handoff=d('handoff.json')
  assert config['source_count']==int(stage)and config['runtime_count']==403
  assert config['source_manifest_sha256']==lineage['stages'][stage]['manifest_sha256']
  assert config['source_archive_sha256']==lineage['stages'][stage]['historical_archive_sha256']
  assert config['cpu_affinity']==[112,113,114,115]and config['pytest_config']=='/dev/null'
  for name,key in [('executed_driver.py','driver_sha256'),('worker_driver.py','worker_sha256'),('inspect_children.py','inspector_sha256'),('run_config.json','config_sha256'),('launch-proposal.json','proposal_sha256')]:
   assert sha(read(safe(prefix+'/'+name)))==handoff[key]
  for name,pin in handoff['pins'].items():
   if name=='sources.tar.gz':
    assert pin==lineage['stages'][stage]['historical_archive_sha256'];continue
   assert sha(read(safe(prefix+'/'+name)))==pin
  assert d('sources-before.json')==d('sources-after.json')==d('execution-after.json')==maps[stage]
  xml=ET.fromstring(read(safe(prefix+'/tests.xml')))
  counts={key:len(list(xml.iter(tag)))for key,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
  assert counts==run['counts']
  assert [dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')]==config['selected_xml_cases']
  audit=d('worker-audit.json');owner=d('worker-owner-terminal.json');collector=d('collector-terminal.json');live=d('live-observation.json')
  code=0 if stage=='460'else 1
  assert audit['collected_nodes']==config['selected_nodes']and len(config['selected_nodes'])==59
  assert audit['pytest_exit_code']==audit['planned_exit_code']==owner['exit_code']==code
  assert audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is False and audit['runtime_module_origins']=={}
  assert audit['runtime_sources_before']==audit['runtime_sources_after']==runtime
  assert owner['direct_wait_completed']is True and owner['child_reaped']is True and owner['pid_absent']is True and owner['group_absent']is True
  assert collector['pid']==collector['pgid']==owner['parent_pid']==owner['parent_pgid']==live['outer_pid']==live['outer_pgid']
  assert owner['worker_pid']==owner['observed_pgid']==owner['owned_session_pgid']==audit['pid']==audit['pgid']==live['worker_pid']==live['worker_pgid']
  assert collector['start_ticks']==owner['parent_start_ticks']==live['outer_start_ticks']
  assert owner['start_ticks']==audit['start_ticks']==live['worker_start_ticks']
  assert collector['planned_exit_code']==(0 if stage=='460'else 70)
  assert collector['normal_main_return']is(stage=='460')and collector['error_type']==(None if stage=='460'else 'FileNotFoundError')
  pins=d('owned-record-pins.json');groups={}
  assert len(pins)==run['child_records']
  for name,pin in pins.items():
   raw=read(safe(prefix+'/'+name),2<<20);assert sha(raw)==pin['sha256']and len(raw)==pin['bytes']
   folder,base_name=name.rsplit('/',1);assert base_name in ('result.json','child-terminal.json','parent-terminal.json','child.log')
   groups.setdefault(folder,{})[base_name]=raw
  assert len(groups)==59;normal=0;observed_cases=set();identities={collector['pid'],owner['worker_pid']}
  for folder,raws in groups.items():
   p=json.loads(raws['parent-terminal.json']);c=json.loads(raws['child-terminal.json']);case=p['case']
   assert case in config['selected_cases']and case not in observed_cases and c['case']==case;observed_cases.add(case)
   assert type(c['pid'])is int and c['pid']==c['pgid']==p['pid']==p['owned_session_pgid']
   assert p['observed_pgid']is None or p['observed_pgid']==c['pid']
   assert c['pid']not in identities;identities.add(c['pid'])
   assert p['direct_wait_completed']is True and p['reaped']is True and p['group_absent']is True
   assert p['starttime_ticks']is None or p['starttime_ticks']==c['starttime_ticks']
   if p['exit_code']==0:
    normal+=1;r=json.loads(raws['result.json'])
    assert r['passed']is True and r['case']==case and r['pid']==r['pgid']==c['pid']
    assert c['normal_return']is True and c['result_sha256']==p['result_sha256']==sha(raws['result.json'])
    assert p['child_terminal_sha256']==sha(raws['child-terminal.json'])
    assert r['backend_denials']==[]and r['definition_only_entry']is True
    for field in ('original_main_restored','temporary_names_removed','profile_restored','declared_exports_restored','source_unchanged'):assert r[field]is True
    for field in ('actual_execution_authorized','capacity_measured','native_backend_imported','genuine_cli_bootstrap','model_key_ciphertext_body_reads'):assert r[field]is False
   else:
    assert p['exit_code']==70 and 'result.json'not in raws and c['normal_return']is False and c['status']=='entered'
  assert normal==59-counts['failures']and len(identities)==61
  metadata=d('fixture-metadata.json');assert len(metadata)==run['fixture_metadata_count']
  assert all(v['body_read']is False and v['body_copied']is False for v in metadata.values())
  assert all(v['followed']is False for v in d('aliases.json').values())
  absence=doc('reviews/terminal/cnn'+stage+'/process-absence.json')
  assert len(absence)==61 and {r['pid']for r in absence}==identities
  assert all(r['pid_absent']is True and r['pgid_absent']is True for r in absence)
  # These are retained observations, not a new current process probe.
  if stage=='460':
   report=d('report.json');assert report['passed']is True and report['tests']==counts and report['source_count']==460
   assert report['validated_children']==59 and report['retained_regular_files']==236
   assert report['artifact_review_sha256']==sha(read(safe(prefix+'/child-artifact-review.json')))
  else:assert prefix+'/report.json'not in whitelist
 assert seen_stages==['458','459','460']
 assert doc('reviews/root/tool-terminal460.json')['exit_code']==0
 assert doc('reviews/root/tool-terminal460.json')['terminal_tool']=='5aa837'
 assert doc('reviews/terminal/cnn460/review.json')['tests_passed']is True
 summary=doc('SUMMARY.json')
 assert summary['actual_ready']is False and summary['fixture_bodies_included']is False and summary['runs_not_aggregated']is True
 if current:
  for stage,row in lineage['stages'].items():
   root=Path(row['current_source_root']);actual={n:sha(read(p,64<<20))for n,p in actual_files(root).items()}
   assert actual==maps[stage]
   assert sha(read(Path(row['historical_archive_path'])))==row['historical_archive_sha256']
 print(json.dumps(dict(passed=True,files=len(whitelist),indexed=len(indexed),copied=len(origins),reconstructed_source_counts=[458,459,460],runtime_count=403,
  results=['458: 0 PASS / 59 FAIL','459: 43 PASS / 16 FAIL','460: 59 PASS / 0 FAIL'],
  child_regular_records=[177,220,236],fixture_metadata_only=[0,6,14],historical_archive_bytes_bundled=['460'],
  current_public_sources_archives_checked=current,fixture_bodies_read=False,candidate_imports_tests=False),sort_keys=True))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--current',action='store_true');args=parser.parse_args();main(args.current)
