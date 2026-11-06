import pathlib,os,stat,json,hashlib,tarfile,xml.etree.ElementTree as ET,datetime
r=pathlib.Path('/tmp/qwen-full28-device-focused-proof-20261104/preparation-v2');sha=lambda b:hashlib.sha256(b).hexdigest();pins={}
def read(p,limit=2<<20):
 s=p.lstat();assert stat.S_ISREG(s.st_mode)and s.st_uid==os.geteuid()and 0<=s.st_size<=limit
 fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC);state=lambda z:(z.st_dev,z.st_ino,z.st_mode,z.st_size,z.st_mtime_ns,z.st_ctime_ns)
 try:
  assert state(s)==state(os.fstat(fd));raw=b''
  while len(raw)<=limit:
   b=os.read(fd,min(65536,limit+1-len(raw)))
   if not b:break
   raw+=b
  assert len(raw)==s.st_size and state(s)==state(os.fstat(fd))==state(p.lstat())
 finally:os.close(fd)
 pins[p.relative_to(r).as_posix()]={'bytes':len(raw),'sha256':sha(raw)};return raw
def gone(pid):
 for fn in(os.kill,os.killpg):
  try:fn(pid,0)
  except ProcessLookupError:continue
  raise AssertionError('original process/group remains')
expected=json.loads(read(r/'sources-before.json'));assert len(expected)==458
for name in('sources-after.json','execution-after.json'):assert json.loads(read(r/name))==expected
for root in(r/'execution',pathlib.Path('/tmp/qwen-full28-device-focused-tests-20261104')):assert {p.relative_to(root).as_posix():sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}==expected
with tarfile.open(r/'sources.tar.gz')as t:assert {m.name:sha(t.extractfile(m).read())for m in t.getmembers()}==expected
config=json.loads(read(r/'run_config.json'));live=json.loads(read(r/'live-observation.json'));collector=json.loads(read(r/'collector-terminal.json'));owner=json.loads(read(r/'worker-owner-terminal.json'));worker=json.loads(read(r/'worker-terminal.json'));audit=json.loads(read(r/'worker-audit.json'));early=json.loads(read(r/'worker-early.json'))
assert collector['pid']==collector['pgid']==live['outer_pid']==live['outer_pgid']==900657 and collector['start_ticks']==live['outer_start_ticks']==20100084
assert collector['planned_exit_code']==70 and collector['normal_main_return']is False and collector['error_type']=='FileNotFoundError'
assert owner['direct_wait_completed']and owner['child_reaped']and owner['pid_absent']and owner['group_absent']and owner['exit_code']==1
assert all(v['pid']==v['pgid']==900659 and v['start_ticks']==20100111 for v in(worker,audit,early));assert owner['worker_pid']==live['worker_pid']==900659
assert audit['collected_nodes']==config['selected_nodes']and audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is False
xml=ET.fromstring(read(r/'tests.xml'));assert len(list(xml.iter('testcase')))==len(list(xml.iter('failure')))==59;assert not list(xml.iter('error'))and not list(xml.iter('skipped'))
read(r/'tests.log');base=r/'pytest-owned';children=[];aliases=[]
for p in sorted(base.iterdir()):
 info=p.lstat()
 if stat.S_ISLNK(info.st_mode):
  target=os.readlink(p);tp=pathlib.Path(target);named=tp if tp.is_absolute()else base/tp
  assert named.parent==base and str(named)==target and '..'not in named.parts
  assert stat.S_ISDIR(named.lstat().st_mode);aliases.append({'name':p.name,'target':target,'followed':False});continue
 assert stat.S_ISDIR(info.st_mode);assert set(v.name for v in p.iterdir())=={'child.log','child-terminal.json','parent-terminal.json'}
 child=json.loads(read(p/'child-terminal.json'));parent=json.loads(read(p/'parent-terminal.json'));log=read(p/'child.log').decode()
 assert child['case']==parent['case']and child['case']in config['selected_cases'];assert child['pid']==child['pgid']==parent['pid']==parent['owned_session_pgid'];assert child['status']=='entered'and child['normal_return']is False
 assert parent['exit_code']==70 and parent['direct_wait_completed']and parent['reaped']and parent['group_absent']and parent['child_report_read']is False
 assert parent['starttime_ticks']is None or parent['starttime_ticks']==child['starttime_ticks'];assert 'UnboundLocalError' in log and 'importlib' in log and 'spec=importlib.util.spec_from_file_location' in log
 gone(child['pid']);children.append({'case':child['case'],'pid':child['pid'],'pgid':child['pgid'],'starttime_ticks':child['starttime_ticks'],'exit_code':70,'original_wait_complete':True,'pid_group_absent':True})
assert len(children)==59 and {v['case']for v in children}==set(config['selected_cases']);gone(900657);gone(900659)
assert not(r/'report.json').exists();assert not list((r/'tmp').iterdir())
out={'schema':'full28-focused-first59-root-failed-terminal-v1','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'proof_root':str(r),'tool_session':20699,'launch_tool':'f209b7','terminal_tool':'2d61ff','collector':collector,'worker_owner':owner,'test_items':59,'pytest_failures':59,'pytest_errors':0,'pytest_skipped':0,'reported_pytest_seconds':2.06,'candidate_test_bodies_reached':False,'failure':'owned_child.main local importlib assignment shadows module import before first importlib.util call; all59 children reject before sourcehelper execution/candidate imports','collector_failure':'success-only inspector requests missing result.json after failed child; no report or pass manufactured','source458_current_execution_archive_unchanged':True,'children':children,'bounded_regular_child_records':177,'fixture_files':0,'aliases':aliases,'public_pins':pins,'all61original_pids_and_groups_absent':True,'no_rerun_or_frozen_changes':True,'reviewer_history':['101f05 attempted live proc stat after rapidcollector exit and wrote no receipt/journal; terminal2d61ff then boundedread3073a9 admitted diagnosis','c16c10 stopped at reviewer relative-only alias assumption before any write; beda5f confirmed actual pytest absolute sibling aliases; this metadata-only review uses literal absolute sibling names plus lstat only'],'scope':'failed recording harness bootstrap only; no native/backend/model/key/ciphertext/capacity or HE evidence'}
p=pathlib.Path('/tmp/qwen-full28-device-focused-root-harness-review-20261105/failed-terminal-review.json');p.write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print('receipt',p,sha(p.read_bytes()),'children',len(children),'pins',len(pins),'records',177)
j=pathlib.Path('/tmp/root-turn-progress-20261104-0937.json');x=json.loads(j.read_text());x['continuation_20261104_2357']['full59_first_run']={'root':str(r),'session_id':20699,'launch_tool':'f209b7','terminal_tool':'2d61ff','exit_code':70,'worker_exit':1,'test_items':59,'allfailed_before_candidate_imports':True,'pytest_seconds':2.06,'receipt':str(p),'sha256':sha(p.read_bytes()),'all61originalpidsgroupsabsent':True,'records':177,'fixtures':0,'source458unchanged':True,'next':'CNN independentfailedterminalreview then narrow source successor removinglocalimportlib; rootreviewbeforeanyrerun; inspector must allowlexical absolute siblingpytestaliases'};j.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n');print('journal',sha(j.read_bytes()))
