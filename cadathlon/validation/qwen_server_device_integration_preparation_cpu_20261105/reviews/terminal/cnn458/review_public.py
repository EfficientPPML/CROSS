import ast,hashlib,json,os,stat,tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
proof=Path('/tmp/qwen-full28-device-focused-proof-20261104/preparation-v2')
source=Path('/tmp/qwen-full28-device-focused-tests-20261104')
out=Path('/tmp/qwen-full28-device-failed-security-peer-20261105');out.mkdir(exist_ok=True)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canon=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def read(path,limit=2<<20):
 before=path.lstat();assert stat.S_ISREG(before.st_mode)and before.st_size<=limit
 state=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mode,v.st_mtime_ns,v.st_ctime_ns)
 fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  assert state(os.fstat(fd))==state(before);blocks=[];size=0
  while size<=limit:
   b=os.read(fd,min(65536,limit+1-size))
   if not b:break
   blocks.append(b);size+=len(b)
  assert size==before.st_size and state(os.fstat(fd))==state(before)==state(path.lstat())
  return b''.join(blocks)
 finally:os.close(fd)
def inv(root):
 result={}
 for p in sorted(root.rglob('*')):
  assert not p.is_symlink()
  if p.is_file():result[p.relative_to(root).as_posix()]=sha(p.read_bytes())
 return result
public_names=['sources-before.json','sources-after.json','execution-after.json','worker-audit.json','worker-early.json','worker-terminal.json','worker-owner-terminal.json','collector-terminal.json','live-observation.json','tests.xml','tests.log','run_config.json','handoff.json','launch-proposal.json','executed_driver.py','worker_driver.py','inspect_children.py']
raws={n:read(proof/n)for n in public_names};pins={n:dict(path=str(proof/n),sha256=sha(raw),bytes=len(raw))for n,raw in raws.items()}
expected=json.loads(raws['sources-before.json']);assert len(expected)==458
assert inv(source)==inv(proof/'execution')==expected
assert json.loads(raws['sources-after.json'])==json.loads(raws['execution-after.json'])==expected
with tarfile.open(proof/'sources.tar.gz')as a:
 members=a.getmembers();assert len(members)==len({m.name for m in members})==458 and all(m.isfile()for m in members)
 assert {m.name:sha(a.extractfile(m).read())for m in members}==expected
config=json.loads(raws['run_config.json']);assert sha((proof/'sources.tar.gz').read_bytes())==config['source_archive_sha256']
parentmap=json.loads((source/'runtime_sources.json').read_bytes());assert len(parentmap)==403 and {n[8:]:h for n,h in expected.items()if n.startswith('runtime/')}==parentmap
xml=ET.fromstring(raws['tests.xml']);cases=list(xml.iter('testcase'))
counts={key:len(list(xml.iter(tag)))for key,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==dict(tests=59,failures=59,errors=0,skipped=0)
assert [dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in cases]==config['selected_xml_cases']
audit=json.loads(raws['worker-audit.json']);worker=json.loads(raws['worker-owner-terminal.json']);collector=json.loads(raws['collector-terminal.json']);live=json.loads(raws['live-observation.json'])
assert audit['collected_nodes']==config['selected_nodes']and audit['pytest_exit_code']==audit['planned_exit_code']==1
assert audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is False
assert audit['runtime_sources_before']==audit['runtime_sources_after']==parentmap
assert worker['exit_code']==1 and worker['direct_wait_completed']is True and worker['child_reaped']is True and worker['pid_absent']is True and worker['group_absent']is True
assert collector['error_type']=='FileNotFoundError'and collector['planned_exit_code']==70 and collector['normal_main_return']is False
assert collector['pid']==collector['pgid']==worker['parent_pid']==worker['parent_pgid']==900657
assert worker['worker_pid']==worker['observed_pgid']==worker['owned_session_pgid']==900659
assert collector['start_ticks']==worker['parent_start_ticks']==live['outer_start_ticks']==20100084
assert worker['start_ticks']==live['worker_start_ticks']==20100111
records={};aliases={};owned_pins={};base=proof/'pytest-owned'
for child in sorted(base.iterdir()):
 info=child.lstat()
 if stat.S_ISLNK(info.st_mode):
  target=os.readlink(child);assert 0<len(target)<=4096 and '\\'not in target
  t=Path(target);assert str(t)==target and '..'not in t.parts and '.'not in t.parts
  if t.is_absolute():named=t
  else:
   assert target not in('.', '..')and '/'not in target;named=child.parent/target
  assert named.parent==base and named!=child and stat.S_ISDIR(named.lstat().st_mode)
  aliases[child.name]=dict(target=target,followed=False);continue
 assert stat.S_ISDIR(info.st_mode)
 names={p.name for p in child.iterdir()};assert names=={'parent-terminal.json','child-terminal.json','child.log'}
 data={n:read(child/n)for n in sorted(names)}
 p=json.loads(data['parent-terminal.json']);c=json.loads(data['child-terminal.json']);log=data['child.log'].decode()
 case=p['case'];assert case in config['selected_cases']and case not in records and c['case']==case
 assert p['status']=='terminal'and p['exit_code']==70 and p['direct_wait_completed']is True and p['reaped']is True and p['group_absent']is True
 assert c['status']=='entered'and c['normal_return']is False
 assert type(c['pid'])is int and c['pid']==c['pgid']==p['pid']==p['owned_session_pgid']
 assert p['observed_pgid']is None or p['observed_pgid']==c['pgid']
 assert type(c['starttime_ticks'])is int and(p['starttime_ticks']is None or p['starttime_ticks']==c['starttime_ticks'])
 assert "UnboundLocalError: cannot access local variable 'importlib'"in log and 'spec=importlib.util.spec_from_file_location' in log
 for n,raw in data.items():owned_pins[(child/n).relative_to(proof).as_posix()]=dict(sha256=sha(raw),bytes=len(raw))
 records[case]=dict(parent=p,child=c,log_sha256=sha(data['child.log']),failure='same pre-helper importlib local-binding error',result_present=False)
assert set(records)==set(config['selected_cases'])and len(records)==59 and len(owned_pins)==177
assert not list((proof/'tmp').iterdir())and not(proof/'report.json').exists()
identities=[dict(pid=collector['pid'],pgid=collector['pgid'],role='collector'),dict(pid=worker['worker_pid'],pgid=worker['owned_session_pgid'],role='worker')]+[dict(pid=v['child']['pid'],pgid=v['child']['pgid'],role=case)for case,v in records.items()]
assert len({r['pid']for r in identities})==61
for row in identities:
 for field,call in [('pid',os.kill),('pgid',os.killpg)]:
  try:call(row[field],0)
  except ProcessLookupError:row[field+'_absent']=True
  else:raise AssertionError('recorded process/group remains present')
# Only source AST is evaluated, not candidate definitions.
tree=ast.parse((source/'owned_child.py').read_bytes());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
inner=[n for n in ast.walk(main)if isinstance(n,ast.Import)and any(a.name=='importlib'for a in n.names)];assert len(inner)==1
first=next(n for n in ast.walk(main)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='spec_from_file_location')
assert first.lineno<inner[0].lineno
review=dict(schema='full28-failed59-independent-public-review-v1',review_passed=True,tests_passed=False,source_ready=False,source_count=458,runtime_count=403,counts=counts,worker_normal_exit_code=1,collector_exit_code_parent_reported=70,collector_error_type='FileNotFoundError',root_direct_owner=dict(session=20699,launch_tool='f209b7',terminal_tool='2d61ff',authority='parent message; this peer did not wait/reap'),child_failure_count=59,child_exit_code=70,child_result_count=0,early_child_records=59,parent_direct_wait_records=59,retained_regular_records=177,aliases=len(aliases),pid_group_pairs_absent=61,cause='inner import importlib makes main local name unbound at earlier importlib.util use before helper execution',candidate_behavior_tested=False,actual_authority=False,scope='bounded public source/log/early-owner review only; no imports/rerun/body reads')
for name,value in [('review.json',review),('public-pins.json',pins),('owned-record-pins.json',owned_pins),('owned-records.json',records),('aliases.json',aliases),('process-absence.json',identities)]: (out/name).write_bytes(canon(value))
(out/'review_public.py').write_bytes(Path(__file__).read_bytes())
assert inv(source)==inv(proof/'execution')==expected
assert all(read(proof/n)==raw for n,raw in raws.items())
print('review',sha((out/'review.json').read_bytes()));print('59 identical pre-helper setup failures; exact458/403; all61 PID/groups absent; no candidate imports/reruns')
