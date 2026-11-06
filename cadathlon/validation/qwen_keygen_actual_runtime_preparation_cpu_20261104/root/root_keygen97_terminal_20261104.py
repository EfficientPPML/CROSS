from pathlib import Path
import hashlib,json,os,stat,tarfile,datetime
import xml.etree.ElementTree as ET
sha=lambda x:hashlib.sha256(x).hexdigest()
proof=Path('/tmp/qwen-keygen-actual-focused-proof-20261104/preparation-v1');dest=Path('/tmp/qwen-key-generation-actual-root-review-20261104/keygen97-terminal-review.json')
fd=os.open(proof/'report.json',os.O_RDONLY|os.O_NOFOLLOW)
try:
 before=os.fstat(fd);assert stat.S_ISREG(before.st_mode)
 raw=b''
 while True:
  b=os.read(fd,1<<20)
  if not b:break
  raw+=b
 after=os.fstat(fd);named=(proof/'report.json').lstat()
 state=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
 assert state(before)==state(after)==state(named)
finally:os.close(fd)
report=json.loads(raw);config=json.loads((proof/'run_config.json').read_bytes());expected=json.loads((proof/'sources-before.json').read_bytes())
assert report['passed']is True and report['child_returncode']==0 and report['child_reaped']is True and report['child_group_absent']is True
assert report['tests']==dict(tests=61,failures=0,errors=0,skipped=0)
assert all(report[n]is False for n in ('actual_execution_authorized','actual_key_generation','actual_client_role_execution','backend_context_construction','capacity_acquisition_authorized','genuine_backend_imports_authorized','model_execution','encrypted_execution','numpy_imported'))
assert report['synthetic_only']is True and report['sources_unchanged']is True
for root in (Path(config['source_root']),proof/'execution'):
 assert not any(p.is_symlink()for p in root.rglob('*'))
 assert {p.relative_to(root).as_posix():sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}==expected
for name in ('sources-after.json','execution-after.json'):assert(proof/name).read_bytes()==(proof/'sources-before.json').read_bytes()
assert sha((proof/'sources.tar.gz').read_bytes())==config['source_archive_sha256']
with tarfile.open(proof/'sources.tar.gz')as tar:
 members=tar.getmembers();assert len(members)==len({m.name for m in members})==97
 assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
 assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
xml=ET.parse(proof/'tests.xml').getroot();cases=[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')]
assert cases==config['selected_xml_cases']==report['selected_xml_cases']
assert all(len(list(xml.iter(n)))==0 for n in('failure','error','skipped'))
audit=json.loads((proof/'worker-audit.json').read_bytes());live=json.loads((proof/'live-observation.json').read_bytes())
assert audit['collected_nodes']==config['selected_nodes']==report['selected_nodes']
assert audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is False
assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']=={}
assert audit['affinity']==[112,113,114,115]and audit['pytest_exit_code']==0
assert audit['pid']==audit['pgid']==live['worker_pid']==live['worker_pgid']
absent={}
for role in ('worker','outer'):
 pid=live[role+'_pid'];assert not Path('/proc',str(pid)).exists();absent[role+'_pid']=True
 try:os.killpg(live[role+'_pgid'],0)
 except ProcessLookupError:absent[role+'_group']=True
 else:raise AssertionError('group remains')
for name,key in (('executed_driver.py','driver_sha256'),('worker_driver.py','worker_driver_sha256'),('run_config.json','config_sha256')):assert sha((proof/name).read_bytes())==report[key]
files=('report.json','tests.log','tests.xml','worker-audit.json','live-observation.json','sources-before.json','sources-after.json','execution-after.json','sources.tar.gz','run_config.json','executed_driver.py','worker_driver.py')
result=dict(schema='qwen-keygen97-root-terminal-admission-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,tests=61,elapsed_s=report['elapsed_s'],source_count=97,source_current_execution_before_after_archive_exact=True,xml_order_and_audit_exact=True,direct_tool_chunk='03b09d',direct_exit_code=0,direct_reaped=True,process_absence=absent,independent_live_observation=False,live_observation_limit='Run completed within launch call; owner live receipt retained, root verified terminal absence only.',held_report_fd_stable=True,evidence_sha256={n:sha((proof/n).read_bytes())for n in files},review_script_sha256=sha(Path(__file__).read_bytes()),scope='Synthetic focused checks only; temporary name doubles explicitly disclosed. No genuine backend/NumPy, actual keygen, model/key/CT body access, capacity or full HE claim.')
with dest.open('x')as f:f.write(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(path=str(dest),sha256=sha(dest.read_bytes()),tests=61,elapsed_s=report['elapsed_s'],report_sha256=sha(raw))))
