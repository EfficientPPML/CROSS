import json,hashlib,os,tarfile,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
sha=lambda b:hashlib.sha256(b).hexdigest()
p=Path('/tmp/qwen-keygen-authority-boundary-proof-20261104/preparation-v1');root=Path(__file__).parent
hraw=(p/'terminal-handoff.json').read_bytes();assert sha(hraw)=='53037bd05667d42c437fee628e64bef2d358ab75f0ca8cd2d1f0d60b83909592';h=json.loads(hraw)
oraw=(p/'owner-terminal.json').read_bytes();assert sha(oraw)==h['owner_receipt_sha256']=='55faf18e48c75dda9c788d9d0fe5d66865d4189db0c578d7762413fa77e69708';o=json.loads(oraw)
for n,d in o['files'].items():assert sha((p/n).read_bytes())==d,n
assert o['owner_session']==22566 and o['owner_terminal_chunk']=='c646b3' and type(o['owner_tool_exit_code'])is int and o['owner_tool_exit_code']==0 and o['owner_reaped']is True
config=json.loads((p/'run_config.json').read_bytes());r=json.loads((p/'report.json').read_bytes());a=json.loads((p/'worker-audit.json').read_bytes());live=json.loads((p/'live-observation.json').read_bytes());launch=json.loads((root/'keygen-launch-review.json').read_bytes())
for key,reportkey in [('source_manifest_sha256','source_manifest_sha256'),('source_archive_sha256','source_archive_sha256'),('driver_sha256','driver_sha256'),('worker_sha256','worker_driver_sha256'),('config_sha256','config_sha256')]:assert r[reportkey]==launch[key]
assert r['passed']is True and r['child_returncode']==0 and r['child_reaped']is r['child_group_absent']is True
expected=json.loads((p/'sources-before.json').read_bytes())
for name in ['sources-before.json','sources-after.json','execution-after.json']:assert json.loads((p/name).read_bytes())==expected
for s in [Path(config['source_root']),p/'execution']:
 assert not any(f.is_symlink()for f in s.rglob('*'));assert{f.relative_to(s).as_posix():sha(f.read_bytes())for f in s.rglob('*')if f.is_file()}==expected
with tarfile.open(p/'sources.tar.gz')as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==39;assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in ms);assert{m.name:sha(t.extractfile(m).read())for m in ms}==expected
parent=Path('/tmp/qwen-keygen-authority-boundary-20261104');pm=json.loads(Path('/tmp/qwen-keygen-authority-boundary-sources-v1-20261104.json').read_bytes());assert len(pm)==28 and all(expected[n]==d for n,d in pm.items());assert{f.relative_to(parent).as_posix():sha(f.read_bytes())for f in parent.rglob('*')if f.is_file()}==pm
xml=ET.parse(p/'tests.xml').getroot();cases=[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')];counts={k:len(list(xml.iter(tag)))for k,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==r['tests']==o['tests']==dict(tests=48,failures=0,errors=0,skipped=0)
assert cases==r['selected_xml_cases']==config['selected_xml_cases']and a['collected_nodes']==r['selected_nodes']==config['selected_nodes']
assert a['denied']==a['loaded_backend']==[] and type(a['pytest_exit_code'])is int and a['pytest_exit_code']==0
assert a['pid']==live['worker_pid']==43440 and a['pgid']==live['worker_pgid']==43440 and live['outer_pid']==43435
assert a['affinity']==[112,113,114,115]
assert a['runtime_sources_before']==a['runtime_sources_after']==config['runtime_source_expectations']=={}
assert a['runtime_module_origins']==config['runtime_module_origins']=={}
assert a['forbidden_roots']==config['forbidden_roots']and 'numpy'in a['forbidden_roots']and a['numpy_imported']is r['numpy_imported']is False
for role in ['outer','worker']:
 assert not(Path('/proc')/str(live[role+'_pid'])).exists()
 try:os.killpg(live[role+'_pgid'],0)
 except ProcessLookupError:pass
 else:raise AssertionError('group still present')
report=dict(schema='qwen-keygen-authority-focused-root-terminal-review-v1',passed=True,source_count=39,parent_count=28,tests=counts,source_manifest_sha256=launch['source_manifest_sha256'],source_archive_sha256=launch['source_archive_sha256'],launch_review_sha256=sha((root/'keygen-launch-review.json').read_bytes()),report_sha256=sha((p/'report.json').read_bytes()),owner_terminal_sha256=sha(oraw),handoff_sha256=sha(hraw),xml_sha256=sha((p/'tests.xml').read_bytes()),audit_sha256=sha((p/'worker-audit.json').read_bytes()),source_current_execution_archive_exact=True,all_owned_processes_absent=True,authoritative_owner_session=22566,authoritative_wait_tool='c646b3',independent_root_live_observation=False,independent_live_limit='Root fca9c1 saw processes already absent; owner d56d20 separately saw both active; owner223595 later ps race preserved. No root live identity claim.',actual_model_integrity_admitted=False,actual_keygen_authorized=False,scope=r['scope'],script_sha256=sha(Path(__file__).read_bytes()),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=root/'keygen-terminal-review.json';out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(str(out),sha(out.read_bytes()))
