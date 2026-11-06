import json,hashlib,os,tarfile,datetime
from pathlib import Path
import xml.etree.ElementTree as ET
sha=lambda b:hashlib.sha256(b).hexdigest()
p=Path('/tmp/qwen-model-store-readonly-focused-proof-20261104/preparation-v1');root=Path(__file__).parent
h=json.loads((p/'handoff.json').read_bytes());assert sha((p/'handoff.json').read_bytes())=='e33fb6e12cf5c44754e7ba655c1b719aada1334dd98a62d2cbfdbfe777ffb5fd'
for n,d in h['files'].items():assert sha((p/n).read_bytes())==d,n
owner_raw=(p/'owner-terminal.json').read_bytes();assert sha(owner_raw)==h['owner_terminal_sha256']=='bc252201ce44537fc04cab38ed5dddc864c6b5be749e7e3a410c66920f9af9c2';o=json.loads(owner_raw)
assert o['session_id']==20688 and o['authoritative_wait_tool_chunk']=='751336' and type(o['tool_exit_code'])is int and o['tool_exit_code']==0 and o['tool_session_reaped']is True
config=json.loads((p/'run_config.json').read_bytes());r=json.loads((p/'report.json').read_bytes());a=json.loads((p/'worker-audit.json').read_bytes());live=json.loads((p/'live-observation.json').read_bytes());launch=json.loads((root/'readonly-launch-review.json').read_bytes())
for key,reportkey in [('source_manifest_sha256','source_manifest_sha256'),('source_archive_sha256','source_archive_sha256'),('driver_sha256','driver_sha256'),('worker_sha256','worker_driver_sha256'),('config_sha256','config_sha256')]:assert r[reportkey]==launch[key]
assert r['passed']is True and r['child_returncode']==0 and r['child_reaped']is r['child_group_absent']is True
expected=json.loads((p/'sources-before.json').read_bytes())
for name in ['sources-before.json','sources-after.json','execution-after.json']:assert json.loads((p/name).read_bytes())==expected
for s in [Path(config['source_root']),p/'execution']:
 assert not any(f.is_symlink()for f in s.rglob('*'));assert{f.relative_to(s).as_posix():sha(f.read_bytes())for f in s.rglob('*')if f.is_file()}==expected
with tarfile.open(p/'sources.tar.gz')as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==54;assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in ms);assert{m.name:sha(t.extractfile(m).read())for m in ms}==expected
rt=json.loads(Path('/tmp/qwen-model-store-readonly-sources-v2-20261104.json').read_bytes());assert len(rt)==42 and all(expected['runtime/'+n]==d for n,d in rt.items())
actual=Path('/tmp/qwen-model-store-readonly-v2-20261104');assert{f.relative_to(actual).as_posix():sha(f.read_bytes())for f in actual.rglob('*')if f.is_file()}==rt
xml=ET.parse(p/'tests.xml').getroot();cases=[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')];counts={k:len(list(xml.iter(tag)))for k,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==r['tests']==h['tests']==dict(tests=62,failures=0,errors=0,skipped=0)
assert cases==r['selected_xml_cases']==config['selected_xml_cases']and a['collected_nodes']==r['selected_nodes']==config['selected_nodes']
assert a['denied']==a['loaded_backend']==[] and type(a['pytest_exit_code'])is int and a['pytest_exit_code']==0
assert a['pid']==live['worker_pid']==41757 and a['pgid']==live['worker_pgid']==41757 and live['outer_pid']==41756
assert a['affinity']==[108,109,110,111]
assert a['runtime_sources_before']==a['runtime_sources_after']==config['runtime_source_expectations']
for path,d in config['runtime_source_expectations'].items():assert sha(Path(path).read_bytes())==d
assert a['runtime_module_origins']=={n:dict(file=path,spec=path)for n,path in config['runtime_module_origins'].items()}
def absent(pid,pgid):
 assert not(Path('/proc')/str(pid)).exists()
 try:os.killpg(pgid,0)
 except ProcessLookupError:return True
 raise AssertionError('group still present')
for role in ['outer','worker']:assert absent(live[role+'_pid'],live[role+'_pgid'])
assert len(o['child_terminals'])==15
for row in o['child_terminals']:
 raw=(p/'test-child-terminals'/row['file']).read_bytes();assert sha(raw)==row['sha256'];d=json.loads(raw)
 assert d['reaped']is d['process_group_absent']is True and d['pid']==row['pid']and d['pgid']==row['pgid']and d['exit_code']==row['exit_code'];assert absent(d['pid'],d['pgid'])
report=dict(schema='qwen-readonly-focused-root-terminal-review-v1',passed=True,source_count=54,runtime_count=42,tests=counts,source_manifest_sha256=launch['source_manifest_sha256'],source_archive_sha256=launch['source_archive_sha256'],launch_review_sha256=sha((root/'readonly-launch-review.json').read_bytes()),report_sha256=sha((p/'report.json').read_bytes()),owner_terminal_sha256=sha(owner_raw),handoff_sha256=sha((p/'handoff.json').read_bytes()),xml_sha256=sha((p/'tests.xml').read_bytes()),audit_sha256=sha((p/'worker-audit.json').read_bytes()),source_current_execution_archive_exact=True,all_owned_processes_absent=True,test_child_receipts=15,authoritative_owner_session=20688,authoritative_wait_tool='751336',independent_live_observation=False,independent_live_limit='Root6f6276 saw processes already absent after3.46second proof; no live or wrapper identity claim.',actual_model_integrity_admitted=False,actual_model_execution_authorized=False,scope=r['scope'],script_sha256=sha(Path(__file__).read_bytes()),utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
out=root/'readonly-terminal-review.json';out.write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(str(out),sha(out.read_bytes()))
