from pathlib import Path
import hashlib,io,json,os,tarfile,xml.etree.ElementTree as ET
p=Path('/tmp/qwen-heldout1088-runtime-preflight-v2-20261104');lab=Path('/tmp/qwen-heldout1088-runtime-v2-20261104')
sha=lambda x:hashlib.sha256(x).hexdigest()
raw=(p/'source_manifest.json').read_bytes();m=json.loads(raw);rraw=(p/'report.json').read_bytes();r=json.loads(rraw);wraw=(p/'worker/report.json').read_bytes();w=json.loads(wraw)
assert len(m)==167 and sha(raw)=='cce2630028708f41e79614bb367039fd146fc924471bfd674ade1737f9c6017e'
for row in (r,w):
 for k in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'):assert row[k]is True,k
 assert row['audit_denials']==[] and 'error_type'not in row
 assert row['source_hashes_start']==row['source_hashes_end']==m
 assert row['source_manifest_sha256']==sha(raw)
assert r['worker']==w and r['worker_report_sha256']==sha(wraw)
assert type(r['child_returncode'])is int and r['child_returncode']==0 and r['child_reaped']is True and r['child_group_absent']is True
assert w['tests']==478 and w['failures']==w['errors']==w['skipped']==w['pytest_returncode']==0
assert w['installed_import_denials']==['torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla']
x=ET.parse(p/'worker/tests.xml');assert len(x.findall('.//testcase'))==478
assert not any(x.findall('.//'+k)for k in('failure','error','skipped'))
for root in (lab,p/'execution_source'):
 actual={q.relative_to(root).as_posix():sha(q.read_bytes())for q in root.rglob('*')if q.is_file()}
 assert actual==m
 assert not any(q.is_symlink()for q in root.rglob('*'))
a=(p/'sources.tar.gz').read_bytes();assert sha(a)==r['source_archive']['sha256']and len(a)==r['source_archive']['bytes']
with tarfile.open(fileobj=io.BytesIO(a),mode='r:gz')as t:
 members=t.getmembers();assert len(members)==len(m)and all(x.isfile()for x in members)
 assert {x.name:sha(t.extractfile(x).read())for x in members}==m
parent=json.loads((lab/'parent139_sources.json').read_text());assert len(parent)==139 and all(m[n]==h for n,h in parent.items())
observed_processes={'collector':{'pid':2938023,'ppid':16918,'pgid':2938023},'worker':{'pid':2938141,'ppid':2938023,'pgid':2938141}}
assert r['pid']==observed_processes['collector']['pid'] and r['child_pid']==w['pid']==observed_processes['worker']['pid']
processes={}
for role,record in observed_processes.items():
 for kind in ('pid','pgid'):
  number=record[kind]
  try:(os.kill if kind=='pid'else os.killpg)(number,0)
  except ProcessLookupError:processes[f'{kind}:{number}']='absent'
  else:raise AssertionError((kind,number,'still present'))
result=dict(schema='qwen-heldout-preflight-terminal-readback-v1',passed=True,source_count=167,parent139_unchanged=True,tests=478,installed_denials=10,current_execution_start_end_archive_equal=True,
 report_sha256=sha(rraw),worker_report_sha256=sha(wraw),source_archive_sha256=sha(a),source_manifest_sha256=sha(raw),processes=processes,observed_processes=observed_processes,process_observation_tool_chunk='6f9d12',
 collector_parent_scope='PPID16918 observed; no assertion that it is an owned workload or should retire',tool_wrapper_pid=None,
 actual_model_validation_or_private_body_reads=False,tool_session=62369,tool_exit_code=0,tool_reaped=True,tool_terminal_chunk='47d970')
(p/'terminal_observation.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(result,sort_keys=True))
