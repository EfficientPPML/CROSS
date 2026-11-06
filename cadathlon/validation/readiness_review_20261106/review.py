"""Read-only source/public-result recovery after original tool handles disappeared.
Does not import candidates, read workspace bodies, launch roles, or remove data.
"""
import ast, hashlib, io, json, math, os, stat, tarfile
from pathlib import Path
from datetime import datetime, timezone
BASE=Path('/tmp/bert-toy4-activation-preparation-20261105')
RUN=BASE/'setup-v1'
OUT=Path('/home/jianming_gatech/CROSS/cadathlon/validation/readiness_review_20261106')
MAP_SHA='ec2bb07eb7689b2b3ec6851b0c62c851200473afe716af27331d0655d3a0239e'
READS={}
def sha(b): return hashlib.sha256(b).hexdigest()
def read(p,h=None):
 p=Path(p); before=p.lstat(); assert stat.S_ISREG(before.st_mode) and before.st_size<64*1024*1024
 b=p.read_bytes(); after=p.lstat(); assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
 assert len(b)==before.st_size and (h is None or sha(b)==h),str(p)
 assert READS.setdefault(str(p),b)==b; return b
def parse(b):
 def pairs(xs):
  d={}
  for k,v in xs: assert k not in d;d[k]=v
  return d
 return json.loads(b,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def jd(p,h=None): return parse(read(p,h))
def check_tree(p,m):
 actual={x.relative_to(p).as_posix() for x in p.rglob('*') if x.is_file()}
 assert actual==set(m)
 for n,h in m.items(): assert not Path(n).is_absolute() and '..' not in Path(n).parts;read(p/n,h)
def check_tar(p,m,h):
 raw=read(p,h)
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
  ms=t.getmembers();assert len(ms)==len(m) and {x.name for x in ms}==set(m)
  for x in ms: assert x.isfile() and not x.issym() and not x.islnk() and sha(t.extractfile(x).read())==m[x.name]
 return sha(raw)
def absent(p):
 try: os.kill(p,0)
 except ProcessLookupError: return True
 return False
def process(pid,expected_start,required_command):
 p=Path('/proc')/str(pid)
 try:
  raw=(p/'stat').read_text(); fields=raw[raw.rindex(')')+2:].split(); cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except FileNotFoundError: return {'pid':pid,'present':False}
 assert int(fields[19])==expected_start and required_command in cmd,(pid,'identity mismatch')
 return {'pid':pid,'present':True,'start_ticks':int(fields[19]),'parent_pid':int(fields[1]),'pgid':int(fields[2]),'state':fields[0],'cpu_ticks':int(fields[11])+int(fields[12]),'command':cmd,'status':[s for s in (p/'status').read_text().splitlines() if s.startswith(('VmRSS:','Cpus_allowed_list:'))]}
manifest=jd(BASE/'source_manifest.json',MAP_SHA);assert len(manifest)==260
check_tree(BASE/'execution_source',manifest);check_tree(RUN/'execution',manifest)
assert jd(RUN/'source_hashes.json')==jd(RUN/'setup/source_hashes.json')==manifest
outer=jd(RUN/'report.json');stage=jd(RUN/'setup/report.json');worker=jd(RUN/'setup/worker/report.json')
for r in [outer,stage,worker]:
 assert r['stage']=='setup' and all(r.get(k) is True for k in ['passed','complete','source_verified','cleanup_completed','terminal_scope_completed','termination_scope_completed'])
 assert 'error' not in r and math.isfinite(r['elapsed_s']) and r['elapsed_s']>0
 assert r['workspace_identity']==[24,4072] and r['workspace']=='/dev/shm/cross-toy4-workspace-parent-6gtd51vm/toy4-diagnostic-e937f39ca30abd0e61beca4bfff74c19'
assert stage['source_hashes']==stage['source_hashes_end']==stage['execution_hashes_end']==outer['execution_hashes_end']==manifest
for r in [outer,worker]:
 assert r['source_hashes_start']==r['source_hashes_end']==manifest
 g=r['fresh_source_guard'];assert g['fresh'] is True and g['passed'] is True and g['source_count']==260 and g['manifest_sha256']==MAP_SHA
assert outer['execute'] and outer['workspace_retained'] and outer['next_phase_not_automatically_launched']
assert outer['role_attempts']==['setup'] and outer['role_retirement']=={'setup':True}
assert outer['roles']==[{'stage':'setup','report_sha256':sha(read(RUN/'setup/report.json')),'passed':True}]
assert stage['worker_verified'] and stage['worker_report_sha256']==sha(read(RUN/'setup/worker/report.json'))
assert outer['cleanup_errors']==stage['cleanup_errors']==worker['forbidden_attempts']==[]
child=stage['child'];assert child['pid']==child['process_group']==1624869
assert child['direct_child_reaped'] and child['group_empty_verified'] and child['cleanup_completed'] and child['direct_returncode']==0 and child['cleanup_errors']==[] and child['signals_sent']==[]
for r in worker['imports'].values(): assert manifest[r['path']]==r['sha256']
assert worker['loader_cleanup_completed'] and worker['private_permissions_verified']
assert (worker['private_array_count'],worker['public_array_count'],worker['rotation_count'])==(1,401,199)
assert worker['device']['platform']=='tpu' and worker['device']['kind']=='TPU v6 lite'
archives=[check_tar(BASE/'sources.tar.gz',manifest,'cf51aa50aa182b2544f0af583d5f3d322a619daa58b5fb414d9767248747c857'),check_tar(RUN/'sources.tar.gz',manifest,outer['source_archive_sha256']),check_tar(RUN/'setup/worker/sources.tar.gz',manifest,worker['source_archive_sha256'])]
profile=jd(BASE/'execution_source/profile.json');q,p=profile['q_towers'],profile['p_towers']
assert (profile['degree'],profile['dnum'],profile['composite_degree'],len(q),len(p))==(131072,4,2,100,24)
assert all(type(x) is int and x>1 for x in q+p)
alpha=(len(q)+3)//4;assert math.prod(p)>=max(math.prod(q[i:i+alpha]) for i in range(0,len(q),alpha))
t=ast.parse(read(BASE/'execution_source/runtime/jaxite_word/he_params.py'))
a,=[n.value for n in t.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='_OPENFHE_CEILINGS_RAW' for x in n.targets)]
key=ast.dump(ast.parse('(CostModel.CLASSICAL,128)',mode='eval').body,include_attributes=False)
table,=[v for k,v in zip(a.keys,a.values) if ast.dump(k,include_attributes=False)==key]
ceiling=ast.literal_eval(table)[131072];assert ceiling==3523 and math.prod(q+p).bit_length()<=ceiling
security=worker['key_generation_preflight']['security'];assert security['target_bits']==128 and security['attack']=='classical' and security['P_coverage']
assert math.isclose(security['actual_log2_qp'],sum(math.log2(x) for x in q+p),abs_tol=1e-9,rel_tol=0)
request=worker['request_contract'];assert request['nonce']=='e937f39ca30abd0e61beca4bfff74c19' and request['context_id']==worker['key_context_id']
assert request['output_count']==96 and request['request_count']==1 and request['absolute_gate']==1e-6 and request['original_decoder_stddev_limit']==.03125
assert request['profile_sha256']==sha(read(BASE/'execution_source/profile.json')) and request['numerical_plan_sha256']==sha(read(BASE/'execution_source/public_plan/plan.json'))
plan=jd(BASE/'execution_source/public_plan/plan.json');assert plan['profile']==profile and request['case_sha256']==plan['case_sha256'] and request['input_pattern_sha256']==plan['checkpoints'][0]['expected_array_sha256']
read(BASE/'setup-launch-v1.json','96c97c984178f2671012e8b1bb0f164d3b23e12593d4d74f2c07654e5ea8928e')
read(BASE/'setup-launch-root-live-readback.json','d6ee5b06dd42d0d6ec4f73e7817b1ea52c703050e5195c8937a1dbf5e66aaab7')
assert all(absent(x) for x in [1624866,-1624866,1624869,-1624869])
retained=[]
for path,inode in [(Path('/dev/shm/cross-toy4-workspace-parent-6gtd51vm'),4071),(Path(worker['workspace']),4072)]:
 st=path.lstat(); assert stat.S_ISDIR(st.st_mode) and (st.st_dev,st.st_ino,st.st_uid,st.st_gid,stat.S_IMODE(st.st_mode))==(24,inode,2002,2002,448)
 retained.append({'path':str(path),'dev':24,'inode':inode,'uid':2002,'gid':2002,'mode':448})
processes=[process(1688839,7745136,'bert-server-compute-stage4-successor-v2-20261103'),process(1179041,5853026,'bert-fullslot-p60-refresh-candidate-20261103'),process(2613514,11948644,'bert-prepared-bsgs-server-actual-v1-20261104')]
qreport=jd('/tmp/qwen-expanded-candidate-checker-actual-v1-20261104/worker/report.json')
qsummary={k:qreport[k] for k in ['pid','complete','passed','cleanup_completed','acceptance_requires_independent_exit0_reap_and_exact_report_hash']}
qsummary.update({'worker_absent':absent(qreport['pid']),'original_tool_terminal_receipt_available':False,'numerical_acceptance':False,'private_outcomes_opened':False})
p56key=Path('/dev/shm/cross-transformer-case-kjo6kb0k');p56st=p56key.lstat();assert (p56st.st_dev,p56st.st_ino)==(24,420)
result={'schema':'root-readiness-public-recovery-v1','observed_utc':datetime.now(timezone.utc).isoformat(),'goal_complete':False,'original_session_handles_unavailable':[24171,49839,98012,77543,85433],'worker_processes':processes,'toy4_setup':{'public_reports_consistent':True,'original_sources_unchanged':260,'source_trees_checked':2,'archives_checked':archives,'source_map_sha256':MAP_SHA,'collector_reported_elapsed_seconds':outer['elapsed_s'],'worker_reported_elapsed_seconds':worker['elapsed_s'],'worker_direct_exit0_and_reap_recorded':True,'original_collector_tool_exit_code':None,'original_collector_tool_terminal_receipt_available':False,'collector_and_worker_process_groups_absent':True,'retained_workspace_directories':retained,'device':worker['device'],'security':security,'request_contract':request,'setup_report_sha256':sha(read(RUN/'setup/report.json')),'server_launched':False,'validation_launched':False,'strict_terminal_admission':False,'limits':['Verified public setup result only. Original collector exit/reap cannot be reconstructed from report success and current process absence.','No workspace key/model/ciphertext bodies read; no server/decode/numerical success claim.','Retained directories were not removed. Original strict terminal receipt gate remains unsatisfied.']},'p56':{'original_worker_absent':absent(1002892),'original_wrapper_absent':absent(1002891),'owned_key_directory_still_present':True,'owned_key_directory_identity':[24,420],'terminal_result_admitted':False,'numerical_body_read':False},'qwen_calibration':qsummary,'scope':'Public recovery and source integrity; missing tool handles do not establish success, failure or cleanup of other workers. No job restarted, cancelled, launched or deleted.'}
read(Path(__file__))
for s,b in READS.items(): assert Path(s).read_bytes()==b,s
OUT.mkdir(exist_ok=False)
for name,source in [('toy4-collector-report.json',RUN/'report.json'),('toy4-setup-report.json',RUN/'setup/report.json'),('toy4-worker-report.json',RUN/'setup/worker/report.json'),('toy4-source-manifest.json',BASE/'source_manifest.json'),('toy4-sources.tar.gz',RUN/'sources.tar.gz'),('toy4-launch.json',BASE/'setup-launch-v1.json'),('toy4-initial-observation.json',BASE/'setup-launch-root-live-readback.json'),('review.py',Path(__file__))]:
 (OUT/name).write_bytes(READS[str(source)])
(OUT/'report.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
(OUT/'verified-source-and-public-pins.json').write_text(json.dumps({s:sha(b) for s,b in READS.items()},indent=2,sort_keys=True)+'\n')
print(json.dumps({'path':str(OUT/'report.json'),'sha256':sha((OUT/'report.json').read_bytes()),'source_and_public_files_verified':len(READS),'toy4_public_reports_consistent':True,'strict_terminal_admission':False,'live_original_workers':[x['pid'] for x in processes if x['present']],'qwen_public_report_passed':qsummary['passed'],'qwen_numerical_acceptance':False}))
