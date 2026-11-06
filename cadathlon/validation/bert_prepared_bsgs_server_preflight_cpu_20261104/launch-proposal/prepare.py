"""Public metadata/stat/source/resource proposal only. Does not execute command."""
from pathlib import Path
import datetime,hashlib,json,os,shlex,tarfile
HERE=Path('/tmp/bert-prepared-bsgs-launch-proposal-20261104');HERE.mkdir(exist_ok=False)
LAB=Path('/tmp/bert-prepared-bsgs-server-component-20261104');PROOF=Path('/tmp/bert-prepared-bsgs-server-preflight-v2-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
report=json.loads((PROOF/'report.json').read_bytes());expected=report['source_hashes_start']
assert len(expected)==286 and expected==report['source_hashes_end']
assert {n:sha((LAB/n).read_bytes())for n in expected}==expected
with tarfile.open(PROOF/'sources.tar.gz')as t:assert {m.name:sha(t.extractfile(m).read())for m in t}==expected
pins=[('key',Path('/tmp/bert-full-key-generation-component-20261103/actual-v1'),'022cbdf687273e765668184f32cb7cd501d371c4220916e1c8f02574cf3a7f08'),('model',Path('/tmp/bert-public-model-stage2-v2-component-20261103/actual-v2'),'5b9589b51828d066706c4c46a6404d8c52fcd67a2c9aa9f3f57c73df8caeefe0'),('stage3',Path('/tmp/bert-input-encryption-stage3-component-v3-20261103/actual-v3'),'add5e6a3143be3931ad29c018cdee40bdf059497b8bfa987114f39baeb88631e')]
for _,p,h in pins:assert sha((p/'report.json').read_bytes())==h
key=Path('/dev/shm/cross-full-bert-keys-143b3f5c1573427d841760a3c9d5d33d');model=Path('/tmp/cross-bert-public-model-b51668856dab42a3846ba1b7c7a71d53');ioroot=Path('/tmp/cross-bert-input-encryption-7b255c2357c94060b73e474a1f8ef5fa')
owned=Path('/tmp/bert-prepared-bsgs-server-actual-v1-20261104');response=Path('/tmp/cross-bert-prepared-bsgs-response-20261104-v1');log=Path(str(owned)+'.log')
for p in (owned,response,log):assert not p.exists()and not p.is_symlink()
protected=(LAB,key,model,ioroot,*[p for _,p,_ in pins]);
for p in (owned,response):
 for q in protected:assert p.resolve()!=q.resolve()and q.resolve()not in p.resolve().parents and p.resolve()not in q.resolve().parents
assert owned not in response.parents and response not in owned.parents
mem={k:int(v.split()[0])*1024 for k,v in (line.split(':',1)for line in Path('/proc/meminfo').read_text().splitlines())if v.strip().endswith('kB')}
resources={}
for p in (Path('/tmp'),Path('/dev/shm')):
 s=os.statvfs(p);i=p.stat();resources[str(p)]=dict(identity=[i.st_dev,i.st_ino],free_bytes=s.f_bavail*s.f_frsize,total_bytes=s.f_blocks*s.f_frsize)
required=(6<<20)+(1<<30);assert resources['/tmp']['free_bytes']>=required
identities={}
for name,p in (('key_root',key),('model_root',model),('original_io',ioroot),('borrowed_upload',ioroot/'upload')):
 s=p.lstat();identities[name]=dict(path=str(p),identity=[s.st_dev,s.st_ino])
assert identities['key_root']['identity']==[24,1563]and identities['model_root']['identity']==[66305,2383041]and identities['borrowed_upload']['identity']==[66305,2383744]
env={'PATH':'/usr/local/bin:/usr/bin:/bin','HOME':'/home/jianming_gatech','LANG':'C.UTF-8','JAX_PLATFORMS':'cpu','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','NUMEXPR_NUM_THREADS':'4','VECLIB_MAXIMUM_THREADS':'4','PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
python='/tmp/cross-pretrained-validation-env/bin/python'
command=['env','-i',*[k+'='+v for k,v in env.items()],'taskset','-c','100-103',python,str(LAB/'combined_collect.py'),'--execute','--output',str(owned),'--response-root',str(response),'--worker-python',python,'--key-workspace',str(key),'--model-workspace',str(model),'--io-workspace',str(ioroot)]
for name,p,h in pins:command+=['--'+name+'-evidence',str(p),'--'+name+'-report-sha256',h]
result=dict(schema='combined-server-launch-proposal-v1',prepared_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,actual_launch_authorized=False,command_executed=False,
 source_count=286,source_inventory_sha256=sha((PROOF/'source_manifest.json').read_bytes()),source_report_sha256=sha((PROOF/'report.json').read_bytes()),source_archive_sha256=sha((PROOF/'sources.tar.gz').read_bytes()),source_current_equal=True,
 client_preflight=dict(report_sha256='c695723f88ced9e66a1dc134bc0277e7e50e4f07de463659bfb6f278814637b2',sources=115,tests=198,installed_denials=21,actual_decode=False),
 prerequisite_reports={name:dict(evidence=str(p),sha256=h)for name,p,h in pins},borrowed_root_identities=identities,
 cpu_affinity=[100,101,102,103],threads=4,command=command,working_directory=str(LAB),stdout_stderr_log=str(log),evidence_root=str(owned),separate_response_root=str(response),new_outputs_absent=True,
 memory_snapshot=dict(mem_available_bytes=mem['MemAvailable'],mem_free_bytes=mem['MemFree'],swap_total_bytes=mem.get('SwapTotal',0),peak_memory_bound_established=False),storage_snapshot=resources,new_response_storage_required_bytes=required,
 logical_prepared_cache=dict(owners=['graph','linear'],entries_per_owner=2,bytes_per_owner=2<<30,aggregate_logical_bytes=4<<30,refresh='ordinary'),
 existing_public_store=dict(available_rotation_keys=130,required_actual_union=118,missing=[],public_array_npy_bytes=125380363136,new_key_generation=False),
 exclusions=['Logical cache cap excludes mmap/master backing, ordinary refresh caches, formatted views, compiler/control/conversion temporaries, ciphertext/intermediate storage, and physical allocator overhead.','Available memory is a point-in-time host observation, not a peak-allocation proof.','Original ordinary server download and private request/completed ledger are neither used nor retired.','No actual server, decryption, new keys, private inputs or tensor/ciphertext bodies read by this proposal.','No latency/accuracy/ranked claim.'],
 lifecycle=dict(healthy_timeout=None,failure_term_grace_seconds=30,failure_kill_grace_seconds=30,exact_execution_copy=True,all_actual_failures_retain_response=True,retained_upload_only_checked=True,source_manifest_recheck_immediately_before_launch_required=True,completed_server_required_before_client=True))
(HERE/'proposal.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');(HERE/'command.txt').write_text('NOT EXECUTED; awaiting root grant.\n'+shlex.join(command)+' > '+shlex.quote(str(log))+' 2>&1\n')
(HERE/'prepare.py').write_bytes(Path(__file__).read_bytes());h={p.name:sha(p.read_bytes())for p in HERE.iterdir()if p.is_file()};(HERE/'files.sha256.json').write_text(json.dumps(h,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(HERE),proposal_sha256=h['proposal.json'],mem_available_bytes=mem['MemAvailable'],tmp_available_bytes=resources['/tmp']['free_bytes'],required_new_bytes=required)))
