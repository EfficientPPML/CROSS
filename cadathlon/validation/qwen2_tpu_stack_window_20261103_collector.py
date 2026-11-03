"""Short nonblocking stack-only observation of the already-running public proof."""
from pathlib import Path
import hashlib,json,os,resource,subprocess,time
pid=612486
base=Path('/tmp/qwen2-stack-sampling-20261103')
proof=Path('/tmp/cross-bounded-qwen2-tpu-20261103/report.json')
r=json.loads(proof.read_text())
source_hashes=r['source_hashes_start']
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items())
command=['sudo','-n','/home/jianming_gatech/miniconda3/envs/jaxite/bin/py-spy','record','--pid',str(pid),'--duration','20','--rate','10','--format','raw','--threads','--idle','--nonblocking','--output',str(base.with_suffix('.raw'))]
started=time.monotonic();usage=resource.getrusage(resource.RUSAGE_CHILDREN)
record={'scope':__doc__,'pid':pid,'command':command,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'source_archive_sha256':r['source_archive_sha256'],'proof_source_count':len(source_hashes),'proof_sources_unchanged_before':True,'sampling_semantics':'10Hz Python stack snapshots including idle/waiting threads; not native CPU attribution or a whole-request performance measurement. Nonblocking collection can miss or race frames. No locals, no native stacks and no signals that pause target execution requested.','target_overhead':'Profiler wall/child CPU cost is measured below; perturbation of the running target is not separately measured, so these are not calibration timings.'}
process=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=False)
record.update(returncode=process.returncode,elapsed_s=time.monotonic()-started)
after=resource.getrusage(resource.RUSAGE_CHILDREN)
record.update(observer_child_user_cpu_s=after.ru_utime-usage.ru_utime,observer_child_system_cpu_s=after.ru_stime-usage.ru_stime,proof_sources_unchanged_after=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source_hashes.items()),raw_sha256=hashlib.sha256(base.with_suffix('.raw').read_bytes()).hexdigest())
base.with_suffix('.log').write_text(process.stdout)
base.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(process.returncode)
