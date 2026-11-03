"""Public resource accounting, not measured peak memory or encrypted readiness."""
from pathlib import Path
import datetime,hashlib,json,shutil
HERE=Path(__file__).resolve().parent
plan=json.loads((HERE/'public_plan.json').read_text());profile=plan['profiles'][0];budget=profile['budgets'][2]
table_path=Path('/home/jianming_gatech/CROSS/cadathlon/validation/scoped_ntt_subset_cpu/large_tables.json')
tables=json.loads(table_path.read_text())
assert tables['profile']['q_towers']==profile['q'] and tables['profile']['p_towers']==profile['p']
info={}
for line in Path('/proc/meminfo').read_text().splitlines():
 name,rest=line.split(':',1)
 if name in ('MemTotal','MemAvailable','Shmem'):info[name+'_bytes']=int(rest.split()[0])*1024
cgroup=Path('/sys/fs/cgroup')/Path('/proc/self/cgroup').read_text().strip().split(':',2)[-1].lstrip('/')
cgroups={}
while cgroup!=Path('/sys/fs'):
 record={name:(cgroup/name).read_text().strip() for name in ('memory.max','memory.current','memory.high') if (cgroup/name).exists()}
 if record:cgroups[str(cgroup)]=record
 cgroup=cgroup.parent
filesystems={}
for path in ('/tmp','/dev/shm'):
 usage=shutil.disk_usage(path);filesystems[path]={'total_bytes':usage.total,'used_bytes':usage.used,'free_bytes':usage.free}
result={'scope':__doc__,'observed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'actual_profile':{k:profile[k] for k in ('degree','slots','depth','dnum','q_count','p_count','log2_QP','QP_integer_bits','P_product_bits','max_Q_partition_bits','P_covers_each_Q_partition','secure_128_classical')},
 'selected_budget':budget,'host_memory':info,'cgroup_ancestors':cgroups,'filesystems':filesystems,
 'actual_shape_prior_public_NTT_master_inventory':tables['masters'],
 'historical_table_report_sha256':hashlib.sha256(table_path.read_bytes()).hexdigest(),
 'historical_table_scope':'Public CPU tables for same exactQ/P; excludes controls/keys/ciphertexts/compiler. Not full-refresh HBM evidence.',
 'public_plan_sha256':hashlib.sha256((HERE/'public_plan.json').read_bytes()).hexdigest(),
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'admission':'CPU parameters and tmpfs key storage have no demonstrated preflight blocker. Full-slot numerical accuracy and actual memory/latency remain unvalidated; no launch occurred.',
 'component_composition_limits':['Ring-degree changes between existing components are not ciphertext-compatible; no ring relabeling is permitted.',
 'Full-slot bound16 proof would not establish bounds32/64 or arbitrary native BERT full-model domains.',
 'A one-refresh and square1e-3 gate is not a proof of accumulated BERT error or a local1e-6 component gate.',
 'Both denseApproxMod branches and their meta-correction are new workload relative to period128 evidence.'],
 'collector_plan':{'requested_backend':'cpu','affinity':[132,133,134,135],'OMP_NUM_THREADS':4,'OPENBLAS_NUM_THREADS':4,'MKL_NUM_THREADS':4,
 'timeout_s':None,'cleanup_grace_s':60.,'key_directory':'unique fd-owned child under /dev/shm','shared_parent_must_survive':True,
 'source_archive_readback':True,'child_source_closure_bound_to_parent':True,'all_slots_checked':65536,
 'input_range':[-16.,16.],'public_bound':16.,'meta_precision':10,'refresh_gate':1e-3,'square_gate':1e-3,
 'runtime_source_changes':False,'key_generation_or_encryption_launched':False}}
(HERE/'resource_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'raw_key_GiB':budget['raw_rotation_key_bytes']/2**30,'required_free_bytes':budget['key_store_free_space_preflight_bytes'],'host_memory':info,'filesystems':filesystems},indent=2))
