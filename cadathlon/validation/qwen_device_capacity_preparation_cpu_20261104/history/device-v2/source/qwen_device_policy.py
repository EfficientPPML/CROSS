"""Pure, externally pinned TPU policy authentication. No backend or I/O.

There are no policy instances, numeric defaults or live-capacity claims here.
Independent factual review and the source-fixed process remain trust inputs.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import PurePosixPath

ROLES=('resource_policy','resource_review','dependency_policy','selection','allocator','access','workspace','evidence','review','lease')
STAGES=('post_init','post_keys','pre_evaluation','terminal')
HOOKS=('device_capacity_check_bindings','device_capacity_check_host','device_capacity_check_lease','device_capacity_initialize','device_capacity_synchronize')
MODULES=('qwen_device_policy.py','qwen_device_capacity.py','qwen_server_process.py')
COVERAGE=('exclusive_physical_memory','logical_physical_device_mapping','trusted_native_access_inventory',
    'exact_plugin_and_allocator_semantics','initialization_host_dma_fd_costs','all335_public_array_lifetimes',
    'lazy_rotation_and_ntt_caches','refresh_and_compiler_workspace','async_and_executable_lifetimes',
    'retained_inputs_intermediates_and38_outputs','stage_additional_memory_not_double_counted',
    'fragmentation_and_untracked_reserve','one_device_per_ciphertext','host_and_device_admission_separate')

def canonical(value):return(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def obj(value,fields,label):
    if type(value)is not dict or set(value)!=set(fields):raise ValueError('closed '+label+' fields required')
    return value
def integer(value,minimum,label):
    if type(value)is not int or value<minimum:raise ValueError('exact '+label+' integer required')
    return value
def text(value,label,limit=4096):
    if type(value)is not str or not 0<len(value)<=limit or '\x00'in value:raise ValueError('bounded '+label+' string required')
    return value
def digest(value):
    if type(value)is not str or len(value)!=64 or any(x not in'0123456789abcdef'for x in value):raise ValueError('exact SHA256 required')
    return value
def path(value):
    text(value,'path');p=PurePosixPath(value)
    if not p.is_absolute()or str(p)!=value or '..'in p.parts or value.startswith('//')or'\\'in value or p==p.parent or any(c in value for c in'*?[]'):raise ValueError('canonical literal absolute nonroot path required')
    return value
def same(actual,expected,label):
    if canonical(actual)!=canonical(expected):raise ValueError(label)
def parse(raw,label):
    if type(raw)is not bytes or not 0<len(raw)<=256*1024:raise ValueError('bounded immutable '+label+' bytes required')
    def pairs(rows):
        result={}
        for k,v in rows:
            if k in result:raise ValueError('duplicate JSON key')
            result[k]=v
        return result
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in()).throw(ValueError('nonfinite JSON')))
    if type(value)is not dict or canonical(value)!=raw:raise ValueError('canonical compact '+label+' required')
    return value
def hashes(value,fields,label):
    obj(value,fields,label)
    for v in value.values():digest(v)
def version(value,wanted):
    if value!=wanted or type(value)is not str:raise ValueError('exact schema required')

@dataclass(frozen=True)
class DevicePolicyAdmission:
    phase:object
    blobs:tuple
    expected_pins:bytes
    @property
    def execution_authorized(self):return False
    def check(self):return authenticate(self.phase,self.blobs,expected_pins=self.expected_pins)

def authenticate(phase,blobs,*,expected_pins):
    """Expected pins are independent launch inputs, never harvested from blobs."""
    from qwen_server_admission import ServerPhaseAdmission
    from qwen_server_resource import authenticate_policy
    if type(phase)is not ServerPhaseAdmission:raise TypeError('exact admitted server phase required')
    phase.check();pp=phase.expected_pins.check();grant=parse(phase.grant_bytes,'server phase')
    pins=parse(expected_pins,'independent device pins')
    obj(pins,('schema','source_manifest_sha256','source_archive_sha256','phase_grant_sha256','documents'),'device pins')
    version(pins['schema'],'qwen-server-device-pins-v1');hashes(pins['documents'],ROLES,'device document hashes')
    for n in('source_manifest_sha256','source_archive_sha256','phase_grant_sha256'):
        digest(pins[n]);same(pins[n],pp[n],'device/phase '+n+' differs')
    same(sha(phase.grant_bytes),pins['phase_grant_sha256'],'original phase grant differs')
    for n in MODULES:
        if n not in pp['source_map']:raise ValueError('device component missing from selected server closure')
    if type(blobs)is not tuple or tuple(x[0]for x in blobs if type(x)is tuple and len(x)==2)!=ROLES:raise ValueError('exact immutable ordered device proof tuple required')
    if any(type(row)is not tuple or len(row)!=2 or type(row[1])is not bytes for row in blobs):raise ValueError('exact immutable document bytes required')
    raw=dict(blobs)
    for name in ROLES:
        if sha(raw[name])!=pins['documents'][name]:raise ValueError('external '+name+' hash differs')
    documents={n:parse(raw[n],n)for n in ROLES}
    dep=documents['dependency_policy']
    obj(dep,('schema','interpreter','resolved_interpreter','python_major_minor','cores','environment','installed_sources','library_roots','backend_module_origins'),'selected dependency policy')
    version(dep['schema'],'qwen-server-tpu-dependencies-v1')
    for role,field in(('dependency_policy','dependency_policy_sha256'),('resource_policy','resource_policy_sha256'),('resource_review','resource_review_sha256')):
        same(sha(raw[role]),pp['policies'][field],'device/phase policy differs')
        same(sha(raw[role]),grant['policies'][field],'device/granted policy differs')
    rp={n:pins[n]for n in('source_manifest_sha256','source_archive_sha256','phase_grant_sha256')}
    rp.update({field:sha(raw[role])for role,field in(('dependency_policy','dependency_policy_sha256'),('resource_policy','resource_policy_sha256'),('resource_review','resource_review_sha256'))})
    resource=authenticate_policy(phase,raw['resource_policy'],raw['resource_review'],expected_pins=canonical(rp))
    d=resource['device'];same(d['platform'],'tpu','only selected TPU supported')
    same(d['selection_sha256'],sha(raw['selection']),'resource selection bytes differ')
    same(d['allocator_policy_sha256'],sha(raw['allocator']),'resource allocator bytes differ')

    s=documents['selection']
    obj(s,('schema','host_identity','physical_identity','device','process_count','physical_devices_per_ciphertext','environment','access_sha256','workspace_sha256'),'selection')
    version(s['schema'],'qwen-tpu-device-selection-v1');text(s['host_identity'],'host identity');text(s['physical_identity'],'physical memory identity')
    same(s['process_count'],1,'dedicated single-process selection required');same(s['physical_devices_per_ciphertext'],1,'single-device ciphertext selection required')
    obj(s['device'],('platform','id','process_index','local_hardware_id','device_kind'),'selected device')
    same(s['device']['platform'],'tpu','genuine TPU required');text(s['device']['device_kind'],'device kind')
    for n in('id','process_index','local_hardware_id'):integer(s['device'][n],0,'device '+n)
    same(s['device']['process_index'],0,'one local process required')
    if type(s['environment'])is not dict or not s['environment']:raise ValueError('exact reviewed initialization environment required')
    for k,v in s['environment'].items():text(k,'environment name',256);text(v,'environment value',8192)
    same(s['environment'],dep['environment'],'selection/full dependency environment differs')
    same(s['access_sha256'],sha(raw['access']),'selection/access differs');same(s['workspace_sha256'],sha(raw['workspace']),'selection/workspace differs')

    a=documents['allocator']
    obj(a,('schema','dependency_policy_sha256','jax_module','required_stats','semantics','configuration','evidence_sha256'),'allocator')
    version(a['schema'],'qwen-tpu-allocator-policy-v1');same(a['dependency_policy_sha256'],sha(raw['dependency_policy']),'allocator/dependency differs')
    obj(a['jax_module'],('path','sha256'),'genuine JAX module');path(a['jax_module']['path']);digest(a['jax_module']['sha256'])
    installed=dep['installed_sources'];origins=dep['backend_module_origins']
    if type(installed)is not dict or type(origins)is not dict:raise ValueError('actual dependency source/origin inventory required')
    jax_path=a['jax_module']['path'];jax_row=installed.get(jax_path)
    obj(jax_row,('sha256','size'),'installed JAX source');integer(jax_row['size'],1,'JAX source size')
    same(jax_row['sha256'],a['jax_module']['sha256'],'allocator/JAX source pin differs')
    jax_origin=origins.get('jax');obj(jax_origin,('file','spec','search_locations'),'selected JAX origin')
    same(jax_origin['file'],jax_path,'JAX file origin differs');same(jax_origin['spec'],jax_path,'JAX spec origin differs')
    same(a['required_stats'],['bytes_in_use','bytes_limit'],'genuine required allocator observations differ')
    same(a['semantics'],'selected_allocator_with_exclusive_physical_memory','explicit allocator interpretation required')
    # Empty is an explicit reviewed native-default configuration, not an
    # invented TPU allocator environment flag or a numeric capacity default.
    if type(a['configuration'])is not dict:raise ValueError('reviewed allocator configuration required')
    for k,v in a['configuration'].items():text(k,'configuration name',256);text(v,'configuration value',8192)
    for k,v in a['configuration'].items():same(dep['environment'].get(k),v,'allocator configuration not in exact environment')
    same(a['evidence_sha256'],sha(raw['evidence']),'allocator factual evidence differs')

    access=documents['access']
    obj(access,('schema','trust_mode','dependency_policy_sha256','native_libraries','device_nodes','metadata_files','native_interfaces','owned_side_effects','network_ipc','evidence_sha256','python_audit_is_native_sandbox'),'access')
    version(access['schema'],'qwen-tpu-native-access-v1');same(access['trust_mode'],'trusted_pinned_native_backend','explicit native trust required')
    same(access['dependency_policy_sha256'],sha(raw['dependency_policy']),'native/dependency differs')
    if access['python_audit_is_native_sandbox']is not False:raise ValueError('Python audit is not native containment')
    if type(access['native_libraries'])is not list or not access['native_libraries']:raise ValueError('actual native library inventory required')
    for row in access['native_libraries']:
        obj(row,('path','sha256'),'native library');path(row['path']);digest(row['sha256'])
        observed=installed.get(row['path']);obj(observed,('sha256','size'),'installed native library');integer(observed['size'],1,'native library extent')
        same(observed['sha256'],row['sha256'],'native library absent from exact dependency pins')
    if len({r['path']for r in access['native_libraries']})!=len(access['native_libraries']):raise ValueError('duplicate native library')
    if type(access['device_nodes'])is not list or not access['device_nodes']:raise ValueError('actual device-node inventory required')
    for row in access['device_nodes']:
        obj(row,('path','major','minor','open_flags'),'device node');path(row['path'])
        for n in('major','minor','open_flags'):integer(row[n],0,n)
    if len({r['path']for r in access['device_nodes']})!=len(access['device_nodes']):raise ValueError('duplicate device node')
    for n in('metadata_files','native_interfaces','network_ipc'):
        values=access[n]
        if type(values)is not list or any(type(v)is not str or not v or len(v)>4096 for v in values):raise ValueError('bounded explicit native inventory required')
        if len(values)!=len(set(values)):raise ValueError('duplicate native inventory')
    for value in access['metadata_files']:path(value)
    if type(access['owned_side_effects'])is not list:raise ValueError('explicit owned side-effect inventory required')
    for row in access['owned_side_effects']:
        obj(row,('path','kind','mode','retirement'),'owned native side effect');path(row['path']);integer(row['mode'],0,'owned mode')
        if row['kind']not in('file','directory','lock')or row['retirement']not in('retain_evidence','original_owner_only')or row['mode']>0o777:raise ValueError('explicit owned native side-effect semantics required')
    if len({r['path']for r in access['owned_side_effects']})!=len(access['owned_side_effects']):raise ValueError('duplicate native side effect')
    same(access['evidence_sha256'],sha(raw['evidence']),'access factual evidence differs')

    w=documents['workspace']
    obj(w,('schema','source_manifest_sha256','source_archive_sha256','request_contract_sha256','accounting','stages','evidence_sha256'),'workspace')
    version(w['schema'],'qwen-tpu-workspace-policy-v1')
    for n in('source_manifest_sha256','source_archive_sha256','request_contract_sha256'):same(w[n],pp[n],'workspace '+n+' differs')
    same(w['accounting'],'additional_after_original_stage_baseline','stage additional-memory accounting required')
    obj(w['stages'],STAGES,'workspace stages')
    for stage,row in w['stages'].items():
        obj(row,('additional_live_bytes','workspace_bytes','reserve_bytes'),'stage bounds')
        integer(row['additional_live_bytes'],0,'additional live');integer(row['workspace_bytes'],0,'workspace');integer(row['reserve_bytes'],1,'reserve')
        if row['additional_live_bytes']>d['working_memory_bytes']or row['workspace_bytes']>d['workspace_bytes']or row['reserve_bytes']>d['reserve_bytes']:
            raise ValueError('stage exceeds independently reviewed resource envelope')
    same(w['evidence_sha256'],sha(raw['evidence']),'workspace factual evidence differs')

    evidence=documents['evidence']
    obj(evidence,('schema','facts','limitations'),'independent factual evidence');version(evidence['schema'],'qwen-tpu-device-factual-evidence-v1')
    obj(evidence['facts'],COVERAGE,'complete factual evidence')
    for v in evidence['facts'].values():text(v,'factual rationale',32768)
    text(evidence['limitations'],'limitations',32768)
    reviewed=('resource_policy','resource_review','dependency_policy','selection','allocator','access','workspace','evidence')
    wanted=dict(schema='qwen-tpu-device-review-v1',approved=True,
        source_manifest_sha256=pins['source_manifest_sha256'],source_archive_sha256=pins['source_archive_sha256'],
        documents={n:sha(raw[n])for n in reviewed},coverage={n:True for n in COVERAGE},
        hooks=list(HOOKS),native_trust_mode='trusted_pinned_native_backend',python_audit_is_native_sandbox=False,
        observations_are_reservations=False,policy_review_is_live_capacity=False)
    same(documents['review'],wanted,'exact independent factual review required')

    lease=documents['lease']
    obj(lease,('schema','lease_id','host_identity','physical_identity','phase_grant_sha256','selection_sha256','review_sha256','exclusive','lifetime','initialization_authorized','other_job_interference_authorized'),'lease')
    version(lease['schema'],'qwen-tpu-exclusive-lease-v1');text(lease['lease_id'],'independently owned lease',256)
    for n in('host_identity','physical_identity'):same(lease[n],s[n],'lease physical allocation differs')
    same(lease['phase_grant_sha256'],pins['phase_grant_sha256'],'lease phase differs');same(lease['selection_sha256'],sha(raw['selection']),'lease selection differs')
    same(lease['review_sha256'],sha(raw['review']),'lease independent review differs')
    if lease['exclusive']is not True or lease['initialization_authorized']is not True or lease['other_job_interference_authorized']is not False:raise ValueError('exclusive independently authorized initialization required')
    same(lease['lifetime'],'until_owned_process_retired','lease retirement ownership required')
    return DevicePolicyAdmission(phase,blobs,expected_pins)
