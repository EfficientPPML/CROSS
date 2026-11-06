"""Externally pinned server resource policy and host observations, not a grant.

No probing occurs at import. Existing key/model stores are never opened or
copied here. A host result is incomplete until a separately source-selected
device/workspace checker is implemented and admitted; policy bytes alone are
not live device evidence. This module supplies no resource-policy defaults.
"""
from pathlib import Path,PurePosixPath
import os,resource,stat
from client_role_common import raw_object,obj,integer
from qwen_file_contract import canonical,sha,same,hash_value
from qwen_finalizer import Finalizer

_QUARANTINE=[]
STRUCTURAL_FDS=781
PUBLIC_KEY_MAPPED_BYTES=159807176704
MATRIX_MAPPED_BYTES=4767875072
MAPPED_EXTENT_CAP=159808548864+4771102720+4194304
RESPONSE_PUBLICATION_CAP=242614272
COVERAGE=('all335_public_mapping_lifetimes','197_matrix_mappings_and113_gamma_vectors',
    'mapped_residency_distinct_from_anonymous_memory','model_immutable_copies_and_finite_temporaries',
    'ordinary_rotation_cache_eviction_not_all166_coexist','evaluation_transfer_and_NTT_basis_tables',
    'retained_fresh_reduced_and38_response_ciphertexts','refresh_compiler_allocator_and_workspace_peaks',
    'single_physical_device_per_ciphertext','mapping_descriptor_tracking_at_most_one_extra_fd',
    '781_structural_fds_plus_live_baseline_and_transients','same_filesystem_summed_new_publication',
    'tmpfs_new_writes_charged_to_host_and_current_cgroup','existing_stores_are_not_fresh_disk_copies',
    'failure_quarantine_is_not_reusable_capacity','concurrent_work_and_host_device_reserves',
    'bounded_scientific_proof_metadata_retention_and_reauthentication','public_metadata_memory_before_proof_read')


# Source/parser-retention lower bound only; independent review must justify the
# larger actual parsing/reauthentication working allowance. No default grant.
PUBLIC_METADATA_SOURCE_FLOOR=209453056+(64<<20)+(16<<20)
POLICY_FIELDS=('schema','cores','public_metadata_headroom_bytes','import_headroom_bytes','host_working_headroom_bytes','host_reserve_bytes',
        'mapped_residency','transient_fds','minimum_soft_nofile','storage_required_bytes','filesystem_reserve_bytes',
        'cgroup_memory_current','cgroup_memory_max','device','logical_facts','other_new_writes_allowed','compiler_disk_cache_allowed')

def _phase(phase):
    from qwen_server_admission import ServerPhaseAdmission
    if type(phase)is not ServerPhaseAdmission:raise TypeError('exact complete server phase required before resource observations')
    phase.check()
    return phase


def _path(value):
    if type(value)is not str or not 1<len(value)<=4096 or value.startswith('//')or '\\'in value or '\x00'in value:
        raise ValueError('exact absolute resource metadata path required')
    path=PurePosixPath(value)
    if not path.is_absolute()or str(path)!=value or '..'in path.parts or path==path.parent:
        raise ValueError('canonical non-root resource path required')
    return Path(value)


def logical_facts(phase):
    _phase(phase);request=raw_object(phase.request_bytes,1<<20,'server request')
    tiles=request['input']['input_tile_count'];integer(tiles,1,'request tile count')
    if tiles not in(1,2):raise ValueError('one or two original request tiles required')
    return dict(public_key_arrays=335,public_key_mapped_payload_bytes=PUBLIC_KEY_MAPPED_BYTES,
        logical_matrices=198,physical_matrix_files=197,matrix_mapped_payload_bytes=MATRIX_MAPPED_BYTES,
        gamma_vectors=113,gamma_container_cap_bytes=4194304,mapped_file_extent_cap_bytes=MAPPED_EXTENT_CAP,
        request_tiles=tiles,request_fresh_payload_bytes=tiles*104857600,
        request_reduced_payload_bytes=tiles*54525952,response_blocks=38,response_payload_bytes=239075328,
        engine_retained_ciphertext_payload_bytes=tiles*(104857600+54525952)+239075328,
        one_evaluation_or_rotation_pair_bytes=956301312,psi_inverse_psi_bytes=239075328,
        coefficient_map_bytes=87031808,largest_matrix_payload_bytes=1244659712,
        largest_matrix_finite_boolean_bytes=155582464,response_publication_cap_bytes=RESPONSE_PUBLICATION_CAP,
        structural_fd_reservation=STRUCTURAL_FDS,all166_formatted_pairs_assumed_concurrent=False,
        logical_sizes_are_peak_memory_bounds=False,borrowed_store_fresh_copy_bytes=0)


def authenticate_policy(phase,policy_bytes,review_bytes,*,expected_pins):
    """Pure authentication; expected_pins must come from independent runtime admission."""
    _phase(phase)
    pins=raw_object(expected_pins,1<<14,'independent server resource pins')
    obj(pins,('source_manifest_sha256','source_archive_sha256','phase_grant_sha256',
        'dependency_policy_sha256','resource_policy_sha256','resource_review_sha256'),'server resource pins')
    for value in pins.values():hash_value(value)
    if sha(phase.grant_bytes)!=pins['phase_grant_sha256']or sha(policy_bytes)!=pins['resource_policy_sha256']or sha(review_bytes)!=pins['resource_review_sha256']:
        raise ValueError('independently pinned server resource bytes differ')
    phase_pins=phase.expected_pins.check()
    phase_grant=raw_object(phase.grant_bytes,1<<16,'server phase grant')
    same(pins['phase_grant_sha256'],phase_pins['phase_grant_sha256'],'resource/phase grant pin differs')
    for name in('source_manifest_sha256','source_archive_sha256'):
        same(pins[name],phase_pins[name],'resource/phase source pin differs')
        same(pins[name],phase_grant[name],'resource/granted source pin differs')
    for name in('dependency_policy_sha256','resource_policy_sha256','resource_review_sha256'):
        same(pins[name],phase_pins['policies'][name],'resource/phase policy pin differs')
        same(pins[name],phase_grant['policies'][name],'resource/granted policy pin differs')
    policy=raw_object(policy_bytes,1<<16,'server resource policy')
    fields=POLICY_FIELDS
    obj(policy,fields,'server resource policy')
    if policy['schema']!='qwen-server-resource-policy-v1':raise ValueError('exact server resource policy schema required')
    cores=policy['cores']
    if type(cores)is not list or not 0<len(cores)<=256 or any(type(c)is not int or c<0 for c in cores)or cores!=sorted(set(cores)):
        raise ValueError('independently selected ordered CPU affinity required')
    for name in('public_metadata_headroom_bytes','import_headroom_bytes','host_working_headroom_bytes','host_reserve_bytes','transient_fds','minimum_soft_nofile','filesystem_reserve_bytes'):
        integer(policy[name],1,name)
    if policy['public_metadata_headroom_bytes']<PUBLIC_METADATA_SOURCE_FLOOR:
        raise ValueError('public metadata allowance below raw source/proof retention floor')
    if policy['minimum_soft_nofile']<STRUCTURAL_FDS+policy['transient_fds']:
        raise ValueError('declared descriptor minimum below structural reservation')
    for name,leaf in(('cgroup_memory_current','memory.current'),('cgroup_memory_max','memory.max')):
        path=_path(policy[name])
        if Path('/sys/fs/cgroup')not in path.parents or path.name!=leaf:raise ValueError('exact cgroup-v2 memory files required')
    if Path(policy['cgroup_memory_current']).parent!=Path(policy['cgroup_memory_max']).parent:raise ValueError('one cgroup memory scope required')
    residency=policy['mapped_residency']
    obj(residency,('mode','allowance_bytes','reviewed_evidence_sha256'),'mapped residency')
    integer(residency['allowance_bytes'],0,'mapped residency allowance');hash_value(residency['reviewed_evidence_sha256'])
    if residency['mode']not in('full_extent_allowance','independently_reviewed_residency_allowance'):
        raise ValueError('explicit mapped residency assumption required')
    if residency['mode']=='full_extent_allowance'and residency['allowance_bytes']<MAPPED_EXTENT_CAP:
        raise ValueError('full mapped extent allowance is incomplete')
    # A smaller allowance is not inferred from mmap or existing tmpfs backing.
    # Its independent evidence hash and reasoning must be covered by the review.
    obj(policy['storage_required_bytes'],('response','evidence'),'new publication storage')
    for value in policy['storage_required_bytes'].values():integer(value,1,'new publication capacity')
    if policy['storage_required_bytes']['response']<RESPONSE_PUBLICATION_CAP:
        raise ValueError('response storage excludes complete38 files/container overhead')
    if policy['other_new_writes_allowed']is not False or policy['compiler_disk_cache_allowed']is not False:
        raise ValueError('unbudgeted file or external compiler-cache writes are forbidden')
    device=policy['device']
    obj(device,('platform','selection_sha256','allocator_policy_sha256','working_memory_bytes','workspace_bytes',
        'reserve_bytes','physical_devices_per_ciphertext','admission_mode'),'separate device policy')
    if type(device['platform'])is not str or not 0<len(device['platform'])<=64:raise ValueError('explicit device platform required')
    for name in('selection_sha256','allocator_policy_sha256'):hash_value(device[name])
    for name in('working_memory_bytes','workspace_bytes','reserve_bytes'):integer(device[name],1,name)
    if type(device['physical_devices_per_ciphertext'])is not int or device['physical_devices_per_ciphertext']!=1:
        raise ValueError('ordinary owner requires one physical device per ciphertext')
    if device['admission_mode']!='required_external_source_fixed_checker':raise ValueError('live source-fixed device admission is still required')
    same(policy['logical_facts'],logical_facts(phase),'source-derived server facts differ')
    review=raw_object(review_bytes,1<<16,'independent server resource review')
    expected=dict(schema='qwen-server-resource-review-v1',approved=True,
        source_manifest_sha256=pins['source_manifest_sha256'],source_archive_sha256=pins['source_archive_sha256'],
        dependency_policy_sha256=pins['dependency_policy_sha256'],resource_policy_sha256=sha(policy_bytes),
        mapped_residency_evidence_sha256=residency['reviewed_evidence_sha256'],coverage={name:True for name in COVERAGE},
        logical_counts_are_peak_bounds=False,live_capacity_measured=False,actual_execution_authorized=False)
    same(review,expected,'complete independent server resource reasoning review required')
    return policy


def _close(fd,label,primary=None):
    final=Finalizer({},primary)
    def retire():
        try:os.close(fd)
        except BaseException as error:_QUARANTINE.append((fd,label,error));raise
    final.attempt(label,retire);final.raise_if_failed()


def _kernel(path,limit=1<<20):
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK);primary=None
    try:
        blocks=[];size=0
        while size<=limit:
            block=os.read(fd,min(65536,limit+1-size))
            if not block:break
            blocks.append(block);size+=len(block)
        if not 0<size<=limit:raise ValueError('bounded kernel metadata required')
        return b''.join(blocks).decode('ascii')
    except BaseException as error:primary=error;raise
    finally:_close(fd,'kernel memory descriptor retirement',primary)


def check_host_capacity(phase,policy_bytes,review_bytes,*,expected_pins,after_import):
    """Observe own host and output-parent metadata after grant; never reserve capacity."""
    if type(after_import)is not bool:raise TypeError('exact capacity phase required')
    policy=authenticate_policy(phase,policy_bytes,review_bytes,expected_pins=expected_pins)
    if _QUARANTINE:raise RuntimeError('uncertain capacity descriptor retirement requires process exit')
    if sorted(os.sched_getaffinity(0))!=policy['cores']:raise ValueError('actual CPU lane differs from reviewed server policy')
    def kibibytes(path,label):
        rows=[x.split()for x in _kernel(path).splitlines()if x.startswith(label+':')]
        if len(rows)!=1 or len(rows[0])!=3 or rows[0][2]!='kB'or not rows[0][1].isdecimal():raise ValueError('exact kernel memory observation required')
        return int(rows[0][1])*1024
    available=kibibytes('/proc/meminfo','MemAvailable');rss=kibibytes('/proc/self/status','VmRSS')
    groups=[x.split(':',2)for x in _kernel('/proc/self/cgroup').splitlines()]
    unified=[x[2]for x in groups if len(x)==3 and x[:2]==['0','']]
    if len(unified)!=1:raise ValueError('one actual cgroup-v2 scope required')
    current_path=Path('/sys/fs/cgroup')/unified[0].lstrip('/')/'memory.current'
    if str(current_path)!=policy['cgroup_memory_current']:raise ValueError('reviewed memory group is not this process')
    current_text=_kernel(policy['cgroup_memory_current']).strip();maximum_text=_kernel(policy['cgroup_memory_max']).strip()
    if not current_text.isdecimal()or not(maximum_text=='max'or maximum_text.isdecimal()):raise ValueError('exact cgroup memory observations required')
    current=int(current_text);maximum=None if maximum_text=='max'else int(maximum_text)
    if maximum is not None and maximum<current:raise MemoryError('current cgroup limit exceeded')
    mounts=[]
    for line in _kernel('/proc/self/mountinfo').splitlines():
        left,separator,right=line.partition(' - ');fields=left.split();post=right.split()
        if not separator or len(fields)<6 or len(post)<3:raise ValueError('complete kernel mount metadata required')
        decoded=fields[4].replace('\\040',' ').replace('\\011','\t').replace('\\012','\n').replace('\\134','\\')
        mounts.append((Path(decoded),post[0]))
    soft,_=resource.getrlimit(resource.RLIMIT_NOFILE);baseline=len(os.listdir('/proc/self/fd'))
    needed=baseline+STRUCTURAL_FDS+policy['transient_fds']
    if soft!=resource.RLIM_INFINITY and(soft<policy['minimum_soft_nofile']or soft<needed):
        raise OSError('live descriptor baseline plus781 and transients exceeds soft limit')
    outputs=raw_object(phase.grant_bytes,1<<16,'server phase grant')['outputs']
    obj(outputs,('response','evidence'),'server publication outputs')
    devices={};observed={};tmpfs_charge=0;original_uid=os.geteuid()
    for role,description in outputs.items():
        expected=description['parent'];obj(expected,('path','device','inode','mode'),'original output parent')
        path=_path(expected['path']);integer(expected['device'],0,'parent device');integer(expected['inode'],1,'parent inode');integer(expected['mode'],0,'parent mode')
        if path.resolve()!=path:raise ValueError('output parent symlink alias differs')
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC);primary=None
        try:
            held=os.fstat(fd)
            def check_parent():
                if os.geteuid()!=original_uid:raise ValueError('original resource observer effective UID changed')
                for info in(os.fstat(fd),path.lstat()):
                    mode=stat.S_IMODE(info.st_mode)
                    if not stat.S_ISDIR(info.st_mode)or info.st_uid not in(0,original_uid)or info.st_uid!=held.st_uid or(mode&0o022 and not mode&stat.S_ISVTX):
                        raise ValueError('original trusted granted output parent required')
                    same(dict(path=str(path),device=info.st_dev,inode=info.st_ino,mode=mode),expected,'original granted output parent differs')
            check_parent();fs=os.fstatvfs(fd);free=fs.f_bavail*fs.f_frsize
            entry=devices.setdefault(held.st_dev,dict(required_bytes=policy['filesystem_reserve_bytes'],free_bytes=free))
            entry['required_bytes']+=policy['storage_required_bytes'][role];entry['free_bytes']=min(entry['free_bytes'],free)
            matches=[(p,t)for p,t in mounts if p==path or p in path.parents]
            if not matches:raise ValueError('output mount type unavailable')
            mount,kind=max(matches,key=lambda item:len(item[0].parts))
            if kind in('tmpfs','ramfs'):tmpfs_charge+=policy['storage_required_bytes'][role]
            observed[role]=dict(device=held.st_dev,parent_inode=held.st_ino,available_bytes=free,
                mount=str(mount),filesystem=kind,new_write_budget_bytes=policy['storage_required_bytes'][role])
            check_parent()
        except BaseException as error:primary=error;raise
        finally:_close(fd,'capacity parent retirement',primary)
    if any(row['free_bytes']<row['required_bytes']for row in devices.values()):raise OSError('combined same-device publication capacity unavailable')
    working=policy['mapped_residency']['allowance_bytes']+policy['host_working_headroom_bytes']
    headroom=working if after_import else policy['import_headroom_bytes']
    required=headroom+policy['host_reserve_bytes']+tmpfs_charge
    if available<required or(maximum is not None and maximum-current<required):raise MemoryError('server host/cgroup headroom unavailable')
    _phase(phase)
    return dict(after_import=after_import,observed_rss_bytes=rss,mem_available_bytes=available,
        cgroup_current_bytes=current,cgroup_max_bytes=maximum,required_host_headroom_bytes=required,
        mapped_residency_allowance_bytes=policy['mapped_residency']['allowance_bytes'],
        tmpfs_new_write_charge_bytes=tmpfs_charge,fd_baseline=baseline,structural_fd_reservation=STRUCTURAL_FDS,
        transient_fds=policy['transient_fds'],required_fd_capacity=needed,soft_nofile=None if soft==resource.RLIM_INFINITY else soft,
        outputs=observed,filesystems=devices,existing_stores_copied=False,borrowed_body_reads=False,
        capacity_reserved=False,peak_memory_measured=False,device_capacity_checked=False,resource_admission_complete=False)


def require_complete_capacity(*args,**kwargs):
    raise PermissionError('live source-fixed server device/workspace capacity admission is not implemented; host policy or snapshot alone cannot authorize evaluation')
