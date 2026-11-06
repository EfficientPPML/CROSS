"""Synthetic metadata only; no policies describe the current host or device.

The device and resource authenticators remain real. Only complete phase checking
and its expected-pin method have explicit recording seams in the child.
"""
import hashlib
import json
from pathlib import Path

ROLES=('resource_policy','resource_review','dependency_policy','selection','allocator','access','workspace','evidence','review','lease')
STAGES=('post_init','post_keys','pre_evaluation','terminal')
HOOKS=('device_capacity_check_bindings','device_capacity_check_host','device_capacity_check_lease','device_capacity_initialize','device_capacity_synchronize')
DEVICE_COVERAGE=('exclusive_physical_memory','logical_physical_device_mapping','trusted_native_access_inventory',
    'exact_plugin_and_allocator_semantics','initialization_host_dma_fd_costs','all335_public_array_lifetimes',
    'lazy_rotation_and_ntt_caches','refresh_and_compiler_workspace','async_and_executable_lifetimes',
    'retained_inputs_intermediates_and38_outputs','stage_additional_memory_not_double_counted',
    'fragmentation_and_untracked_reserve','one_device_per_ciphertext','host_and_device_admission_separate')
RESOURCE_COVERAGE=('all335_public_mapping_lifetimes','197_matrix_mappings_and113_gamma_vectors',
    'mapped_residency_distinct_from_anonymous_memory','model_immutable_copies_and_finite_temporaries',
    'ordinary_rotation_cache_eviction_not_all166_coexist','evaluation_transfer_and_NTT_basis_tables',
    'retained_fresh_reduced_and38_response_ciphertexts','refresh_compiler_allocator_and_workspace_peaks',
    'single_physical_device_per_ciphertext','mapping_descriptor_tracking_at_most_one_extra_fd',
    '781_structural_fds_plus_live_baseline_and_transients','same_filesystem_summed_new_publication',
    'tmpfs_new_writes_charged_to_host_and_current_cgroup','existing_stores_are_not_fresh_disk_copies',
    'failure_quarantine_is_not_reusable_capacity','concurrent_work_and_host_device_reserves',
    'bounded_scientific_proof_metadata_retention_and_reauthentication','public_metadata_memory_before_proof_read')

def canonical(value):return(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def label(name):return sha(('synthetic metadata only: '+name).encode())

def logical_fixture():
    # Independent fixed public geometry fixture for one tile. These are mapped
    # extents, not resident-memory or capacity assertions. No data is allocated.
    return dict(public_key_arrays=335,public_key_mapped_payload_bytes=159807176704,
        logical_matrices=198,physical_matrix_files=197,matrix_mapped_payload_bytes=4767875072,
        gamma_vectors=113,gamma_container_cap_bytes=4194304,mapped_file_extent_cap_bytes=164583845888,
        request_tiles=1,request_fresh_payload_bytes=104857600,request_reduced_payload_bytes=54525952,
        response_blocks=38,response_payload_bytes=239075328,engine_retained_ciphertext_payload_bytes=398458880,
        one_evaluation_or_rotation_pair_bytes=956301312,psi_inverse_psi_bytes=239075328,
        coefficient_map_bytes=87031808,largest_matrix_payload_bytes=1244659712,largest_matrix_finite_boolean_bytes=155582464,
        response_publication_cap_bytes=242614272,structural_fd_reservation=781,
        all166_formatted_pairs_assumed_concurrent=False,logical_sizes_are_peak_memory_bounds=False,borrowed_store_fresh_copy_bytes=0)

def build(root,runtime_sources,phase_module,scenario):
    root=Path(root)
    source_digest=sha(canonical(runtime_sources));archive_digest=label('not a real server source archive')
    request=canonical({'input':{'input_tile_count':1}});request_digest=sha(request)
    fake_path=str(root/'recording_jax.py');fake_bytes=(root/'recording_jax.py').read_bytes()
    fake_hash=sha(fake_bytes);native='/synthetic-device-fixture/libfixture.so';native_hash=label('no native library exists')
    dep=dict(schema='qwen-server-tpu-dependencies-v1',interpreter='/synthetic-device-fixture/python',
        resolved_interpreter='/synthetic-device-fixture/python',python_major_minor=[3,13],cores=[0],
        environment={'SYNTHETIC_DEVICE_FIXTURE':'1'},installed_sources={fake_path:dict(sha256=fake_hash,size=len(fake_bytes)),native:dict(sha256=native_hash,size=1)},
        library_roots=['/synthetic-device-fixture'],backend_module_origins={'jax':dict(file=fake_path,spec=fake_path,search_locations=[])})
    if scenario=='document_origin_join':dep['backend_module_origins']['jax']['spec']='/synthetic-device-fixture/wrong_jax.py'
    dep_raw=canonical(dep)
    evidence=dict(schema='qwen-tpu-device-factual-evidence-v1',
        facts={name:'SYNTHETIC TEST MARKER; no factual host/device claim: '+name for name in DEVICE_COVERAGE},
        limitations='Entirely fabricated metadata; no genuine scientific, resource, device, allocator or lease admission.')
    evidence_raw=canonical(evidence)
    if scenario=='document_noncanonical':evidence_raw=(json.dumps(evidence,sort_keys=True,indent=2)+'\n').encode()
    workspace=dict(schema='qwen-tpu-workspace-policy-v1',source_manifest_sha256=source_digest,source_archive_sha256=archive_digest,
        request_contract_sha256=request_digest,accounting='additional_after_original_stage_baseline',
        stages={s:dict(additional_live_bytes=4,workspace_bytes=3,reserve_bytes=3)for s in STAGES},evidence_sha256=sha(evidence_raw))
    if scenario=='document_workspace_bound':workspace['stages']['post_init']['additional_live_bytes']=5
    workspace_raw=canonical(workspace)
    access=dict(schema='qwen-tpu-native-access-v1',trust_mode='trusted_pinned_native_backend',dependency_policy_sha256=sha(dep_raw),
        native_libraries=[dict(path=native,sha256=native_hash)],device_nodes=[dict(path='/dev/synthetic-device-fixture',major=0,minor=0,open_flags=0)],
        metadata_files=[],native_interfaces=['synthetic recording only'],owned_side_effects=[],network_ipc=[],
        evidence_sha256=sha(evidence_raw),python_audit_is_native_sandbox=False)
    access_raw=canonical(access)
    selection=dict(schema='qwen-tpu-device-selection-v1',host_identity='synthetic-host',physical_identity='synthetic-memory',
        device=dict(platform='tpu',id=0,process_index=0,local_hardware_id=0,device_kind='synthetic-recording-device'),
        process_count=1,physical_devices_per_ciphertext=1,environment=dep['environment'],access_sha256=sha(access_raw),workspace_sha256=sha(workspace_raw))
    selection_raw=canonical(selection)
    allocator=dict(schema='qwen-tpu-allocator-policy-v1',dependency_policy_sha256=sha(dep_raw),jax_module=dict(path=fake_path,sha256=fake_hash),
        required_stats=['bytes_in_use','bytes_limit'],semantics='selected_allocator_with_exclusive_physical_memory',configuration={},evidence_sha256=sha(evidence_raw))
    allocator_raw=canonical(allocator)
    resource=dict(schema='qwen-server-resource-policy-v1',cores=[0],public_metadata_headroom_bytes=293339136,
        import_headroom_bytes=1,host_working_headroom_bytes=1,host_reserve_bytes=1,
        mapped_residency=dict(mode='full_extent_allowance',allowance_bytes=164583845888,reviewed_evidence_sha256=label('synthetic residency rationale')),
        transient_fds=1,minimum_soft_nofile=782,storage_required_bytes=dict(response=242614272,evidence=1),filesystem_reserve_bytes=1,
        cgroup_memory_current='/sys/fs/cgroup/synthetic-device-fixture/memory.current',cgroup_memory_max='/sys/fs/cgroup/synthetic-device-fixture/memory.max',
        device=dict(platform='tpu',selection_sha256=sha(selection_raw),allocator_policy_sha256=sha(allocator_raw),
            working_memory_bytes=4,workspace_bytes=3,reserve_bytes=3,physical_devices_per_ciphertext=1,admission_mode='required_external_source_fixed_checker'),
        logical_facts=logical_fixture(),other_new_writes_allowed=False,compiler_disk_cache_allowed=False)
    resource_raw=canonical(resource)
    resource_review=dict(schema='qwen-server-resource-review-v1',approved=True,source_manifest_sha256=source_digest,
        source_archive_sha256=archive_digest,dependency_policy_sha256=sha(dep_raw),resource_policy_sha256=sha(resource_raw),
        mapped_residency_evidence_sha256=resource['mapped_residency']['reviewed_evidence_sha256'],coverage={n:True for n in RESOURCE_COVERAGE},
        logical_counts_are_peak_bounds=False,live_capacity_measured=False,actual_execution_authorized=False)
    resource_review_raw=canonical(resource_review)
    policy_pins=dict(dependency_policy_sha256=sha(dep_raw),resource_policy_sha256=sha(resource_raw),resource_review_sha256=sha(resource_review_raw))
    grant_raw=canonical(dict(source_root=str(root),source_manifest_sha256=source_digest,source_archive_sha256=archive_digest,policies=policy_pins))
    pp=dict(source_map=dict(runtime_sources),source_manifest_sha256=source_digest,source_archive_sha256=archive_digest,
        phase_grant_sha256=sha(grant_raw),request_contract_sha256=request_digest,policies=policy_pins)
    # Exact real types, intentionally incomplete synthetic phase evidence. The
    # child's two recording check methods explicitly fence this fixture seam.
    phase=phase_module.ServerPhaseAdmission(None,None,None,request,b'synthetic upload',grant_raw,
        phase_module.ExpectedServerPhasePins(canonical(pp)),None)
    raw=dict(resource_policy=resource_raw,resource_review=resource_review_raw,dependency_policy=dep_raw,selection=selection_raw,
        allocator=allocator_raw,access=access_raw,workspace=workspace_raw,evidence=evidence_raw)
    review=dict(schema='qwen-tpu-device-review-v1',approved=True,source_manifest_sha256=source_digest,source_archive_sha256=archive_digest,
        documents={n:sha(b)for n,b in raw.items()},coverage={n:True for n in DEVICE_COVERAGE},hooks=list(HOOKS),
        native_trust_mode='trusted_pinned_native_backend',python_audit_is_native_sandbox=False,
        observations_are_reservations=False,policy_review_is_live_capacity=False)
    if scenario=='document_review_coverage':review['coverage']['exclusive_physical_memory']=False
    raw['review']=canonical(review)
    lease=dict(schema='qwen-tpu-exclusive-lease-v1',lease_id='SYNTHETIC-NOT-A-LEASE',host_identity='synthetic-host',physical_identity='synthetic-memory',
        phase_grant_sha256=sha(grant_raw),selection_sha256=sha(selection_raw),review_sha256=sha(raw['review']),
        exclusive=True,lifetime='until_owned_process_retired',initialization_authorized=True,other_job_interference_authorized=False)
    if scenario=='document_lease_ownership':lease['exclusive']=False
    raw['lease']=canonical(lease)
    pins=dict(schema='qwen-server-device-pins-v1',source_manifest_sha256=source_digest,source_archive_sha256=archive_digest,
        phase_grant_sha256=sha(grant_raw),documents={n:sha(raw[n])for n in ROLES})
    if scenario=='document_source_phase_join':pins['source_archive_sha256']=label('intentionally wrong independent source join')
    blobs=tuple((n,raw[n])for n in ROLES)
    if scenario=='document_role_order':blobs=(blobs[1],blobs[0],*blobs[2:])
    return dict(phase=phase,phase_pins=pp,blobs=blobs,pins=canonical(pins),documents=raw,resource=resource,
        synthetic_only=True,resource_validator_is_real=True,phase_validator_is_recording=True)
