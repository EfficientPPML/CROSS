"""Synthetic externally trusted fixture arguments, never actual evidence.

Only fixed public aggregate metadata is reused. Every process/source/grant,
inode, manifest body descriptor hash and readiness claim below is fabricated
for isolated pure semantic tests. Nothing is written as a real receipt.
"""
import json
from types import SimpleNamespace
from qwen_file_contract import canonical, sha
from qwen_key_contract import PublicProfile, public_body, KEYGEN, bindings
from model_admission import (Aggregate, ExpectedModelPins, ModelAdmission,
                             MODEL_ID, AGGREGATE_SHA256, CATALOG_SHA256)
from keygen_boundary_pins import CORE_MANIFEST_SHA256, REQUIRED_ACTUAL_SOURCES
from keygen_authority import (ActualKeygenIntent, ExpectedLaunchPins, ExpectedReadyPins,
                              authenticate_launch, authenticate_readiness)


def fake_hash(label):
    return sha(('SYNTHETIC-NOT-ACTUAL: '+label).encode())


def clone(value):
    return json.loads(canonical(value))


def synthetic_model(public, aggregate_raw, consumer, change_report=None):
    aggregate = Aggregate(public, aggregate_raw)
    catalog = aggregate.catalog
    roots = {family:dict(path='/synthetic-unopened-model/'+family, device=7,
                        inode=100+i, mode=0o700)
             for i,family in enumerate(('qkv','decoder','gamma'))}
    physical = {}
    for row in catalog['matrices'].values():
        descriptor = row['descriptor']
        physical[(row['family'],descriptor['file'])] = descriptor
    files = []
    for i,((family,name),descriptor) in enumerate(sorted(physical.items())):
        size = 8
        for dimension in descriptor['shape']:
            size *= dimension
        files.append(dict(family=family,name=name,
            state=dict(device=7,inode=1000+i,size=size+128,mtime_ns=1,ctime_ns=2,mode=0o644),
            file_sha256=descriptor['file_sha256'],raw_sha256=descriptor['sha256']))
    files.append(dict(family='gamma',name='synthetic-gamma.npz',
        state=dict(device=7,inode=2000,size=1024,mtime_ns=1,ctime_ns=2,mode=0o644),
        file_sha256=catalog['gamma_container_sha256'],raw_sha256=None))
    stores = dict(roots=roots,files=sorted(files,key=lambda r:(r['family'],r['name'])))
    source = {'synthetic_model_validator.py':fake_hash('model validator source')}
    source_digest = sha(canonical(source))
    archive = fake_hash('model archive');grant = fake_hash('model grant')
    report = dict(schema='qwen-model-store-validation-v1',mode='reuse_existing_exports',
        passed=True,complete=True,error_type=None,model_id=MODEL_ID,
        aggregate_manifest_sha256=AGGREGATE_SHA256,catalog_sha256=CATALOG_SHA256,
        execution_grant_sha256=grant,source_manifest_sha256=source_digest,
        source_archive_sha256=archive,source_hashes_before=clone(source),
        source_hashes_after=clone(source),process=dict(pid=101,pgid=100),stores=stores,
        counts=dict(logical_matrices=198,physical_matrices=197,gamma_vectors=113,
                    physical_matrix_raw_bytes=4767875072,gamma_raw_bytes=524288),
        checks={n:True for n in ('public_provenance_authenticated','complete_catalog_rederived',
            'all_matrix_headers_extents_hashes_finite','all_gamma_members_headers_hashes_finite',
            'original_held_identity_unchanged','readback_completed','source_postcheck_completed',
            'cleanup_completed','signals_restored','borrowed_stores_retained')},
        scope=dict(parameter_body_reads_performed=True,new_export_performed=False,
            model_forward=False,backend_context_constructed=False,keys_read=False,
            ciphertexts_read=False,token_inputs_read=False,coefficient_selection=False,
            scientific_calibration_pass_claimed=False),audit_denials=[])
    if change_report: change_report(report)
    report_raw = canonical(report)
    outer = dict(schema='qwen-model-store-validation-outer-v1',report_sha256=sha(report_raw),
        aggregate_manifest_sha256=AGGREGATE_SHA256,model_id=MODEL_ID,
        source_manifest_sha256=source_digest,source_archive_sha256=archive,
        execution_grant_sha256=grant,pid=101,pgid=100,exit_code=0,
        normal_return=True,reaped=True,process_group_absent=True)
    outer_raw = canonical(outer)
    review = dict(schema='qwen-model-stage-root-admission-v1',passed=True,model_id=MODEL_ID,
        aggregate_manifest_sha256=AGGREGATE_SHA256,catalog_sha256=CATALOG_SHA256,
        validation_report_sha256=sha(report_raw),validation_outer_sha256=sha(outer_raw),
        source_manifest_sha256=source_digest,source_archive_sha256=archive,
        execution_grant_sha256=grant,current_store_binding_sha256=sha(canonical(stores)),
        all_required_links_verified=True,source_archive_members_verified=True,
        actual_server_execution_authorized=False)
    review_raw = canonical(review)
    expected = dict(schema='qwen-model-expected-pins-v1',model_id=MODEL_ID,
        aggregate_manifest_sha256=AGGREGATE_SHA256,catalog_sha256=CATALOG_SHA256,
        producer_source_manifest_sha256=source_digest,producer_source_archive_sha256=archive,
        producer_source_map=source,execution_grant_sha256=grant,
        validation_report_sha256=sha(report_raw),validation_outer_sha256=sha(outer_raw),
        root_review_sha256=sha(review_raw),consumer_source_manifest_sha256=consumer)
    # Construct without pre-validating: the tested keygen gate must reauthenticate it.
    return ModelAdmission(public,aggregate_raw,report_raw,outer_raw,review_raw,
                          ExpectedModelPins(canonical(expected)))


def build_chain(public, aggregate_raw, *, change_source=None, change_intent=None,
                change_model_report=None, model_consumer=None, change_grant=None,
                change_launch_pins=None, change_public=None, change_report=None,
                change_outer=None, change_root=None, change_ready_pins=None):
    """Hooks re-pin modified public documents to exercise semantic rejection."""
    values = public.check()
    profile = PublicProfile.from_bytes(values['inputs/profile_audit.json'],values['inputs/actual_key_plan.json'])
    source = dict(REQUIRED_ACTUAL_SOURCES)
    source['synthetic_actual_keygen_runner.py'] = fake_hash('not an actual runner')
    if change_source: change_source(source)
    source_hash = sha(canonical(source));archive = fake_hash('keygen source archive')
    model = synthetic_model(public,aggregate_raw,model_consumer or source_hash,change_model_report)
    model_evidence = dict(manifest_sha256=sha(model.manifest_bytes),report_sha256=sha(model.report_bytes),
        outer_sha256=sha(model.outer_bytes),root_review_sha256=sha(model.review_bytes),
        expected_pins_sha256=sha(model.expected_pins.raw))
    roles = {role:dict(source_manifest_sha256=fake_hash(role+' actual map'),
                       source_archive_sha256=fake_hash(role+' actual archive'),
                       readiness_review_sha256=fake_hash(role+' readiness'),
                       execution_grant_sha256=fake_hash(role+' grant'))
             for role in ('request_producer','response_consumer')}
    policies = {name:fake_hash(name) for name in ('dependency_policy_sha256','role_policy_sha256',
                                                 'resource_policy_sha256','resource_review_sha256')}
    scope = dict(fresh_key_generation=True,new_owned_key_readback=True,
        preexisting_key_bodies_read=False,model_bodies_read=False,token_or_calibration_bodies_read=False,
        ciphertexts_read=False,ckks_context_constructed=False,model_forward=False,
        scientific_acceptance_claimed=False,coefficient_selection_or_revision=False,
        downstream_execution_authorized=False)
    intent_value = dict(schema='qwen-keygen-actual-intent-v1',nonce='synthetic-only',model_id=MODEL_ID,
        generation_source_manifest_sha256=source_hash,core_source_manifest_sha256=CORE_MANIFEST_SHA256,
        keygen_source_sha256=KEYGEN,
        request_producer_source_manifest_sha256=roles['request_producer']['source_manifest_sha256'],
        response_consumer_source_manifest_sha256=roles['response_consumer']['source_manifest_sha256'],
        bindings=bindings(),parameters=profile.parameters,
        paths=dict(public_root='/synthetic-unopened-keygen/public',
                   private_root='/synthetic-unopened-keygen/private',
                   evidence_root='/synthetic-unopened-keygen/evidence'),
        logical_storage=dict(public_arrays=335,private_arrays=1,public_files=336,private_files=2,
            public_raw_bytes=159807176704,private_raw_bytes=52428800,
            public_required_bytes=160882290688,private_required_bytes=53481472,
            retained_store_descriptors=342,store_temporary_allowance=8,
            scope='inherited logical store allowance only; actual memory/runtime/evidence headroom requires separate reviewed policy'))
    if change_intent: change_intent(intent_value)
    intent = ActualKeygenIntent(public,canonical(intent_value))
    parents = {name:dict(path='/synthetic-unopened-keygen',device=9,inode=900,mode=0o700)
               for name in ('public_root','private_root','evidence_root')}
    grant = dict(schema='qwen-keygen-actual-grant-v1',role='generate_fresh_qwen166_keys',execute=True,
        intent_sha256=sha(intent.raw),source_root='/synthetic-unopened-source',
        source_manifest_sha256=source_hash,source_archive_sha256=archive,
        model_evidence=clone(model_evidence),role_evidence=clone(roles),
        prelaunch_review_sha256=fake_hash('prelaunch review'),policies=clone(policies),
        parents=parents,scope=clone(scope))
    if change_grant: change_grant(grant)
    grant_raw = canonical(grant)
    launch_pins = dict(schema='qwen-keygen-launch-pins-v1',intent_sha256=sha(intent.raw),
        grant_sha256=sha(grant_raw),source_manifest_sha256=source_hash,source_archive_sha256=archive,
        source_map=source,model_evidence=model_evidence,role_evidence=roles,
        prelaunch_review_sha256=fake_hash('prelaunch review'),policies=policies)
    if change_launch_pins: change_launch_pins(launch_pins)
    launch_expected = ExpectedLaunchPins(canonical(launch_pins))
    def launch():
        return authenticate_launch(intent,grant_raw,model,expected_pins=launch_expected)

    arrays = {spec.name:dict(sha256=fake_hash('unopened '+spec.name),dtype='uint32',
                            shape=list(spec.shape),bytes=spec.raw_bytes+128)
              for spec in profile.arrays}
    body = public_body(profile,arrays,KEYGEN)
    if change_public: change_public(body)
    context = sha(canonical(body));public_raw = canonical(body | {'context_id':context})
    private_hash = fake_hash('unopened private manifest')
    stores = {}
    for i,role in enumerate(('public','private')):
        manifest_hash = sha(public_raw) if role=='public' else private_hash
        stores[role] = dict(path=intent_value['paths'][role+'_root'],device=9,inode=910+i,
            mode=0o700,parent=clone(parents[role+'_root']),
            manifest=dict(name=role+'_manifest.json',sha256=manifest_hash,
                state=dict(device=9,inode=920+i,size=len(public_raw) if role=='public' else 1024,
                           mtime_ns=10,ctime_ns=11,mode=0o600)))
    report = dict(schema='qwen-keygen-actual-report-v1',passed=True,complete=True,error_type=None,
        intent_sha256=sha(intent.raw),grant_sha256=sha(grant_raw),source_manifest_sha256=source_hash,
        source_archive_sha256=archive,source_hashes_before=clone(source),source_hashes_after=clone(source),
        model_id=MODEL_ID,model_evidence=clone(model_evidence),role_evidence=clone(roles),
        core_source_manifest_sha256=CORE_MANIFEST_SHA256,keygen_source_sha256=KEYGEN,bindings=bindings(),
        request_producer_source_manifest_sha256=intent_value['request_producer_source_manifest_sha256'],
        response_consumer_source_manifest_sha256=intent_value['response_consumer_source_manifest_sha256'],
        context_id=context,public_manifest_sha256=sha(public_raw),private_manifest_sha256=private_hash,
        stores=stores,counts=dict(pke_pairs=1,evaluation_key_pairs=1,rotation_key_pairs=166,
            primitive_calls=168,public_arrays=335,private_arrays=1,public_files=336,private_files=2),
        security=dict(target_bits=128,attack='classical',degree=131072,q_towers=100,p_towers=14,
            dnum=8,qp_bits=3403,qp_inclusive_ceiling=3523,p_product_bits=415,
            largest_q_partition_bits=390,sigma=3.190000057220458984375,
            secret_distribution='uniform_ternary',noise_scale=1,key_generation_version=3,
            qp_is_secure=True,p_coverage=True),
        backend=dict(dependency_policy_sha256=policies['dependency_policy_sha256'],
            observed_device_platforms=['cpu'],jax_enable_x64=True,genuine_backend_imports_completed=True),
        policies=clone(policies),checks={name:True for name in (
            'model_admission_authenticated','source_precheck_completed','real_security_predicates_completed',
            'resource_admission_completed','capacity_checks_completed','external_parent_identity_checked_before_creation',
            'pre_rng_checks_completed','original_secret_operand_preserved','all335_public_arrays_authenticated',
            'secret_array_authenticated','public_private_manifest_link_authenticated','all_owned_files_fsynced',
            'root_and_parent_fsync_completed','final_readback_completed','original_owned_identity_unchanged',
            'cleanup_completed','source_postcheck_completed','signals_restored')},
        scope=clone(scope),process=dict(pid=303,pgid=302),audit_denials=[])
    if change_report: change_report(report)
    report_raw = canonical(report)
    links = dict(report_sha256=sha(report_raw),intent_sha256=sha(intent.raw),grant_sha256=sha(grant_raw),
        source_manifest_sha256=source_hash,source_archive_sha256=archive,context_id=context,
        public_manifest_sha256=sha(public_raw),private_manifest_sha256=private_hash,
        stores_sha256=sha(canonical(report['stores'])))
    outer = dict(schema='qwen-keygen-actual-outer-v1',**links,pid=303,pgid=302,
                 exit_code=0,normal_return=True,reaped=True,process_group_absent=True)
    if change_outer: change_outer(outer)
    outer_raw = canonical(outer)
    review = dict(schema='qwen-keygen-root-readiness-v1',passed=True,**links,
        outer_sha256=sha(outer_raw),model_id=MODEL_ID,model_evidence=clone(model_evidence),
        role_evidence=clone(roles),policies=clone(policies),all_required_links_verified=True,
        source_archive_members_verified=True,current_owned_stores_verified=True,
        downstream_execution_authorized=False)
    if change_root: change_root(review)
    review_raw = canonical(review)
    ready_pins = dict(schema='qwen-keygen-ready-pins-v1',**links,
                      outer_sha256=sha(outer_raw),root_review_sha256=sha(review_raw))
    if change_ready_pins: change_ready_pins(ready_pins)
    ready_expected = ExpectedReadyPins(canonical(ready_pins))
    def ready():
        return authenticate_readiness(launch(),public_raw,report_raw,outer_raw,review_raw,
                                      expected_pins=ready_expected)
    return SimpleNamespace(public=public,profile=profile,model=model,intent=intent,
        grant=grant,grant_raw=grant_raw,launch_pins=launch_pins,launch_expected=launch_expected,
        launch=launch,public_raw=public_raw,report=report,report_raw=report_raw,outer=outer,
        outer_raw=outer_raw,review=review,review_raw=review_raw,ready_pins=ready_pins,
        ready_expected=ready_expected,ready=ready)
