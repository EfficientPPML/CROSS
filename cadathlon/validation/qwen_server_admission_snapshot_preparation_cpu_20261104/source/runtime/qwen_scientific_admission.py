"""Pure full28 scientific prerequisite admission, with no execution authority.

All incoming proofs are public metadata bytes, independently hash-pinned by the
caller. No files, model arrays, private journals, keys, contexts or processes
are accessed. The actual server grant remains a separate boundary.
"""
from dataclasses import dataclass
from qwen_file_contract import (PublicContract,canonical,sha,same,hash_value,
    validate_request,validate_request_manifest,CANDIDATE,CHECKPOINT,CONFIG)
from client_role_common import obj,integer,source_map,model_pins
from keygen_authority import GeneratedKeyReadiness
import freeze_admission as freeze
import freeze_issuer as issuer
from scientific_source_pins import PREREQUISITES,HELDOUT_SOURCE

PHASE_EVIDENCE=('source_manifest_sha256','source_archive_sha256','grant_sha256','parent_report_sha256','worker_report_sha256','outer_receipt_sha256','root_review_sha256')
EVIDENCE={
 'calibration':PHASE_EVIDENCE+('reconstruction_source_manifest_sha256','reconstruction_source_archive_sha256','reconstruction_grant_sha256','reconstruction_script_sha256','reconstruction_review_sha256','reconstruction_outer_receipt_sha256','reconstruction_root_review_sha256'),
 'freeze':('receipt_sha256','issuer_source_manifest_sha256','issuer_source_archive_sha256','publication_script_sha256','publication_outer_receipt_sha256','root_publication_review_sha256'),
 'heldout':PHASE_EVIDENCE+('validation_input_sha256','tokenizer_report_sha256','tokenizer_review_sha256'),
 'row0':PHASE_EVIDENCE,
 'model':('manifest_sha256','report_sha256','outer_sha256','root_review_sha256','expected_pins_sha256','scientific_export_lineage_review_sha256'),
 'embedding':('provenance_receipt_sha256','producer_source_manifest_sha256','producer_source_archive_sha256','producer_outer_receipt_sha256','root_review_sha256',
    'request_producer_source_manifest_sha256','request_producer_source_archive_sha256','request_producer_conditional_grant_sha256','request_producer_phase_grant_sha256',
    'request_producer_report_sha256','request_producer_outer_sha256','request_producer_root_review_sha256')}
CHECKS=('all_external_pins_authenticated','all_source_archive_and_terminal_closures_verified','complete3200_calibration_pass_verified','independent_recorded_statistic_reconstruction_verified','first_pass_freeze_published_once','complete1088_heldout_pass_verified','separate_fixed_row0_pass_verified','strict_metric_semantic_domain_coverage_verified','scientific_model_export_identity_bridge_verified','fixed_row0_embedding_to_request_provenance_verified','candidate_unchanged_no_feedback_or_revision')
SCOPE=dict(verification_scope='recorded-statistic-calibration-and-source-bound-heldout-row0',full28_scientific_prerequisites_satisfied=True,
 actual_execution_authorized=False,he_precision_accepted=False,client_result_accepted=False,fresh_array_recomputation=False,
 argmax_recomputed_from_logits=False,invariance_recomputed_from_arrays=False,selection_authorized=False,candidate_revision_authorized=False,heldout_retry_authorized=False)
BINDING_FIELDS={'candidate_manifest_sha256','candidate_ordinal','protocol_sha256','arithmetic_sha256','reference_sha256','model_id','scientific_model_pins_sha256','checkpoint_sha256','config_sha256','server_source_manifest_sha256','server_source_archive_sha256','request_contract_sha256','request_manifest_sha256','input_digest','context_id','public_manifest_sha256','fixed_input'}
PROOF_LINKS={
 'calibration_parent':('calibration','parent_report_sha256'),'calibration_worker':('calibration','worker_report_sha256'),
 'calibration_source':('calibration','source_manifest_sha256'),'calibration_outer':('calibration','outer_receipt_sha256'),
 'calibration_root':('calibration','root_review_sha256'),'reconstruction_review':('calibration','reconstruction_review_sha256'),
 'reconstruction_outer':('calibration','reconstruction_outer_receipt_sha256'),'reconstruction_root':('calibration','reconstruction_root_review_sha256'),
 'freeze_receipt':('freeze','receipt_sha256'),'freeze_outer':('freeze','publication_outer_receipt_sha256'),'freeze_root':('freeze','root_publication_review_sha256'),
 'heldout_parent':('heldout','parent_report_sha256'),'heldout_worker':('heldout','worker_report_sha256'),'heldout_source':('heldout','source_manifest_sha256'),
 'heldout_outer':('heldout','outer_receipt_sha256'),'heldout_root':('heldout','root_review_sha256'),
 'row0_parent':('row0','parent_report_sha256'),'row0_worker':('row0','worker_report_sha256'),'row0_source':('row0','source_manifest_sha256'),
 'row0_outer':('row0','outer_receipt_sha256'),'row0_root':('row0','root_review_sha256'),
 'model_lineage':('model','scientific_export_lineage_review_sha256'),
 'embedding_receipt':('embedding','provenance_receipt_sha256'),'embedding_outer':('embedding','producer_outer_receipt_sha256'),'embedding_root':('embedding','root_review_sha256')}
# Preserve the existing 32MiB producer-report ceiling; small new root receipts
# use 64KiB. These are parser ceilings, not a process memory/resource grant.
PROOF_LIMITS={name:1<<16 for name in PROOF_LINKS}
PROOF_LIMITS.update({name:32<<20 for name in ('calibration_parent','calibration_worker','heldout_parent','heldout_worker','row0_parent','row0_worker')})
PROOF_LIMITS.update({name:1<<20 for name in ('calibration_source','heldout_source','row0_source','reconstruction_review','calibration_outer','reconstruction_outer','freeze_receipt')})
PROOF_BUNDLE_LIMIT=sum(PROOF_LIMITS.values())
TERMINAL_FLAGS=('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash')


def _json(raw,limit=32<<20):
    if type(raw)is not bytes or not 0<len(raw)<=limit:raise ValueError('bounded immutable public scientific proof required')
    return issuer.bounded_public(raw)


def _closed(raw,names,schema,limit=1<<16):
    value=_json(raw,limit);obj(value,names,schema)
    if value['schema']!=schema or canonical(value)!=raw:raise ValueError('exact canonical scientific metadata required')
    return value


def _hashes(value,fields,label):
    obj(value,fields,label)
    for h in value.values():hash_value(h)
    return value


def _binding(value):
    obj(value,BINDING_FIELDS,'scientific binding')
    for n in BINDING_FIELDS-{'candidate_ordinal','fixed_input'}:hash_value(value[n])
    integer(value['candidate_ordinal'],1,'candidate ordinal',1)
    fixed=dict(candidate_manifest_sha256=CANDIDATE,protocol_sha256=issuer.PROTOCOL,arithmetic_sha256=issuer.ARITHMETIC,
        reference_sha256=issuer.REFERENCE,scientific_model_pins_sha256=issuer.MODEL,checkpoint_sha256=CHECKPOINT,config_sha256=CONFIG)
    for n,v in fixed.items():same(value[n],v,'fixed scientific identity differs')
    original=value['fixed_input'];fields={'kind','token_id','sequence_length','original_positions','shape','dtype','embedding_file_sha256','embedding_raw_sha256','embedding_phase_grant_sha256'}
    obj(original,fields,'fixed row0 input')
    for n in ('embedding_file_sha256','embedding_raw_sha256','embedding_phase_grant_sha256'):hash_value(original[n])
    same({n:original[n]for n in fields if not n.endswith('_sha256')},dict(kind='checkpoint-token0-row0',token_id=0,sequence_length=1,original_positions=[0],shape=[1,1,1024],dtype='float64'),'exact fixed row0 geometry required')
    return value


@dataclass(frozen=True)
class ExpectedScientificPins:
    """Must originate outside the checked statement and evidence bundle."""
    raw:bytes
    def check(self):
        if type(self)is not ExpectedScientificPins:raise TypeError('exact independent scientific pins required')
        value=_closed(self.raw,('schema','statement_sha256','bindings','evidence','server_source_map'),'qwen-full28-scientific-expected-pins-v1',2<<20)
        hash_value(value['statement_sha256']);b=_binding(value['bindings'])
        obj(value['evidence'],EVIDENCE,'scientific evidence groups')
        for name,fields in EVIDENCE.items():_hashes(value['evidence'][name],fields,name)
        selected=source_map(value['server_source_map'])
        if sha(canonical(selected))!=b['server_source_manifest_sha256']:raise ValueError('independent actual server source map differs')
        for name,h in PREREQUISITES.items():
            if selected.get(name)!=h:raise ValueError('scientific prerequisite source differs')
        if not {'qwen_scientific_admission.py','scientific_source_pins.py'}<=set(selected):raise ValueError('actual selected scientific verifier missing')
        return value


def _root(raw,schema,links,checks):
    expected=dict(schema=schema,passed=True,**links,checks={n:True for n in checks},actual_execution_authorized=False)
    same(_json(raw,1<<16),expected,'independent scientific root proof differs')


def _calibration(proofs,evidence):
    c=evidence['calibration'];f=evidence['freeze']
    roles=dict(parent=proofs['calibration_parent'],worker=proofs['calibration_worker'],producer_outer=proofs['calibration_outer'],review=proofs['reconstruction_review'],reconstruction_outer=proofs['reconstruction_outer'],source_manifest=proofs['calibration_source'])
    pins={name:sha(raw)for name,raw in roles.items()}
    pins|={name:c[name]for name in ('source_archive_sha256','reconstruction_script_sha256','reconstruction_source_manifest_sha256','reconstruction_source_archive_sha256','reconstruction_grant_sha256')}
    # Exact existing issuer independently checks every strict calibration gate
    # and invokes the preexisting complete-freeze verifier. It issues no grant.
    expected_freeze=issuer.issue(roles,pins)
    if expected_freeze!=proofs['freeze_receipt']:raise ValueError('published first-PASS freeze differs from admitted calibration chain')
    frozen=freeze.authenticate_complete_freeze(proofs['freeze_receipt'],f['receipt_sha256'],
        producer_outers=(proofs['calibration_outer'],),reconstruction_outers=(proofs['reconstruction_outer'],),
        reviewed_script_sha256=(c['reconstruction_script_sha256'],),
        reviewed_reconstruction_pins=({n:c[n]for n in issuer.RECONSTRUCTION_PINS},))
    if c['source_manifest_sha256']!=issuer.SOURCE or c['grant_sha256']!=issuer.GRANT:raise ValueError('only admitted original candidate1 calibration chain supported')
    _root(proofs['calibration_root'],'qwen-calibration-pass-root-admission-v1',
        {n:c[n]for n in PHASE_EVIDENCE if n!='root_review_sha256'}|dict(candidate_manifest_sha256=frozen.manifest_sha256),
        ('source_archive_members_verified','actual_normal_zero_parent_worker_retired','public_reports_authenticated','complete3200_pass_verified'))
    _root(proofs['reconstruction_root'],'qwen-calibration-reconstruction-root-admission-v1',
        {n:c[n]for n in ('reconstruction_source_manifest_sha256','reconstruction_source_archive_sha256','reconstruction_grant_sha256','reconstruction_script_sha256','reconstruction_review_sha256','reconstruction_outer_receipt_sha256')}|dict(candidate_manifest_sha256=frozen.manifest_sha256),
        ('source_archive_members_verified','actual_normal_zero_reconstructor_retired','recorded_statistics_review_verified','publication_completed'))
    outer=_json(proofs['freeze_outer'],1<<16)
    for n in('pid','pgid'):integer(outer.get(n),1,'freeze publication '+n)
    outer_links={n:f[n]for n in ('receipt_sha256','issuer_source_manifest_sha256','issuer_source_archive_sha256','publication_script_sha256')}
    same(outer,dict(schema='qwen-first-pass-freeze-publication-outer-v1',**outer_links,pid=outer['pid'],pgid=outer['pgid'],exit_code=0,normal_return=True,reaped=True,process_group_absent=True,publication_completed=True),'independent freeze publication terminal differs')
    _root(proofs['freeze_root'],'qwen-first-pass-freeze-root-admission-v1',
        {n:f[n]for n in f if n!='root_publication_review_sha256'}|dict(candidate_manifest_sha256=frozen.manifest_sha256),
        ('issuer_source_archive_members_verified','complete_calibration_chain_verified','first_pass_published_once','publication_outer_verified','selection_permanently_closed'))
    return frozen


def _phase(mode,proofs,evidence,frozen):
    e=evidence[mode];parent=_json(proofs[mode+'_parent']);worker=_json(proofs[mode+'_worker']);sources=_json(proofs[mode+'_source'])
    if e['source_manifest_sha256']!=HELDOUT_SOURCE or len(source_map(sources))!=171:raise ValueError('exact reviewed171 heldout/row0 source required')
    n,calls=(1088,1120)if mode=='heldout'else(1,2)
    for record in(parent,worker):
        if type(record)is not dict or any(record.get(k)is not True for k in TERMINAL_FLAGS)or record.get('audit_denials')!=[]or 'error_type'in record:
            raise ValueError('complete clean scientific PASS terminal required')
        for field,value in dict(workload_mode=mode,source_manifest_sha256=HELDOUT_SOURCE,manifest_sha256=frozen.manifest_sha256,
                freeze_receipt_sha256=frozen.receipt_sha256,grant_sha256=e['grant_sha256'],selection_permanently_closed=True,revision_eligible_ledger=False).items():
            same(record.get(field),value,'same frozen actual phase required')
        for field in('source_hashes_start','source_hashes_end'):same(record.get(field),sources,'actual source before/after differs')
    same(parent.get('schema'),'qwen-heldout-workflow-parent-v1','parent phase schema differs')
    same(worker.get('schema'),'qwen-heldout-workflow-worker-v1','worker phase schema differs')
    same(parent.get('worker'),worker,'parent/worker publication differs')
    same(parent.get('worker_report_sha256'),e['worker_report_sha256'],'original worker bytes differ')
    if parent.get('actual_heldout_stage')is not True or parent.get('public_proofs_reauthenticated')is not True:raise ValueError('actual frozen phase admission and final public reauthentication required')
    for field in('pid','child_pid','child_pgid'):integer(parent.get(field),1,'phase '+field,2**31-1)
    integer(worker.get('pid'),parent['child_pid'],'original worker PID',parent['child_pid'])
    if parent['child_pgid']!=worker['pid']or parent['pid']==worker['pid']:raise ValueError('distinct actual parent/worker ownership required')
    integer(parent.get('child_returncode'),0,'actual child return code',0)
    if parent.get('child_reaped')is not True or parent.get('child_group_absent')is not True:raise ValueError('actual worker retirement required')
    archive=parent.get('source_archive')
    if type(archive)is not dict or archive.get('sha256')!=e['source_archive_sha256']:raise ValueError('actual source archive differs')
    flags=('ledger_complete','trajectories_complete','model_loaded','model_forward','model_asset_postchecks_completed','model_cleanup_completed','native_capture_cleanup_completed','public_proofs_reauthenticated','private_directory_handle_closed','private_outputs_retained','external_inputs_never_removed','complete_measured_context_coverage','all_length_invariance')
    if any(worker.get(k)is not True for k in flags):raise ValueError('full phase coverage/body postchecks required')
    for field in('coefficient_selection_performed','encrypted_execution','tokenizer_execution','revision_eligible_ledger','contains_tokens_logits_or_failure_text'):
        if worker.get(field)is not False:raise ValueError('forbidden phase/feedback claim')
    if worker.get('candidate_validation_admissible')is True or worker.get('failed_attempt_terminal_candidate')is True:raise ValueError('heldout/row0 cannot authorize revision')
    for field in('expected_entries','retained_entries','candidate_complete','semantic_passes'):integer(worker.get(field),n,field,n)
    integer(worker.get('candidate_ordinal'),1,'candidate ordinal',1)
    for field in('native_forward_started','native_forward_completed'):integer(worker.get(field),calls,field,calls)
    same(worker.get('phase_counts'),dict(native=n,exact=n,candidate=n),'full actual phase counts differ')
    same(worker.get('native_forward_accounting'),dict(expected=calls,started=calls,completed=calls,passed=True),'actual native accounting differs')
    lengths=range(1,33)if mode=='heldout'else(1,)
    for field in('native_invariance_by_length','exact_invariance_by_length'):same(worker.get(field),{str(k):True for k in lengths},'exact length invariance coverage required')
    same(worker.get('limits'),dict(operator_block_nrmse=.05,full_logit_nrmse=.10,semantic_nrmse=1e-5,semantic_max_abs=.001,nrmse_floor=.01),'unchanged strict metric limits required')
    for family in('native','exact'):
        ranges=worker.get('input_ranges')
        if type(ranges)is not dict or type(ranges.get(family))is not dict:raise ValueError('source-bound range coverage required')
        integer(ranges[family].get('contexts'),n,'complete range contexts',n)
    read=mode=='heldout'
    for field in('validation_inputs_read','validation_reauthenticated','validation_read_attempted','validation_input_authenticated'):same(worker.get(field),read,'exact heldout read scope required')
    same(worker.get('validation_read_status'),'confirmed'if read else'not_started','actual input read status differs')
    same(worker.get('validation_input_sha256'),frozen.validation_input_sha256,'same frozen validation identity required')
    if mode=='heldout':
        kinds=worker.get('by_kind');obj(kinds,('prefill','teacher','free'),'heldout kinds')
        for kind,total in(('prefill',1024),('teacher',32),('free',32)):
            row=kinds[kind];obj(row,('expected','records','complete','candidate_invalid','comparison_invalid','unmeasured','top1_matches','top1_denominator'),kind)
            for field in('expected','records','complete','top1_denominator'):integer(row[field],total,field,total)
            for field in('candidate_invalid','comparison_invalid','unmeasured'):integer(row[field],0,field,0)
            integer(row['top1_matches'],0,'diagnostic top1 count',total)
        if kinds['prefill']['top1_matches']<922 or kinds['teacher']['top1_matches']<29:raise ValueError('unchanged heldout agreement gate failed')
        integer(worker.get('complete_free_sequences'),6,'whole free sequences',8);integer(worker.get('complete_free_denominator'),8,'free denominator',8)
        same(worker.get('gates'),dict(all_numeric_semantic_domains=True,prefill_top1=True,teacher_top1=True,complete_free_sequences=True),'all strict heldout gates required')
        if worker.get('free_step_agreement_is_diagnostic')is not True:raise ValueError('no extra free-step gate permitted')
        for field,pin in(('validation_input_sha256',frozen.validation_input_sha256),('tokenizer_report_sha256',issuer.TOKENIZER),('tokenizer_review_sha256',issuer.TOKENIZER_REVIEW)):
            same(e[field],pin,'fixed validation/tokenizer proof differs')
    else:
        if worker.get('status')!='complete'or type(worker.get('top1_match'))is not bool:raise ValueError('complete row0 with diagnostic top1 required')
        if worker.get('original_frozen_failure_unchanged')is not True or worker.get('revision_eligible')is not False:raise ValueError('row0 cannot replace prior scientific history')
    outer=_json(proofs[mode+'_outer'],1<<16)
    links=dict(mode=mode,report_sha256=e['parent_report_sha256'],worker_report_sha256=e['worker_report_sha256'],source_manifest_sha256=HELDOUT_SOURCE,source_archive_sha256=e['source_archive_sha256'],candidate_manifest_sha256=frozen.manifest_sha256,freeze_receipt_sha256=frozen.receipt_sha256,grant_sha256=e['grant_sha256'])
    same(outer,dict(schema='qwen-heldout-workflow-outer-v1',**links,exit_code=0,worker_exit_code=0,reaped=True,process_group_absent=True,worker_reaped=True,worker_group_absent=True,publication_completed=True),'independent normal-zero phase outer required')
    _root(proofs[mode+'_root'],'qwen-frozen-scientific-phase-root-admission-v1',
        links|dict(outer_receipt_sha256=e['outer_receipt_sha256']),
        ('source_archive_members_verified','actual_normal_zero_parent_worker_retired','all_public_links_verified','strict_metric_domain_inventory_source_bound','complete_pass_counts_and_gates_verified','same_candidate_model_freeze_verified','no_feedback_or_revision'))


def _provenance(public,model,ready,request,upload,b,proofs,e):
    current=model_pins(model,public,b['server_source_manifest_sha256'])
    same({n:e['model'][n]for n in current},current,'independent actual server model evidence differs')
    if model.model_id!=b['model_id']:raise ValueError('aggregate model identity differs')
    if type(ready)is not GeneratedKeyReadiness:raise TypeError('independently admitted generated-key readiness required')
    ready.check()
    generation_model=ready.launch.expected_pins.check()['model_evidence']
    # Consumer-specific ExpectedModelPins legitimately differ between keygen
    # and server. The four actual body-validation proof bytes must not differ.
    proof_fields=('manifest_sha256','report_sha256','outer_sha256','root_review_sha256')
    same({n:current[n]for n in proof_fields},{n:generation_model[n]for n in proof_fields},'generated keys and actual server must use the same model body-validation proof')
    ids=ready.expected_ids(role='server');ids.require_identity(request['public'],'server')
    if ids.model_id!=b['model_id']or ids.context_id!=b['context_id']or ids.public_manifest_sha256!=b['public_manifest_sha256']:raise ValueError('request generated context/key identity differs')
    for field,value in(('candidate_manifest_sha256',request['public']['candidate_sha256']),('checkpoint_sha256',request['public']['checkpoint_sha256']),('config_sha256',request['public']['config_sha256'])):same(b[field],value,'request scientific identity differs')
    same(request['input']['original_positions'],[0],'fixed HE row0 original position required')
    if request['input']['sequence_length']!=1:raise ValueError('fixed row0 request requires S1')
    lineage_links=dict(model_id=model.model_id,scientific_model_pins_sha256=issuer.MODEL,checkpoint_sha256=CHECKPOINT,config_sha256=CONFIG,
        aggregate_manifest_sha256=sha(model.manifest_bytes),model_evidence=current)
    _root(proofs['model_lineage'],'qwen-scientific-model-export-lineage-v1',lineage_links,
        ('original_checkpoint_config_matches_scientific_sources','exact_BF16_float64_export_lineage_verified','all198_logical_and113_gamma_identity_verified'))
    embedding=public.json('inputs/decoder_parameters.json')['matrices']['model.embed_tokens.weight']
    fixed=b['fixed_input'];row=_json(proofs['embedding_receipt'],1<<16)
    request_evidence={n:e['embedding'][n]for n in EVIDENCE['embedding']if n.startswith('request_producer_')}
    request_role=ready.launch.expected_pins.check()['role_evidence']['request_producer']
    for suffix,field in(('source_manifest_sha256','source_manifest_sha256'),('source_archive_sha256','source_archive_sha256'),('conditional_grant_sha256','execution_grant_sha256')):
        same(request_evidence['request_producer_'+suffix],request_role[field],'embedding provenance and original generated-key request role differ')
    same(request_evidence['request_producer_source_manifest_sha256'],request['public']['producer_source_manifest_sha256'],'actual request producer provenance differs')
    same(request_evidence['request_producer_phase_grant_sha256'],fixed['embedding_phase_grant_sha256'],'same granted embedding input/request phase required')
    # The post-client provenance producer/root admits that complete nested
    # client terminal chain. This checker binds its hashes/attestation; it does
    # not independently reread client reports or repeat encryption/array work.
    expected=dict(schema='qwen-fixed-row0-embedding-provenance-v1',model_id=b['model_id'],checkpoint_sha256=CHECKPOINT,config_sha256=CONFIG,
        aggregate_manifest_sha256=sha(model.manifest_bytes),embedding_matrix_file_sha256=embedding['file_sha256'],embedding_matrix_raw_sha256=embedding['sha256'],
        fixed_input=fixed,request_contract_sha256=b['request_contract_sha256'],request_manifest_sha256=b['request_manifest_sha256'],input_digest=b['input_digest'],
        context_id=b['context_id'],public_manifest_sha256=b['public_manifest_sha256'],producer_source_manifest_sha256=e['embedding']['producer_source_manifest_sha256'],
        producer_source_archive_sha256=e['embedding']['producer_source_archive_sha256'],complete=True,source_and_input_postchecks_completed=True,
        original_checkpoint_row0_verified=True,no_replication_or_position_remapping=True,no_coefficient_selection=True,
        request_producer_evidence=request_evidence,request_producer_terminal_chain_authenticated=True,
        request_producer_normal_zero_reaped=True,request_producer_exact_input_and_upload_verified=True,
        nested_client_chain_verification='post-client-provenance-verifier-and-independent-root-attestation',actual_execution_authorized=False)
    same(row,expected,'independently bound original row0-to-upload provenance required')
    outer=_json(proofs['embedding_outer'],1<<16)
    for field in('pid','pgid'):integer(outer.get(field),1,'embedding producer '+field)
    links=dict(provenance_receipt_sha256=e['embedding']['provenance_receipt_sha256'],producer_source_manifest_sha256=e['embedding']['producer_source_manifest_sha256'],producer_source_archive_sha256=e['embedding']['producer_source_archive_sha256'])
    same(outer,dict(schema='qwen-row0-embedding-provenance-outer-v1',**links,pid=outer['pid'],pgid=outer['pgid'],exit_code=0,normal_return=True,reaped=True,process_group_absent=True,publication_completed=True),'independent embedding provenance terminal required')
    _root(proofs['embedding_root'],'qwen-row0-embedding-provenance-root-v1',links|dict(producer_outer_receipt_sha256=e['embedding']['producer_outer_receipt_sha256'],request_contract_sha256=b['request_contract_sha256'],request_manifest_sha256=b['request_manifest_sha256'],input_digest=b['input_digest'],request_producer_evidence=request_evidence),
        ('producer_source_archive_members_verified','original_embedding_model_row_identity_verified','exact_granted_input_file_and_request_verified','request_client_source_archive_and_conditional_role_verified','request_client_report_outer_root_chain_admitted','request_client_normal_zero_and_original_owners_retired','request_client_exact_input_and_upload_verified','normal_zero_publication_retired'))


@dataclass(frozen=True)
class ScientificAdmission:
    public:PublicContract
    model:object
    readiness:GeneratedKeyReadiness
    request_bytes:bytes
    request_manifest_bytes:bytes
    statement_bytes:bytes
    proof_bytes:tuple
    expected_pins:ExpectedScientificPins
    def check(self):
        if type(self)is not ScientificAdmission or type(self.proof_bytes)is not tuple or len(self.proof_bytes)!=len(PROOF_LINKS):raise TypeError('exact retained scientific admission required')
        if any(type(x)is not tuple or len(x)!=2 or type(x[0])is not str or type(x[1])is not bytes for x in self.proof_bytes)or len({x[0]for x in self.proof_bytes})!=len(PROOF_LINKS):raise ValueError('retained immutable public proof bundle differs')
        authenticate_scientific(self.public,self.model,self.readiness,self.request_bytes,self.request_manifest_bytes,self.statement_bytes,
            proofs=dict(self.proof_bytes),expected_pins=self.expected_pins)
        return self
    def bindings(self):
        self.check();return _json(self.statement_bytes,1<<16)['binding']
    @property
    def execution_authorized(self):return False


def authenticate_scientific(public,model,readiness,request_bytes,request_manifest_bytes,statement_bytes,*,proofs,expected_pins):
    if type(public)is not PublicContract or type(expected_pins)is not ExpectedScientificPins:raise TypeError('exact public source and independent scientific expectations required')
    public.check();pins=expected_pins.check()
    if sha(statement_bytes)!=pins['statement_sha256']:raise ValueError('independent scientific statement digest differs')
    statement=_closed(statement_bytes,('schema','binding','evidence','checks','scope'),'qwen-full28-scientific-admission-v1')
    b=_binding(statement['binding']);same(b,pins['bindings'],'independent exact scientific request binding differs')
    same(statement['evidence'],pins['evidence'],'independent exact scientific evidence hashes differ');e=statement['evidence']
    same(statement['checks'],{n:True for n in CHECKS},'all independently completed scientific checks required')
    same(statement['scope'],SCOPE,'scientific eligibility cannot grant execution or claim HE precision')
    obj(proofs,PROOF_LINKS,'complete public scientific proof bundle')
    for name,(group,field)in PROOF_LINKS.items():
        _json(proofs[name],PROOF_LIMITS[name])
        if sha(proofs[name])!=e[group][field]:raise ValueError('independently pinned public proof differs: '+name)
    request=_json(request_bytes,1<<20);upload=_json(request_manifest_bytes,1<<20)
    validate_request(public,request);validate_request_manifest(public,request,upload)
    if request_bytes!=canonical(request)or request_manifest_bytes!=canonical(upload):raise ValueError('canonical actual request/upload bytes required')
    if sha(request_bytes)!=b['request_contract_sha256']or sha(request_manifest_bytes)!=b['request_manifest_sha256']or upload['input_digest']!=b['input_digest']:raise ValueError('exact scientific request/upload/input differs')
    frozen=_calibration(proofs,e)
    if frozen.manifest_sha256!=b['candidate_manifest_sha256']or frozen.ordinal!=b['candidate_ordinal']:raise ValueError('published first-PASS candidate differs')
    _phase('heldout',proofs,e,frozen);_phase('row0',proofs,e,frozen)
    _provenance(public,model,readiness,request,upload,b,proofs,e)
    return ScientificAdmission(public,model,readiness,request_bytes,request_manifest_bytes,statement_bytes,tuple(sorted(proofs.items())),expected_pins)


def create_actual_scientific_authorization(*args,**kwargs):
    raise PermissionError('pure scientific proof admission cannot issue an actual execution authorization')
