"""Pure candidate1 freeze construction from independently pinned public proof.

No file access, subprocess, model, private journal or execution permission.
The caller must authenticate the pins independently, including actual process
exits/reaps and the reviewed reconstruction script. A caller-selected set of
digests is not an authorization capability. This is the public proof assembly
step after the separate source-bound private reconstruction has completed.
"""
from freeze_admission import canonical, sha, parse, fields, digest, integer, same
from freeze_admission import authenticate_complete_freeze

CANDIDATE='344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
SOURCE='ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a'
GRANT='c3983fd975047bba16066106e85f9ae0768ed1e8c0fa30bef4da02b53253cd80'
CALIBRATION='80eeaefad423167ee0710797d7eb76d8cf3a422e6f3b2e6069932d682835963b'
PROTOCOL='eea8e4b38b4c2719e63a0aa13028150564901305e9e5f1f9d0f2985c821cb70f'
MODEL='6ce7cff55898969d0b4f60140179ebf8ce34a3b26de886e21f2c0beaba3415f5'
VALIDATION='1507a9472ed7cecdd06453394b7839b4a85a70ad2b27a7c741bc5196fca409bb'
TOKENIZER='0c9f0521465f0fee5085c1c86e46cef25ada3b5216cda9098ea8d6b459d6516c'
TOKENIZER_REVIEW='eb567eb48e980d8ef91d5d72ba187ab6f35cf76cd09973f757900acf43186503'
ARITHMETIC='9689b843b4ec2d0a0ef724dbdca7a516e9a4f23da83f83af5571da13380e9a6f'
REFERENCE='667c8483648adcf583212b0f828347614423c1c1401807f4f9512612f74aa147'
INITIAL='bc77c79e104af3f2f7c0da709bbf5c9319483ce7ad9a1dbb7cb3b8002510afb3'
SELECTION='7ba0ffae120c7b68b51d2ec1fc040e51e3fd12d989f1ab73316f231091917ad9'
ROLES=('parent','worker','producer_outer','review','reconstruction_outer','source_manifest')
RECONSTRUCTION_PINS=('reconstruction_source_manifest_sha256','reconstruction_source_archive_sha256','reconstruction_grant_sha256')
PIN_FIELDS=(*ROLES,'source_archive_sha256','reconstruction_script_sha256',*RECONSTRUCTION_PINS)
TERMINAL=('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged',
    'terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash')
REVIEW_CHECKS=('producer_admitted','exact_private_inventory','calibration_identity','ordered_ledger','causal_prefixes',
    'metric_inventory_and_gates','domain_inventory_and_gates','input_statistics_and_aggregates','public_private_summary_match',
    'native_call_accounting','private_rechecks_completed','source_rechecks_completed','cleanup_completed','signal_restoration_completed')
COVERAGE_FIELDS=('ledger_entries','measured_contexts','candidate_complete','candidate_invalid','unmeasured_free_suffixes',
    'comparison_invalid','semantic_passes','native_forward_started','native_forward_completed','all_length_invariance','phase_counts')
GATE_FIELDS=('all_numeric_semantic_domains','trajectories_complete','prefill_matches','teacher_matches','whole_free_sequences')
REVIEW_BINDINGS=('candidate_manifest_sha256','producer_source_manifest_sha256','producer_source_archive_sha256',
    'producer_parent_report_sha256','producer_worker_report_sha256','producer_outer_receipt_sha256','calibration_sha256',
    'protocol_sha256','model_pins_sha256','grant_sha256','private_inventory_sha256','ledger_sha256',*RECONSTRUCTION_PINS)
SCOPE=dict(recorded_statistic_reconstruction=True,fresh_array_recomputation=False,
    argmax_recomputed_from_logits=False,invariance_recomputed_from_arrays=False)

def require(value,message):
    if value is not True:raise ValueError(message)

def flag_fields(value,names):
    for name in names:require(value.get(name) is True,'Required true field differs: '+name)

def bounded_public(raw):
    # The producer reports contain aggregate domain statistics and may exceed
    # the consumer's small freeze-receipt bound. No source archive is expanded.
    if type(raw) is not bytes or not 0<len(raw)<=32<<20:
        raise ValueError('Bounded immutable public report required')
    import json,math
    def pairs(rows):
        out={}
        for key,value in rows:
            if key in out:raise ValueError('Duplicate public report field')
            out[key]=value
        return out
    def invalid(value):raise ValueError('Nonfinite public report number')
    def number(value):
        result=float(value)
        if not math.isfinite(result):invalid(value)
        return result
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid,parse_float=number)

def exact_coverage(worker):
    kinds=worker.get('by_kind');fields(kinds,('prefill','teacher','free'))
    for kind,total in (('prefill',3072),('teacher',64),('free',64)):
        row=kinds[kind]
        fields(row,('expected','records','complete','candidate_invalid','comparison_invalid','unmeasured','top1_matches','top1_denominator'))
        for name in ('expected','records','complete','top1_denominator'):integer(row[name],total)
        for name in ('candidate_invalid','comparison_invalid','unmeasured'):integer(row[name],0)
        integer(row['top1_matches'],0,total)
    for name in ('expected_entries','retained_entries','candidate_complete','semantic_passes'):integer(worker.get(name),3200)
    integer(worker.get('original416_members'),416)
    integer(worker.get('complete_free_sequences'),12,16);integer(worker.get('complete_free_denominator'),16)
    for name in ('native_forward_started','native_forward_completed'):integer(worker.get(name),3232)
    require(same(worker.get('phase_counts'),dict(native=3200,exact=3200,candidate=3200)),'Full measured phase counts required')
    for name in ('native_invariance_by_length','exact_invariance_by_length'):
        require(same(worker.get(name),{str(k):True for k in range(1,33)}),'All32 producer invariance observations required')
    require(same(worker.get('native_forward_accounting'),dict(expected=3232,started=3232,completed=3232,passed=True)),
        'Exact native forward accounting required')
    for name in ('native','exact'):
        ranges=worker.get('input_ranges')
        require(type(ranges) is dict and type(ranges.get(name)) is dict,'Native/exact aggregate records required')
        integer(ranges[name].get('contexts'),3200)
    coverage=dict(ledger_entries=3200,measured_contexts=3200,candidate_complete=3200,candidate_invalid=0,
        unmeasured_free_suffixes=0,comparison_invalid=0,semantic_passes=3200,native_forward_started=3232,
        native_forward_completed=3232,all_length_invariance=True,phase_counts=dict(native=3200,exact=3200,candidate=3200))
    gates=dict(all_numeric_semantic_domains=True,trajectories_complete=True,prefill_matches=kinds['prefill']['top1_matches'],
        teacher_matches=kinds['teacher']['top1_matches'],whole_free_sequences=worker['complete_free_sequences'])
    require(gates['prefill_matches']>=2765 and gates['teacher_matches']>=58,'Fixed prefill/teacher thresholds required')
    require(same(worker.get('gates'),dict(all_numeric_semantic_domains=True,prefill_top1=True,teacher_top1=True,complete_free_sequences=True)),
        'Unchanged complete calibration gates required')
    return coverage,gates

def issue(blobs,pins):
    """Return candidate1 freeze bytes only after an externally pinned PASS chain.

    Exact pin authentication/actual process observation is the caller's duty.
    This function performs no I/O and grants no validation/model execution.
    """
    fields(blobs,ROLES);fields(pins,PIN_FIELDS)
    for value in pins.values():digest(value)
    for role in ROLES:
        require(type(blobs[role]) is bytes and sha(blobs[role])==pins[role],'Independent public proof digest differs: '+role)
    require(pins['source_manifest']==SOURCE,'Only the reviewed139 producer is supported')
    data={name:bounded_public(raw) for name,raw in blobs.items()}
    require(all(type(value) is dict for value in data.values()),'Public proof objects must be mappings')
    parent,worker,review=data['parent'],data['worker'],data['review']
    expected=data['source_manifest']
    require(type(expected) is dict and len(expected)==139,'Exact source inventory required')
    for row in (parent,worker):
        flag_fields(row,TERMINAL)
        require('error_type' not in row and row.get('audit_denials')==[],'Clean terminal report required')
        require(same(row.get('source_hashes_start'),expected) and same(row.get('source_hashes_end'),expected),'Source closure differs')
        require(row.get('source_manifest_sha256')==SOURCE,'Producer source pin differs')
    require(same(parent.get('worker'),worker) and parent.get('worker_report_sha256')==pins['worker'],'Parent-worker publication differs')
    flag_fields(parent,('actual_candidate_stage','candidate_validation_admissible','child_reaped','child_group_absent'))
    integer(parent.get('child_returncode'),0)
    require(type(parent.get('source_archive')) is dict and parent['source_archive'].get('sha256')==pins['source_archive_sha256'],'Archive pin differs')
    for name in ('pid','child_pid','child_pgid'):integer(parent.get(name),1,2**31-1)
    integer(worker.get('pid'),parent['child_pid'])
    require(parent['child_pgid']==worker['pid'] and parent['pid']!=worker['pid'],'Distinct retired producer roles required')
    flag_fields(worker,('ledger_complete','trajectories_complete','all_length_invariance','complete_measured_context_coverage',
        'candidate_validation_admissible','calibration_reauthenticated','private_directory_handle_closed','private_outputs_retained',
        'external_inputs_never_removed','model_asset_postchecks_completed','model_cleanup_completed','native_capture_cleanup_completed',
        'model_loaded','model_forward'))
    for name in ('coefficient_selection_performed','encrypted_execution','validation_inputs_read','tokenizer_execution',
        'contains_tokens_logits_or_failure_text','failed_attempt_terminal_candidate','revision_eligible_ledger'):
        require(worker.get(name) is False,'Forbidden phase or failure eligibility: '+name)
    integer(worker.get('candidate_ordinal'),1)
    require(worker.get('manifest_sha256')==CANDIDATE and worker.get('grant_sha256')==GRANT,'Candidate/grant pin differs')
    for name,value in (('initial_report_sha256',INITIAL),('selection_proof_sha256',SELECTION),('calibration_sha256',CALIBRATION),('protocol_sha256',PROTOCOL)):
        require(worker.get(name)==value,'Producer lineage differs: '+name)
    coverage,gates=exact_coverage(worker)
    fields(review,('schema','verification_passed','scientific_pass','complete_scientific_failure_eligible','candidate_ordinal',
        'bindings','checks','coverage','gates','aggregate_sha256',*SCOPE,'selection_authorized','validation_authorized','row0_authorized'))
    require(review['schema']=='qwen-candidate-recorded-statistic-review-v2','Recorded-statistic review schema required')
    flag_fields(review,('verification_passed','scientific_pass'))
    require(review['complete_scientific_failure_eligible'] is False,'A failed outcome cannot freeze')
    integer(review['candidate_ordinal'],1)
    for name,value in SCOPE.items():require(review[name] is value,'Explicit recorded-statistic scope required')
    for name in ('selection_authorized','validation_authorized','row0_authorized'):
        require(review[name] is False,'Reconstruction must not self-authorize a later phase')
    fields(review['checks'],REVIEW_CHECKS);flag_fields(review['checks'],REVIEW_CHECKS)
    fields(review['coverage'],COVERAGE_FIELDS);fields(review['gates'],GATE_FIELDS)
    require(same(review['coverage'],coverage) and same(review['gates'],gates),'Independent reconstructed coverage/gates differ')
    digest(review['aggregate_sha256']);fields(review['bindings'],REVIEW_BINDINGS)
    binding=review['bindings']
    for value in binding.values():digest(value)
    fixed=dict(candidate_manifest_sha256=CANDIDATE,producer_source_manifest_sha256=SOURCE,
        producer_source_archive_sha256=pins['source_archive_sha256'],producer_parent_report_sha256=pins['parent'],
        producer_worker_report_sha256=pins['worker'],producer_outer_receipt_sha256=pins['producer_outer'],
        calibration_sha256=CALIBRATION,protocol_sha256=PROTOCOL,model_pins_sha256=MODEL,grant_sha256=GRANT,
        **{name:pins[name] for name in RECONSTRUCTION_PINS})
    require(all(binding[name]==value for name,value in fixed.items()),'Reconstruction proof lineage differs')
    terminal=dict(parent_exit_code=0,worker_exit_code=0,parent_reaped=True,parent_group_absent=True,
        worker_reaped=True,worker_group_absent=True,source_verified=True,assets_verified=True,cleanup_completed=True,
        signal_restoration_completed=True,ordered_ledger_independently_verified=True,
        all_metric_and_domain_records_independently_verified=True)
    attempt=dict(ordinal=1,outcome='complete_pass',manifest_sha256=CANDIDATE,parent_report_sha256=pins['parent'],
        worker_report_sha256=pins['worker'],source_manifest_sha256=SOURCE,source_archive_sha256=pins['source_archive_sha256'],
        grant_sha256=GRANT,producer_outer_receipt_sha256=pins['producer_outer'],independent_review_sha256=pins['review'],
        reconstruction_outer_receipt_sha256=pins['reconstruction_outer'],reconstruction_script_sha256=pins['reconstruction_script_sha256'],
        terminal=terminal,coverage=coverage,gates=gates,**{name:pins[name] for name in RECONSTRUCTION_PINS})
    result=dict(schema='qwen-first-complete-calibration-pass-freeze-v2',protocol_sha256=PROTOCOL,
        arithmetic_sha256=ARITHMETIC,reference_sha256=REFERENCE,first_complete_calibration_pass=True,
        selection_permanently_closed=True,validation_previously_read=False,row0_previously_evaluated=False,
        validation_execution_authorized=False,row0_execution_authorized=False,
        metric_contract=dict(operators=421,blocks=28,logits=151936,normalization_inputs=113,
            operator_block_nrmse=.05,full_logit_nrmse=.10,semantic_nrmse=1e-5,semantic_max_abs=.001,
            semantic_top1=True,nrmse_floor=.01,all_propagated_domains=True),
        row0=dict(token_id=0,original_position=0,sequence_length=1,excluded_from_selection=True),
        attempts=[attempt],manifest_sha256=CANDIDATE,ordinal=1,validation_input_sha256=VALIDATION,
        tokenizer_report_sha256=TOKENIZER,tokenizer_review_sha256=TOKENIZER_REVIEW,model_pins_sha256=MODEL,**SCOPE)
    raw=canonical(result)
    frozen=authenticate_complete_freeze(raw,sha(raw),producer_outers=(blobs['producer_outer'],),
        reconstruction_outers=(blobs['reconstruction_outer'],),reviewed_script_sha256=(pins['reconstruction_script_sha256'],),
        reviewed_reconstruction_pins=({name:pins[name] for name in RECONSTRUCTION_PINS},))
    require(frozen.execution_authorized is False,'Freeze must not grant model or validation access')
    return raw
