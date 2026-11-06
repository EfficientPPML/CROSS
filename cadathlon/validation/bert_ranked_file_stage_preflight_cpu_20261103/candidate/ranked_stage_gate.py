"""Future ranked producer proof schema. It cannot accept size0 predecessor proofs.

The externally reviewed source inventory and caller-pinned terminal report are
mandatory. There is no executable worker/collector or launch path in this file.
Payload readback remains a separate operation after public authentication.
"""
from pathlib import Path
import io,math,tarfile
import key_gate
from ranked_case_gate import RankedCase
from ranked_readback import ranked_count


def _archive(raw,expected):
    with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
        members=tar.getmembers()
        if (len(members)!=len(expected) or len({m.name for m in members})!=len(members)
                or {m.name for m in members}!=set(expected) or any(not m.isfile() or m.size>8<<20 for m in members)
                or sum(m.size for m in members)>16<<20):
            raise ValueError('exact bounded reviewed source archive required')
        if {m.name:key_gate.digest(tar.extractfile(m).read()) for m in members}!=expected:
            raise ValueError('reviewed source archive member bytes differ')


def case_binding(case):
    if type(case)is not RankedCase:raise ValueError('authenticated ranked case descriptor required')
    ranked_count(case.count)
    return dict(size=case.size,seed=case.seed,count=case.count,input_sha256=case.input_sha256,
        target_order_sha256=case.target_order_sha256,fixture_report_sha256=case.fixture_report_sha256,
        reference_report_sha256=case.reference_report_sha256,candidate_sha256=case.candidate_sha256,
        model_revision=case.model_revision)


def _summaries(stage,value,*,count,previous):
    if type(value)is not dict or type(value.get('sample_count'))is not int or value['sample_count']!=count:
        raise ValueError('complete ranked stage sample count required')
    key_gate.sha_field(value.get('public_batch_sha256'))
    if previous is not None and value['public_batch_sha256']!=previous['readback']['public_batch_sha256']:
        raise ValueError('ranked predecessor batch identity differs')
    if stage in (3,4):
        import transport_contract as tc
        import file_stages
        p=tc.parameters();shape=[1,2,*p['degree_layout'],100 if stage==3 else 6];per=math.prod(shape)*4
        if (type(value.get('ciphertext_payload_bytes'))is not int
                or value['ciphertext_payload_bytes']!=count*per*(8 if stage==3 else 1)):
            raise ValueError('complete ranked payload accounting differs')
        inputs,candidate=file_stages._inputs()
        check_scale=file_stages._scale_checker(inputs,candidate) if stage==4 else None
        name='requests' if stage==3 else 'responses';rows=value.get(name)
        if type(rows)is not list or len(rows)!=count:raise ValueError('complete ranked ordered readback rows required')
        nonces=[]
        for i,row in enumerate(rows):
            if type(row)is not dict or type(row.get('ordinal'))is not int or row['ordinal']!=i:
                raise ValueError('ranked readback ordinal differs')
            nonce=row.get('request_id')
            if type(nonce)is not str or len(nonce)!=32 or any(c not in '0123456789abcdef' for c in nonce):
                raise ValueError('ranked opaque request identity differs')
            nonces.append(nonce);key_gate.sha_field(row.get('input_digest'))
            if stage==3:
                tiles=row.get('tiles')
                if type(tiles)is not list or len(tiles)!=8 or any(type(t)is not dict or type(t.get('tile'))is not int or t['tile']!=j for j,t in enumerate(tiles)):
                    raise ValueError('all-eight ordered request metadata required per sample')
                for tile in tiles:
                    key_gate.sha_field(tile.get('manifest_sha256'))
                    if (type(tile.get('payload_bytes'))is not int or tile['payload_bytes']!=per
                            or type(tile.get('shape'))is not list or any(type(x)is not int for x in tile['shape']) or tile['shape']!=shape
                            or type(tile.get('scale'))is not float or tile['scale']!=p['scaling_factor']):
                        raise ValueError('canonical complete request metadata differs')
            else:
                old=previous['readback']['requests'][i]
                if row['request_id']!=old['request_id'] or row['input_digest']!=old['input_digest']:
                    raise ValueError('ranked response order/request binding differs')
                key_gate.sha_field(row.get('response_manifest_sha256'))
                if type(row.get('level'))is not int or row['level']!=2 or type(row.get('scale'))is not float or not math.isfinite(row['scale']) or row['scale']<=0:
                    raise ValueError('actual ranked response scale/level required')
                if (check_scale(row['scale'])is not True or type(row.get('payload_bytes'))is not int or row['payload_bytes']!=per
                        or type(row.get('shape'))is not list or any(type(x)is not int for x in row['shape']) or row['shape']!=shape):
                    raise ValueError('independent response scale/layout differs')
        if len(set(nonces))!=count:raise ValueError('unique ranked nonce inventory required')
        key='tile_count' if stage==3 else 'response_count'
        if type(value.get(key))is not int or value[key]!=(8*count if stage==3 else count):
            raise ValueError('complete ranked ciphertext count required')
        if stage==4:key_gate.sha_field(value.get('response_batch_sha256'))
    else:
        metrics=value.get('metrics')
        if type(metrics)is not dict or metrics.get('sample_count')!=count or type(metrics.get('sample_count'))is not int:
            raise ValueError('complete ranked client metrics required')
        for key in ('he_correct','official_correct'):
            if type(metrics.get(key))is not int or not 0<=metrics[key]<=count:raise ValueError('ranked integer correctness counts required')
        he,native=metrics['he_correct'],metrics['official_correct']
        if not (4*he>=3*count and 100*(native-he)<=2*count):raise ValueError('original ranked accuracy gates failed')
        if any(metrics.get(k)is not True for k in ('all_ordered_finite_logits_validated','postprocessed_predictions_validated','ranked_accuracy_floor_passed','ranked_accuracy_drop_passed','final_ranked_acceptance')):
            raise ValueError('complete ranked client postprocessing/acceptance required')
        if value.get('private_receipt_validated')is not True or value.get('result_permissions_0600')is not True:
            raise ValueError('complete private ranked receipt/permissions required')
        for name in ('decoded_result_sha256','completed_receipt_sha256'):key_gate.sha_field(value.get(name))


def authenticate(evidence,report_sha256,*,stage,case,expected_sources,context_id,model_id,
                 key_report_sha256,model_report_sha256,previous=None):
    if type(stage)is not int or stage not in (3,4,5):raise ValueError('one ranked stage3/4/5 required')
    if (stage==3)!=(previous is None):raise ValueError('exact predecessor stage proof required')
    if previous is not None and (type(previous)is not dict or previous.get('stage')!=stage-1 or previous.get('case_binding')!=case_binding(case)):
        raise ValueError('ranked predecessor stage/case differs')
    if type(expected_sources)is not dict or not expected_sources:raise ValueError('externally reviewed nonempty source inventory required')
    for name,h in expected_sources.items():
        if type(name)is not str or Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('relative reviewed source inventory required')
        key_gate.sha_field(h)
    for h in (context_id,model_id,key_report_sha256,model_report_sha256):key_gate.sha_field(h)
    evidence=Path(evidence);raw=key_gate.read(evidence/'report.json',16<<20)
    if key_gate.digest(raw)!=key_gate.sha_field(report_sha256):raise ValueError('externally pinned ranked terminal report differs')
    report=key_gate.parse(raw)
    flags=('passed','execute','actual_execution_attempted','operation_complete','all_processes_reaped',
        'source_files_unchanged','source_archive_verified','cleanup_completed','terminal_scope_completed','owned_artifacts_retained')
    if (type(report)is not dict or report.get('schema')!='cross-bert-ranked-stage-proof-v1'
            or type(report.get('stage'))is not int or report['stage']!=stage
            or any(report.get(k)is not True for k in flags) or report.get('error') or report.get('cleanup_errors')):
        raise ValueError('completed successful ranked producer terminal proof required')
    if any(report.get(k)!=expected_sources for k in ('source_hashes_start','source_hashes_end','execution_source_hashes_end')):
        raise ValueError('ranked producer reviewed source closure differs')
    archive=key_gate.read(evidence/'sources.tar.gz',16<<20)
    if key_gate.digest(archive)!=key_gate.sha_field(report.get('source_archive_sha256')):raise ValueError('ranked producer archive digest differs')
    _archive(archive,expected_sources)
    proc=report.get('worker_process')
    if (type(proc)is not dict or type(proc.get('returncode'))is not int or proc['returncode']!=0
            or proc.get('process_reaped')is not True or proc.get('error') or type(proc.get('result'))is not dict
            or proc['result'].get('passed')is not True):raise ValueError('ranked worker successful terminal result required')
    worker_raw=key_gate.read(evidence/'worker/worker_report.json',16<<20)
    if key_gate.digest(worker_raw)!=key_gate.sha_field(proc['result'].get('worker_report_sha256')):raise ValueError('ranked worker report bytes differ')
    worker=key_gate.parse(worker_raw)
    wf=('passed','actual_stage_started','actual_stage_completed','readback_complete','source_files_unchanged','cleanup_completed','terminal_scope_completed')
    if (type(worker)is not dict or any(worker.get(k)is not True for k in wf) or worker.get('error') or worker.get('cleanup_errors') or worker.get('forbidden_attempts')
            or any(worker.get(k)!=expected_sources for k in ('source_hashes_start','source_hashes_end'))):
        raise ValueError('ranked worker complete phase/source/cleanup proof required')
    phases=dict(encryption=stage==3,decryption=stage==5,server_compute=stage==4,model_forward=False,key_generation=False)
    if type(worker.get('phases'))is not dict or key_gate.canonical(worker['phases'])!=key_gate.canonical(phases):
        raise ValueError('ranked stage phase separation differs')
    detail_flags={3:('all_ordered_input_bindings_validated',),4:('full_model_graph_completed','all_ordered_response_bindings_validated'),5:('all_ordered_private_outputs_validated',)}[stage]
    if any(worker.get(n)is not True for n in detail_flags):raise ValueError('all-record ranked stage completion required')
    bindings=dict(case_binding=case_binding(case),context_id=context_id,model_id=model_id,key_report_sha256=key_report_sha256,
        model_report_sha256=model_report_sha256,previous_stage_report_sha256=None if previous is None else previous['report_sha256'])
    for value in (report,worker):
        if key_gate.canonical(value.get('bindings'))!=key_gate.canonical(bindings):raise ValueError('ranked stage prerequisite identity differs')
    if previous is not None and (previous.get('context_id')!=context_id or previous.get('model_id')!=model_id):
        raise ValueError('ranked predecessor context/model differs')
    summary=worker.get('readback')
    if key_gate.canonical(summary)!=key_gate.canonical(report.get('readback')):raise ValueError('ranked worker/collector readback linkage differs')
    _summaries(stage,summary,count=case.count,previous=previous)
    if stage==5 and summary['metrics']['official_correct']!=case.official_correct:raise ValueError('official reference aggregate differs')
    return dict(stage=stage,case_binding=case_binding(case),context_id=context_id,model_id=model_id,
        report_sha256=report_sha256,source_archive_sha256=report['source_archive_sha256'],readback=summary)
