"""Literal single-entrypoint operations; partial proofs never stand in for decode/HE."""
from contextlib import redirect_stdout
from pathlib import Path
import sys,time
import bridge_common as common,bridge_protocol as protocol
import ranked_worker as parent,ranked_execution as execution,key_gate

def invoke(name,paths,report):
    import ranked_entrypoints as entry
    function=entry.resolve(name)
    relative='upstream_adapter/stages.py' if name in ('client_preprocess_input','client_postprocess') else 'runtime/stage/file_stages.py'
    source=Path(function.__code__.co_filename).resolve()
    if source!=common.HERE/relative or common.digest(source.read_bytes())!=report['source_hashes_start'][relative]:raise ValueError('literal installed body source differs')
    report['invoked_entrypoints'].append(name)
    report['entrypoint_body']={'file':relative,'sha256':report['source_hashes_start'][relative]}
    started=time.perf_counter();result=entry.invoke(name,paths)
    report.setdefault('handler_durations',[]).append(dict(stage=name,seconds=time.perf_counter()-started,timer_kind='diagnostic-handler-call'))
    return result

def execute_phase(envelope,authorized,case,paths,keys,model,previous,report,calls):
    import file_stages,file_stage_io,ranked_readback as readback,transport_contract as tc
    phase=envelope['phase'];request=envelope['request'];stage=request['stage']
    protocol.require_fresh(paths,phase);report['invoked_entrypoints']=[]
    if phase in ('client_key_generation','server_preprocess_model','server_encrypted_compute'):
        # These existing stages already perform exactly one original handler.
        result=parent.execute(request,authorized,case,paths,keys,model,previous,report,calls)
        report['invoked_entrypoints']=[phase]
        return result
    partial=protocol.authenticate_partial(envelope,report['source_hashes_start'],report['bindings'])
    with parent.guard(stage,calls,paths),redirect_stdout(sys.stderr):
        if phase=='client_preprocess_input':
            raw_sha=parent.input_binding(paths,case,tokenized=False)
            report.update(actual_stage_started=True,actual_preprocessing_started=True)
            invoke(phase,paths,report)
            token_sha=parent.input_binding(paths,case,tokenized=True)
            report['actual_preprocessing_completed']=True
            summary=dict(sample_count=case.count,input_sha256=raw_sha,preprocessed_input_sha256=token_sha)
        elif phase=='client_encode_encrypt_input':
            token_sha=parent.input_binding(paths,case,tokenized=True)
            if partial['readback']['preprocessed_input_sha256']!=token_sha:raise ValueError('preprocessing output changed before encryption')
            report['actual_stage_started']=True;result=invoke(phase,paths,report)
            batch=file_stage_io.read_manifest(paths['upload']/'batch.json')
            summary,batch=readback.requests(paths,public=batch['public'],count=case.count,expected_batch_sha256=result['public_batch_sha256'])
            if batch['public']['context_id']!=keys['context_id'] or batch['public']['model_id']!=model['model_id']:raise ValueError('fresh encrypted setup differs')
            if parent.input_binding(paths,case,tokenized=True)!=token_sha:raise ValueError('preprocessed input changed during encoding')
            import file_stage_contract as batches,client_inputs
            records=client_inputs.read_preprocessed_jsonl(paths['preprocessed_input'])
            private=key_gate.parse(parent.private_bytes(paths['client_private']/'requests.json'))
            batches.validate_private_requests(private,batch,[r.target_idx for r in records],case.input_sha256)
            report.update(all_ordered_input_bindings_validated=True,actual_preprocessing_completed=True,
                preprocessing_producer_report_sha256=envelope['partial']['report_sha256'])
            if type(result.get('sample_count'))is not int or result['sample_count']!=case.count:raise ValueError('all-record encoding incomplete')
        else:
            old=previous['readback'];batch=file_stage_io.read_manifest(paths['upload']/'batch.json')
            if batch['public']['context_id']!=keys['context_id'] or batch['public']['model_id']!=model['model_id']:raise ValueError('response setup identity differs')
            readback.requests(paths,public=batch['public'],count=case.count,expected_batch_sha256=old['public_batch_sha256'])
            response=readback.responses(paths,batch=batch,expected_response_sha256=old['response_batch_sha256'])
            if tc.canonical(response)!=tc.canonical(old):raise ValueError('complete response producer changed')
            parent.input_binding(paths,case,tokenized=True)
            from ranked_private_reference import load
            refs=load(authorized,size=case.size,seed=case.seed)
            if phase=='client_decrypt_decode':
                report['actual_stage_started']=True;result=invoke(phase,paths,report)
                if type(result.get('sample_count'))is not int or result['sample_count']!=case.count or result.get('public_batch_sha256')!=old['public_batch_sha256']:raise ValueError('all-record decoding incomplete')
                summary=readback.client_results(paths,batch=batch,references=refs.records(),count=case.count)
            else:
                # Revalidate completed decode/private receipt before postprocess.
                before=readback.client_results(paths,batch=batch,references=refs.records(),count=case.count)
                if tc.canonical(before)!=tc.canonical(partial['readback']):raise ValueError('decoded partial output/receipt changed')
                report['actual_stage_started']=True;invoke(phase,paths,report)
                raw=parent.private_bytes(paths['predictions'])
                try:predictions=[int(s) for s in raw.decode('utf8').splitlines()]
                except Exception:raise ValueError('ranked predictions invalid') from None
                summary=readback.client_results(paths,batch=batch,references=refs.records(),count=case.count,predictions=predictions)
                if any(summary[n]!=before[n] for n in ('decoded_result_sha256','completed_receipt_sha256','public_batch_sha256')):raise ValueError('postprocess altered retained decode/receipt')
                report.update(all_ordered_private_outputs_validated=True,decryption_producer_report_sha256=envelope['partial']['report_sha256'])
                if summary['metrics']['final_ranked_acceptance']is not True:raise ValueError('official ranked integer gates failed')
    if calls:raise ValueError('forbidden action attempted')
    report.update(actual_stage_completed=True,readback_complete=True,readback=summary,official_harness_timing=False)
    return summary
