"""Strict public held-out Qwen validation, gated on frozen full calibration."""
from pathlib import Path
import argparse,gc,hashlib,importlib.metadata,io,json,math,os,sys,tarfile,time,traceback
import numpy as np
import calibration_gate
import validation_gate
import published_gate
from validation_cleanup import cleanup_signals,termination_signals
from validation_terminal import terminal_scope
from check_approximate import cleanup_failure
import candidate_gate
import check_approximate as base
from check_approximate import sha,write,metric,summarize,decode_trace,merge_ranges,validate_manifest_inputs,NoJax
from native_capture import NativeCapture
import arithmetic,reference
P=Path(__file__).resolve().parent
SNAPSHOT=base.SNAPSHOT
EXPECTED_MANIFEST=base.EXPECTED_MANIFEST
VALIDATION_INPUTS=Path('/tmp/qwen-candidate2-published-validation-20261103/validation_inputs/tokenized_fixtures.json')
def load_validation_inputs(expected_sha):
    if sha(VALIDATION_INPUTS)!=expected_sha:raise ValueError('Original validation input manifest hash mismatch')
    rows=json.loads(VALIDATION_INPUTS.read_text())
    if len(rows)!=56 or len({r['id'] for r in rows})!=56:raise ValueError('Validation inventory must have56 unique records')
    for S in (8,16,32):
        group=[r for r in rows if r['kind']=='prefill' and r['sequence_length']==S]
        if len(group)!=16 or len({tuple(r['token_ids']) for r in group})!=16:raise ValueError('Validation prefill bucket mismatch')
    decode=[r for r in rows if r['kind']=='decode']
    if len(decode)!=8 or any(r['sequence_length']!=16 for r in decode):raise ValueError('Validation decode inventory mismatch')
    if any(len(r['token_ids'])!=r['sequence_length'] for r in rows):raise ValueError('Validation exact length mismatch')
    return rows

def collect(args):
    args.output.mkdir(exist_ok=False,parents=True);start=time.perf_counter();primary=None;paths={};before={};assets={};model=None
    report={'passed':False,'complete':False,'smoke':False,'cases':[],'decode':[],'errors':[],'scope':__doc__,'validation_data_read':False,'validation_input_read_attempted':False,'validation_input_read_status':'not_started','coefficient_selection_performed':False,'encrypted_execution':False}
    try:
        write(args.output/'report.json',report)
        requirements=validation_gate.verify(P)
        report['calibration_prerequisite']=calibration_gate.verify(args.calibration_report,args.calibration_source_archive,P,requirements)
        report['selection_prerequisite']=candidate_gate.verify(P)
        # None denotes an attempted read without confirmed returned bytes;
        #a failed attempt must not be reported as known-unread validation.
        report.update(validation_input_read_attempted=True,validation_data_read=None,validation_input_read_status='attempted')
        def confirmed_read():report.update(validation_data_read=True,validation_input_read_status='confirmed')
        report['published_input_prerequisite']=published_gate.verify_inputs(P,requirements,VALIDATION_INPUTS,on_read=confirmed_read)
        validation=load_validation_inputs(requirements['validation_input_sha256'])
        validation_protocol=json.loads((P/'validation_protocol.json').read_text())
        report['workload']=validation_protocol['workload'];report['scope']=validation_protocol['scope']
        import torch
        from transformers import AutoTokenizer,Qwen3ForCausalLM
        import transformers
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        manifest=json.loads((P/'candidate-v3/manifest.json').read_text())
        verified_inputs=report['selection_prerequisite']['verified']['selection_input_sha256']
        protocol=json.loads((P/'calibration/protocol.json').read_text());pins=json.loads((P/'calibration/model_pins.json').read_text());corpus={'decode_prompt_ids':[r['id'] for r in validation if r['kind']=='decode']}
        if sorted(os.sched_getaffinity(0))!=validation_protocol['runtime']['cores']:raise ValueError('Validation requires CPU124-127 affinity')
        assets={n:sha(SNAPSHOT/n) for n in pins['asset_sha256']}
        if assets!=pins['asset_sha256']:raise ValueError('Pinned model assets mismatch')
        names=[n for n in requirements['expected_runtime_source_hashes'] if not n.startswith('software/')]
        names+=['validate_heldout.py','calibration_gate.py','validation_gate.py','validation_requirement.json','validation_protocol.json','test_validation_gate.py','test_published_gate.py','test_validation_lifecycle.py','preflight_validation.py','preflight_test_entry.py','published_gate.py','validation_cleanup.py','validation_terminal.py']
        paths={n:P/n for n in names}
        paths['validation_inputs/tokenized_fixtures.json']=VALIDATION_INPUTS
        paths['calibration_prerequisite/report.json']=args.calibration_report
        paths['calibration_prerequisite/runtime_sources.tar.gz']=args.calibration_source_archive
        package=Path(transformers.__file__).parent
        software=[*sorted((package/'models/qwen3').glob('*.py')),*[package/n for n in ('activations.py','cache_utils.py','modeling_attn_mask_utils.py','modeling_rope_utils.py','modeling_utils.py','modeling_outputs.py','configuration_utils.py','tokenization_utils_base.py','tokenization_utils_fast.py')]]
        paths.update({'software/transformers/'+str(f.relative_to(package)):f for f in software})
        before={n:sha(f) for n,f in paths.items()}
        with tarfile.open(args.output/'sources.tar.gz','w:gz') as t:
            for n,f in paths.items():b=f.read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
        validation_gate.verify_archive(args.output/'sources.tar.gz',before)
        report.update(verified_manifest_input_sha256=verified_inputs,source_hashes_before=before,source_archive_sha256=sha(args.output/'sources.tar.gz'),asset_sha256_before=assets,candidate_manifest_sha256=EXPECTED_MANIFEST,protocol_sha256=sha(P/'calibration/protocol.json'),selection_protocol_sha256=sha(P/'selection/protocol.json'),validation_protocol_sha256=sha(P/'validation_protocol.json'),calibration_corpus_sha256=sha(P/'calibration/corpus.json'),validation_inputs_sha256=sha(VALIDATION_INPUTS),cpu_affinity=sorted(os.sched_getaffinity(0)),versions={n:importlib.metadata.version(n) for n in ('numpy','torch','transformers','safetensors','tokenizers')})
        tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True)
        rows=[]
        for item in validation:
            tokens=tokenizer(item['text'],add_special_tokens=False)['input_ids'];S=item['sequence_length']
            if len(tokens)<max(32,S) or tokens[:S]!=item['token_ids']:raise ValueError('Held-out input tokenizer identity mismatch')
            if item['kind']=='prefill':rows.append(item)
        model,loading=Qwen3ForCausalLM.from_pretrained(SNAPSHOT,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError(str(loading))
        model.eval();config=model.config.to_dict();fields=('num_hidden_layers','hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','vocab_size')
        if tuple(config[k] for k in fields)!=(28,1024,3072,16,8,128,151936):raise ValueError('Architecture mismatch')
        if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('CPU-only model required')
        if model.lm_head.weight.data_ptr()!=model.model.embed_tokens.weight.data_ptr():raise ValueError('Tied weights mismatch')
        weights={n:v.detach().numpy().astype(np.float64) for n,v in model.state_dict().items() if n!='lm_head.weight'}
        report.update(architecture={k:config[k] for k in fields},loading_info=loading)
        aggregate={};hook_checks=[]
        def native(ids):
            with torch.no_grad():return model(input_ids=torch.tensor([ids],dtype=torch.long),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),position_ids=torch.arange(len(ids))[None,:],use_cache=False).logits[:,-1].detach().numpy().astype(np.float64)
        with NativeCapture(model) as capture:
            def run(ids,label,group,check_hook=False):
                report['active_case']={'id':label,'group':group,'sequence_length':len(ids)}
                bare=native(ids) if check_hook else None
                capture.begin();expected=native(ids);native_blocks,native_ops=capture.end()
                if check_hook:
                    same=bool(np.array_equal(bare,expected));hook_checks.append({'id':label,'all_logits_bitwise_invariant':same})
                    if not same:raise ValueError('Native capture changed logits')
                embeddings=weights['model.embed_tokens.weight'][np.asarray([ids],dtype=np.int64)]
                exact=reference.forward(config,weights,embeddings);semantic=metric(exact,expected,1e-5);semantic['top1_match']=bool(np.array_equal(exact.argmax(-1),expected.argmax(-1)));semantic['passed']=semantic['passed'] and semantic['max_abs']<=1e-3 and semantic['top1_match']
                engine=arithmetic.QwenArithmetic(manifest);approx=None;failure=None;depth=None
                try:
                    output,_,_,_=arithmetic.approximate_forward(config,weights,embeddings,manifest,engine=engine);approx=output.data;depth=output.depth
                except arithmetic.InvalidCircuit as exc:failure={'label':exc.label,'record':exc.record,'kind':'domain_or_nonfinite'}
                merge_ranges(aggregate,engine.domains)
                opmetrics={n:metric(v,native_ops[n],.05) for n,v in engine.operators.items()}
                blockmetrics=[metric(v,native_blocks[i],.05) for i,v in enumerate(engine.blocks)]
                logitmetric=metric(approx,expected,.10) if approx is not None else None
                complete=approx is not None and len(opmetrics)==421 and len(blockmetrics)==28
                numeric=complete and all(v['passed'] for v in opmetrics.values()) and all(v['passed'] for v in blockmetrics) and logitmetric['passed']
                record={'id':label,'group':group,'sequence_length':len(ids),'token_ids':list(ids),'context_sha256':hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest(),
                    'native_top1':int(expected.argmax()),'approximate_top1':None if approx is None else int(approx.argmax()),'exact_semantic_pass':semantic['passed'],'exact_vs_native':semantic,
                    'approximate_complete':complete,'numeric_pass':bool(numeric),'domain_pass':failure is None and not engine.domain_failures(),'failure':failure,
                    'operator_metrics':opmetrics,'block_metrics':blockmetrics,'full_logit_metric':logitmetric,'all_final_logits_checked':151936 if approx is not None else 0,
                    'missing_operator_count':421-len(opmetrics),'missing_block_count':28-len(blockmetrics),'domain_observations':len(engine.domains),'depth':depth,'phase_costs':engine.phase_costs,'peak_abstract_magnitude':engine.peak}
                print(json.dumps({'id':label,'complete':complete,'numeric_pass':bool(numeric),'failure':None if failure is None else failure['label'],'top1_match':record['approximate_top1']==record['native_top1'],'logit_nrmse':None if logitmetric is None else logitmetric['nrmse']}),flush=True)
                return expected,approx,record
            selected=rows
            shapes=set()
            for row in selected:
                S=row['sequence_length'];_,_,record=run(row['token_ids'],row['id'],f'prefill_S{S}',S not in shapes);shapes.add(S);report['cases'].append(record)
            seeds=corpus['decode_prompt_ids']
            for index,name in enumerate(seeds):
                ids=next(r['token_ids'] for r in validation if r['id']==name and r['kind']=='decode')
                def callback(ids,kind,step):return run(ids,name+f'_{kind}_{step}',f'decode_{kind}',index==0 and kind=='teacher' and step==0)
                report['decode'].append(dict(id=name,**decode_trace(ids,callback)))
        pref=summarize(report['cases'],len(selected));teacher=summarize([r for d in report['decode'] for r in d['teacher_records']],4*len(seeds));free=summarize([r for d in report['decode'] for r in d['free_records']],4*len(seeds))
        free_matches=sum(d['complete_sequence_agreement'] for d in report['decode'])
        required_pref=math.ceil(.9*len(selected));required_teacher=math.ceil(.9*4*len(seeds));required_free=math.ceil(.75*len(seeds))
        gates={'all_numeric_and_domains':all(x['numeric_and_domain_pass'] for x in (pref,teacher,free)),'aggregate_prefill_top1':pref['top1_matches']>=required_pref,'teacher_top1':teacher['top1_matches']>=required_teacher,'complete_free_sequences':free_matches>=required_free}
        report.update(prefill_summary=pref,per_suite={str(S):summarize([r for r in report['cases'] if r['sequence_length']==S],16) for S in (8,16,32)},teacher_summary=teacher,free_summary=free,required_prefill_top1_matches=required_pref,required_teacher_top1_matches=required_teacher,required_complete_free_sequences=required_free,complete_free_sequence_matches=free_matches,
            propagated_domain_ranges=aggregate,hook_invariance_checks=hook_checks,gates=gates,all_workload_entries_accounted=len(selected)+8*len(seeds),complete=all(x['complete'] for x in (pref,teacher,free)),
            no_jax_imported=not any(n.split('.')[0] in ('jax','jaxlib','jaxite_word') for n in sys.modules),limits={'operator_block_nrmse':.05,'full_logit_nrmse':.10,'nrmse_floor':.01},
            limitations=[validation_protocol['scope']+' after successful frozen416-context calibration. No hidden benchmark, language-task accuracy, HE or refresh claim.',
                'Invalid declared domains stop this validation trajectory; no clipping/oracle substitution. Uncomputable free-running suffixes are explicitly unmeasured.',
                'Native operator/block comparisons use the same actual token prefix as the approximate circuit, including approximate-chosen free-running tokens.',
                'Prefill top1 is aggregate across buckets; per-suite rates are descriptive. Numerical/domain gates apply to every required context.',
                'Source closure and installed versions are not an OS/binary reproducibility image.'])
        report['passed']=all(gates.values()) and report['no_jax_imported']
    except BaseException as exc:
        primary=exc;report['passed']=False;report['errors'].append({'type':type(exc).__name__,'message':str(exc)})
        if not isinstance(exc,(KeyboardInterrupt,SystemExit)):traceback.print_exc()
    finally:
        body_error=primary
        with terminal_scope(report,args.output/'report.json',write,cleanup_signals,lambda:(body_error,primary)):
            try:
                if model is not None:del model
                gc.collect()
            except BaseException as exc:
                report['passed']=False;report['errors'].append({'phase':'model_cleanup','type':type(exc).__name__,'message':str(exc)})
                primary=cleanup_failure(primary,body_error,exc,'Model cleanup failed')
            try:
                after={n:sha(f) for n,f in paths.items()};report['source_hashes_after']=after;report['source_files_unchanged']=bool(before) and before==after
                report['asset_sha256_after']={n:sha(SNAPSHOT/n) for n in assets};report['assets_unchanged']=bool(assets) and assets==report['asset_sha256_after']
                if not report['source_files_unchanged'] or not report['assets_unchanged']:report['passed']=False
            except BaseException as exc:
                report['passed']=False;report['errors'].append({'phase':'final_provenance','type':type(exc).__name__,'message':str(exc)})
                primary=cleanup_failure(primary,body_error,exc,'Final provenance failed')
            report['elapsed_s']=time.perf_counter()-start
            try:write(args.output/'report.json',report)
            except BaseException as exc:
                primary=cleanup_failure(primary,body_error,exc,'Final publication failed')

    if primary is not None:raise primary
    return 0 if report['passed'] else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--calibration-report',type=Path,required=True);parser.add_argument('--calibration-source-archive',type=Path,required=True);args=parser.parse_args()
    try:
        with termination_signals():return collect(args)
    except BaseException as error:
        # Failure after scope exit must not leave a successful terminal record.
        try:
            record=json.loads((args.output/'report.json').read_bytes());record['passed']=False
            record.setdefault('errors',[]).append({'phase':'termination_scope','type':type(error).__name__,'message':str(error)})
            with cleanup_signals():write(args.output/'report.json',record)
        except BaseException as publication:
            if publication is not error:error.add_note('Final termination publication failed: '+repr(publication))
        raise
if __name__=='__main__':raise SystemExit(main())
