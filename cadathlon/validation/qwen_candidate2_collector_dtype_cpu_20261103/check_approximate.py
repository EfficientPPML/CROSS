"""Strict native-versus-frozen-approximate calibration, never validation fitting."""
from pathlib import Path
from contextlib import contextmanager
import argparse,gc,hashlib,importlib.metadata,importlib.util,io,json,math,os,signal,sys,tarfile,time,traceback
import numpy as np
P=Path(__file__).resolve().parent
SNAPSHOT=Path('/home/jianming_gatech/cross_validation_assets/qwen/c1899de289a04d12100db370d81485cdf75e47ca')
EXPECTED_MANIFEST='0af45a8fc6d13b9d429ee1af2311b2cb6ac0aa4cde02eca56478472762482489'
class NoJax:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('jax','jaxlib','jaxite_word'):return importlib.util.spec_from_loader(fullname,self)
    def create_module(self,spec):return None
    def exec_module(self,module):raise AssertionError('Approximate plaintext calibration forbids JAX/HE imports')
sys.meta_path.insert(0,NoJax())
import arithmetic
from native_capture import NativeCapture
import reference
import candidate_gate

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def validate_manifest_inputs(root,manifest):
    actual={name:sha(root/name) for name in manifest['source_input_sha256']}
    for name,expected in manifest['source_input_sha256'].items():
        if actual[name]!=expected:raise ValueError('Frozen candidate input mismatch: '+name)
    return actual
@contextmanager
def termination_signals():
    old={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGHUP)}
    def stop(sig,frame):raise SystemExit(128+sig)
    try:
        for sig in old:signal.signal(sig,stop)
        yield
    finally:
        for sig,handler in old.items():signal.signal(sig,handler)

def metric(actual,expected,limit):
    actual,expected=np.asarray(actual),np.asarray(expected)
    shape=actual.shape==expected.shape
    bad={'passed':False,'finite':False,'shape_matches':shape,'max_abs':None,'nrmse':None}
    if actual.dtype.kind not in 'fiu' or expected.dtype.kind not in 'fiu':return bad
    if not shape or actual.size==0 or not np.isfinite(actual).all() or not np.isfinite(expected).all():return bad
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        actual=actual.astype(np.float64);expected=expected.astype(np.float64)
        delta=actual-expected
    if not np.isfinite(actual).all() or not np.isfinite(expected).all() or not np.isfinite(delta).all():return bad
    error=float(np.max(np.abs(delta)));reference_scale=float(np.max(np.abs(expected)))
    rms_error=0. if error==0. else error*float(np.sqrt(np.mean((delta/error)**2)))
    rms_reference=0. if reference_scale==0. else reference_scale*float(np.sqrt(np.mean((expected/reference_scale)**2)))
    nrmse=rms_error/max(.01,rms_reference)
    if not math.isfinite(nrmse):return bad
    return {'passed':nrmse<=limit,'max_abs':error,'nrmse':nrmse,'finite':True,'shape_matches':True}

def summarize(records,expected_count):
    complete=sum(bool(r.get('approximate_complete')) for r in records)
    matches=sum(r.get('approximate_top1') is not None and r['approximate_top1']==r.get('native_top1') for r in records)
    return {'expected_contexts':expected_count,'records':len(records),'complete_approximate_contexts':complete,'top1_matches':matches,'top1_denominator':expected_count,
        'numeric_and_domain_pass':len(records)==expected_count and all(r.get('numeric_pass',False) and r.get('domain_pass',False) and r.get('exact_semantic_pass',False) for r in records),
        'complete':len(records)==expected_count and complete==expected_count}

def decode_trace(initial,run):
    teacher=list(initial);native_tokens=[];teacher_records=[]
    for i in range(4):
        native,approx,r=run(teacher,'teacher',i);teacher_records.append(r);choice=int(native.argmax());native_tokens.append(choice);teacher.append(choice)
    free=list(initial);free_tokens=[];free_records=[];stopped=False
    for i in range(4):
        if stopped:
            free_records.append({'step':i,'status':'unmeasured_suffix_after_approximate_failure','approximate_complete':False,'numeric_pass':False,'domain_pass':False,'exact_semantic_pass':False,'approximate_top1':None})
            continue
        native,approx,r=run(free,'free',i);free_records.append(r)
        if approx is None:stopped=True;continue
        choice=int(approx.argmax());free_tokens.append(choice);free.append(choice)
    return {'native_tokens':native_tokens,'approximate_free_tokens':free_tokens,'complete_sequence_agreement':len(free_tokens)==4 and free_tokens==native_tokens,'teacher_records':teacher_records,'free_records':free_records,'free_suffix_stopped':stopped}

def merge_ranges(target,source):
    for label,r in source.items():
        if label not in target:target[label]=dict(r);continue
        out=target[label]
        if out['domain']!=r['domain']:raise ValueError('Domain changed within frozen candidate')
        for key in ('count','below','above','nonfinite'):out[key]+=r[key]
        if r['minimum'] is not None:out['minimum']=r['minimum'] if out['minimum'] is None else min(out['minimum'],r['minimum'])
        if r['maximum'] is not None:out['maximum']=r['maximum'] if out['maximum'] is None else max(out['maximum'],r['maximum'])

def collect(args):
    args.output.mkdir(exist_ok=False,parents=True);start=time.perf_counter();primary=None;paths={};before={};assets={};model=None
    report={'passed':False,'complete':False,'smoke':args.smoke,'cases':[],'decode':[],'errors':[],'scope':__doc__,'validation_data_read':False,'coefficient_selection_performed':False,'encrypted_execution':False}
    try:
        write(args.output/'report.json',report)
        prerequisite=candidate_gate.verify(P)
        report['selection_prerequisite']=prerequisite
        import torch
        from transformers import AutoTokenizer,Qwen3ForCausalLM
        import transformers
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        manifest=json.loads((P/'candidate-v2/manifest.json').read_text())
        verified_inputs=prerequisite['verified']['selection_input_sha256']
        protocol=json.loads((P/'calibration/protocol.json').read_text());pins=json.loads((P/'calibration/model_pins.json').read_text());corpus=json.loads((P/'calibration/corpus.json').read_text())
        if sorted(os.sched_getaffinity(0))!=protocol['runtime']['cores']:raise ValueError('Calibration requires CPU140-143 affinity')
        assets={n:sha(SNAPSHOT/n) for n in pins['asset_sha256']}
        if assets!=pins['asset_sha256']:raise ValueError('Pinned model assets mismatch')
        names=['check_approximate.py','candidate_gate.py','collector_requirement.json','origin.json','arithmetic.py','numeric.py','native_capture.py','reference.py','test_arithmetic.py','test_native_capture.py','test_collector.py','test_prerequisites.py','preflight_collector.py','candidate-v2/manifest.json']
        names=list(dict.fromkeys(names+[n for group in prerequisite['verified'].values() for n in group]))
        paths={n:P/n for n in names}
        package=Path(transformers.__file__).parent
        software=[*sorted((package/'models/qwen3').glob('*.py')),*[package/n for n in ('activations.py','cache_utils.py','modeling_attn_mask_utils.py','modeling_rope_utils.py','modeling_utils.py','modeling_outputs.py','configuration_utils.py','tokenization_utils_base.py','tokenization_utils_fast.py')]]
        paths.update({'software/transformers/'+str(f.relative_to(package)):f for f in software})
        before={n:sha(f) for n,f in paths.items()}
        with tarfile.open(args.output/'sources.tar.gz','w:gz') as t:
            for n,f in paths.items():b=f.read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
        candidate_gate.verify_archive(args.output/'sources.tar.gz',before)
        report.update(verified_manifest_input_sha256=verified_inputs,source_hashes_before=before,source_archive_sha256=sha(args.output/'sources.tar.gz'),asset_sha256_before=assets,candidate_manifest_sha256=EXPECTED_MANIFEST,protocol_sha256=sha(P/'calibration/protocol.json'),selection_protocol_sha256=sha(P/'protocol.json'),corpus_sha256=sha(P/'calibration/corpus.json'),cpu_affinity=sorted(os.sched_getaffinity(0)),versions={n:importlib.metadata.version(n) for n in ('numpy','torch','transformers','safetensors','tokenizers')})
        tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True);rows=json.loads((P/'calibration/tokenized_calibration_inputs.json').read_text());computed=[]
        for item in corpus['calibration_prompts']:
            tokens=tokenizer(item['text'],add_special_tokens=False)['input_ids']
            if len(tokens)<32:raise ValueError('Calibration prompt too short')
            for S in (8,16,32):computed.append({'id':item['id'],'sequence_length':S,'raw_token_count':len(tokens),'token_ids':tokens[:S]})
        if computed!=rows:raise ValueError('Calibration token binding mismatch')
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
            selected=[next(r for r in rows if r['sequence_length']==S) for S in (8,16,32)] if args.smoke else rows
            shapes=set()
            for row in selected:
                S=row['sequence_length'];_,_,record=run(row['token_ids'],row['id']+f'_S{S}',f'prefill_S{S}',S not in shapes);shapes.add(S);report['cases'].append(record)
            seeds=corpus['decode_prompt_ids'][:1] if args.smoke else corpus['decode_prompt_ids']
            for index,name in enumerate(seeds):
                ids=next(r['token_ids'] for r in rows if r['id']==name and r['sequence_length']==16)
                def callback(ids,kind,step):return run(ids,name+f'_{kind}_{step}',f'decode_{kind}',index==0 and kind=='teacher' and step==0)
                report['decode'].append(dict(id=name,**decode_trace(ids,callback)))
        pref=summarize(report['cases'],len(selected));teacher=summarize([r for d in report['decode'] for r in d['teacher_records']],4*len(seeds));free=summarize([r for d in report['decode'] for r in d['free_records']],4*len(seeds))
        free_matches=sum(d['complete_sequence_agreement'] for d in report['decode'])
        required_pref=math.ceil(.9*len(selected));required_teacher=math.ceil(.9*4*len(seeds));required_free=math.ceil(.75*len(seeds))
        gates={'all_numeric_and_domains':all(x['numeric_and_domain_pass'] for x in (pref,teacher,free)),'aggregate_prefill_top1':pref['top1_matches']>=required_pref,'teacher_top1':teacher['top1_matches']>=required_teacher,'complete_free_sequences':free_matches>=required_free}
        report.update(prefill_summary=pref,per_suite={str(S):summarize([r for r in report['cases'] if r['sequence_length']==S],1 if args.smoke else 96) for S in (8,16,32)},teacher_summary=teacher,free_summary=free,required_prefill_top1_matches=required_pref,required_teacher_top1_matches=required_teacher,required_complete_free_sequences=required_free,complete_free_sequence_matches=free_matches,
            propagated_domain_ranges=aggregate,hook_invariance_checks=hook_checks,gates=gates,all_workload_entries_accounted=len(selected)+8*len(seeds),complete=not args.smoke and all(x['complete'] for x in (pref,teacher,free)),
            no_jax_imported=not any(n.split('.')[0] in ('jax','jaxlib','jaxite_word') for n in sys.modules),limits={'operator_block_nrmse':.05,'full_logit_nrmse':.10,'nrmse_floor':.01},
            limitations=['Strict candidate/calibration trajectory only. No held-out validation, hidden benchmark, language-task accuracy, HE or refresh claim.',
                'Invalid declared domains stop this validation trajectory; no clipping/oracle substitution. Uncomputable free-running suffixes are explicitly unmeasured.',
                'Native operator/block comparisons use the same actual token prefix as the approximate circuit, including approximate-chosen free-running tokens.',
                'Prefill top1 is aggregate across buckets; per-suite rates are descriptive. Numerical/domain gates apply to every required context.',
                'Source closure and installed versions are not an OS/binary reproducibility image.'])
        report['passed']=all(gates.values()) and report['no_jax_imported']
    except BaseException as exc:
        primary=exc;report['passed']=False;report['errors'].append({'type':type(exc).__name__,'message':str(exc)})
        if not isinstance(exc,(KeyboardInterrupt,SystemExit)):traceback.print_exc()
    finally:
        try:
            if model is not None:del model
            gc.collect()
        except BaseException as exc:
            report['passed']=False;report['errors'].append({'phase':'model_cleanup','type':type(exc).__name__,'message':str(exc)})
            if primary is None:primary=exc
            else:primary.add_note('Model cleanup failed: '+repr(exc))
        try:
            after={n:sha(f) for n,f in paths.items()};report['source_hashes_after']=after;report['source_files_unchanged']=bool(before) and before==after
            report['asset_sha256_after']={n:sha(SNAPSHOT/n) for n in assets};report['assets_unchanged']=bool(assets) and assets==report['asset_sha256_after']
            if not report['source_files_unchanged'] or not report['assets_unchanged']:report['passed']=False
        except BaseException as exc:
            report['passed']=False;report['errors'].append({'phase':'final_provenance','type':type(exc).__name__,'message':str(exc)})
            if primary is None:primary=exc
            else:primary.add_note('Final provenance failed: '+repr(exc))
        report['elapsed_s']=time.perf_counter()-start
        try:write(args.output/'report.json',report)
        except BaseException as exc:
            if primary is not None:primary.add_note('Final publication failed: '+repr(exc))
            else:raise
    if primary is not None:raise primary
    return 0 if report['passed'] else 1

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    with termination_signals():return collect(args)
if __name__=='__main__':raise SystemExit(main())
