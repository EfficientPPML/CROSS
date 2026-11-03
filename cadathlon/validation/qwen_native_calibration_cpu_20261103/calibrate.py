"""Independent calibration corpus: native/exact trajectories before coefficient selection.

No approximate circuit, HE, JAX, hidden-suite score or language-task accuracy.
"""
from pathlib import Path
from collections import defaultdict
import argparse,contextlib,datetime,gc,hashlib,importlib.metadata,importlib.util,io,json,math,os,sys,tarfile,time,traceback
import numpy as np
HERE=Path(__file__).resolve().parent
SNAPSHOT=Path('/home/jianming_gatech/cross_validation_assets/qwen/c1899de289a04d12100db370d81485cdf75e47ca')
GATES={'max_abs':1e-3,'nrmse':1e-5,'exact_top1':True}

class NoJax:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('jax','jaxlib','jaxite_word'):
            return importlib.util.spec_from_loader(fullname,self)
    def create_module(self,spec):return None
    def exec_module(self,module):raise AssertionError('Native audit forbids JAX/HE import: '+module.__name__)
sys.meta_path.insert(0,NoJax())

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def merge(destination,source):
    for key,row in source.items():
        if key not in destination:destination[key]=dict(row);continue
        out=destination[key]
        for field in ('count','below','above','nonfinite'):out[field]+=row[field]
        out['minimum']=min(out['minimum'],row['minimum'])
        out['maximum']=max(out['maximum'],row['maximum'])

def stats(values,domain=None):
    x=np.asarray(values,dtype=np.float64);finite=np.isfinite(x);v=x[finite]
    if not v.size:raise ValueError('observation has no finite values')
    lo,hi=domain if domain is not None else (-math.inf,math.inf)
    return {'domain':None if domain is None else list(domain),'minimum':float(v.min()),'maximum':float(v.max()),
            'count':int(x.size),'below':int(np.sum(x<lo)),'above':int(np.sum(x>hi)),'nonfinite':int(np.sum(~finite))}

def causal_centered(scores):
    """No padding/cache: exactly keys0..query, never future positions."""
    if scores.ndim!=4 or scores.shape[-2]!=scores.shape[-1]:raise ValueError('square full-prefill scores required')
    rows=[scores[...,q,:q+1] for q in range(scores.shape[-2])]
    centered=[x-x.mean(axis=-1,keepdims=True) for x in rows]
    return rows,centered

class NativeObserver:
    def __init__(self,model):
        self.model=model;self.handles=[];self.records={};self.missing=set();self.active=False
    def observe(self,label,value,domain=None):
        if hasattr(value,'detach'):value=value.detach().cpu().numpy()
        merge(self.records,{label:stats(value,domain)})
    def __enter__(self):
        from transformers.models.qwen3 import modeling_qwen3 as m
        import torch
        self.module=m;self.original=m.eager_attention_forward
        def norm(label,last=False):
            def hook(module,args):
                if not self.active:return
                x=args[0];x=x[:,-1:] if last else x
                variance=x.float().pow(2).mean(-1,keepdim=True)+module.variance_epsilon
                self.observe(label,variance)
            return hook
        def output(label):
            def hook(module,args,result):
                if self.active:self.observe(label,result)
            return hook
        for index,layer in enumerate(self.model.model.layers):
            for suffix,module in [('input',layer.input_layernorm),('post',layer.post_attention_layernorm),('q',layer.self_attn.q_norm),('k',layer.self_attn.k_norm)]:
                label=f'{index}.{suffix}.inverse_sqrt'
                self.handles.append(module.register_forward_pre_hook(norm(label)))
                self.handles.append(module.register_forward_hook(output(label+'.native_output')))
            def silu(module,args,label=f'{index}.silu'):
                if self.active:self.observe(label,args[0])
            self.handles.append(layer.mlp.act_fn.register_forward_pre_hook(silu))
        self.handles.append(self.model.model.norm.register_forward_pre_hook(norm('final.inverse_sqrt',last=True)))
        def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
            if self.active:
                if module.training or dropout!=0 or query.shape[-2]!=key.shape[-2]:raise AssertionError('expected eager eval full prefill')
                keys=m.repeat_kv(key,module.num_key_value_groups)
                scores=(torch.matmul(query,keys.transpose(2,3))*scaling).detach().cpu().numpy()
                sequence=scores.shape[-1]
                if sequence>1 and attention_mask is None:raise AssertionError("Missing causal mask for non-singleton sequence")
                if attention_mask is not None:
                    am=attention_mask[:,:,:,:sequence].detach().cpu().numpy()
                    expected=np.tril(np.ones((sequence,sequence),dtype=bool))
                    if not np.array_equal(np.broadcast_to(am==0,scores.shape),np.broadcast_to(expected,scores.shape)):
                        raise AssertionError('native mask differs from unpadded causal contract')
                rows,centered=causal_centered(scores)
                prefix=f'{module.layer_idx}.softmax'
                self.observe(prefix+'.raw_valid',np.concatenate([a.ravel() for a in rows]))
                self.observe(prefix+'.centered',np.concatenate([a.ravel() for a in centered]))
                self.observe(f'{module.layer_idx}.q_rope.native_output',query)
                self.observe(f'{module.layer_idx}.k_rope.native_output',key)
            return self.original(module,query,key,value,attention_mask,scaling,dropout,**kwargs)
        m.eager_attention_forward=attention;return self
    def begin(self):self.records={};self.missing=set();self.active=True
    def end(self):self.active=False;return dict(self.records),sorted(self.missing)
    def __exit__(self,*args):
        self.active=False
        self.module.eager_attention_forward=self.original
        for handle in self.handles:handle.remove()
        self.handles.clear()

def check_logits(actual,expected):
    finite=bool(np.isfinite(actual).all() and np.isfinite(expected).all())
    if actual.shape!=expected.shape or not finite:return {'passed':False,'finite':finite,'shape_matches':actual.shape==expected.shape}
    difference=actual-expected;maximum=float(np.max(np.abs(difference)))
    nrmse=float(np.sqrt(np.mean(difference*difference))/max(.01,float(np.sqrt(np.mean(expected*expected)))))
    top1=bool(np.array_equal(actual.argmax(-1),expected.argmax(-1)))
    return {'passed':maximum<=GATES['max_abs'] and nrmse<=GATES['nrmse'] and top1,'finite':True,'shape_matches':True,
        'max_abs_error':maximum,'nrmse':nrmse,'top1_agreement':top1,'all_final_logits_checked':int(actual.size)}

def decode_cases(initial,callback):
    teacher=list(initial);native_tokens=[];teacher_records=[]
    for step in range(4):
        native,oracle,record=callback(teacher,'teacher',step)
        choice=int(np.argmax(native));native_tokens.append(choice);teacher_records.append(record);teacher=teacher+[choice]
    free=list(initial);free_tokens=[];free_records=[]
    for step in range(4):
        native,oracle,record=callback(free,'free',step)
        choice=int(np.argmax(oracle));free_tokens.append(choice);free_records.append(record);free=free+[choice]
    return {'native_tokens':native_tokens,'oracle_free_tokens':free_tokens,'complete_sequence_agreement':native_tokens==free_tokens,
            'teacher_records':teacher_records,'free_records':free_records}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True);began=time.perf_counter()
    report={'scope':__doc__,'passed':False,'complete':False,'smoke':args.smoke,'cases':[],'decode':[],'errors':[],
        'candidate_selection_performed':False,'approximate_forward_evaluated':False,'encrypted_execution':False}
    primary=None;before={};source_paths={};model=None;assets={}
    try:
        write(args.output/'report.json',report)
        import torch
        from transformers import AutoTokenizer,Qwen3ForCausalLM
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        oracle=load(HERE/'reference.py','calibration_exact_oracle');observed=load(HERE/'observed_reference.py','calibration_observed_oracle')
        pins=json.loads((HERE/'model_pins.json').read_text());expected_assets=pins['asset_sha256']
        assets={name:digest(SNAPSHOT/name) for name in expected_assets}
        if assets!=expected_assets:raise ValueError('Pinned calibration model asset mismatch')
        protocol=json.loads((HERE/'protocol.json').read_text());corpus=json.loads((HERE/'corpus.json').read_text())
        if not protocol['execution_authorized_in_this_artifact']:raise ValueError('Missing reviewed protocol authorization')
        sources=[*sorted(HERE.glob('*.py')),*sorted(HERE.glob('*.json'))]
        import transformers
        package=Path(transformers.__file__).parent
        software=[*sorted((package/'models/qwen3').glob('*.py')),*[package/name for name in ('activations.py','cache_utils.py','modeling_attn_mask_utils.py','modeling_rope_utils.py','modeling_utils.py','modeling_outputs.py','configuration_utils.py','tokenization_utils_base.py','tokenization_utils_fast.py')]]
        source_paths={str(p.relative_to(HERE)):p for p in sources}
        source_paths.update({'software/transformers/'+str(p.relative_to(package)):p for p in software})
        before={name:digest(path) for name,path in source_paths.items()}
        with tarfile.open(args.output/'sources.tar.gz','w:gz') as archive:
            for name,path in source_paths.items():
                data=path.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
        with tarfile.open(args.output/'sources.tar.gz') as archive:
            archived={i.name:hashlib.sha256(archive.extractfile(i).read()).hexdigest() for i in archive.getmembers()}
        if archived!=before:raise ValueError('Source archive readback mismatch')
        report.update(source_hashes_before=before,source_archive_sha256=digest(args.output/'sources.tar.gz'),asset_sha256_before=assets,
            model_revision=pins['revision'],native_dtype='float32',exact_dtype='float64',cpu_affinity=sorted(os.sched_getaffinity(0)),
            versions={name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','safetensors','tokenizers')},semantic_gates=GATES,
            protocol_sha256=digest(HERE/'protocol.json'),corpus_sha256=digest(HERE/'corpus.json'))
        tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True)
        saved=json.loads((HERE/'tokenized_calibration_inputs.json').read_text());computed=[]
        for row in corpus['calibration_prompts']:
            ids=tokenizer(row['text'],add_special_tokens=False)['input_ids']
            if len(ids)<32:raise ValueError('Calibration paragraph too short')
            for S in (8,16,32):computed.append({'id':row['id'],'sequence_length':S,'raw_token_count':len(ids),'token_ids':ids[:S]})
        if computed!=saved:raise ValueError('Frozen calibration tokenizer output mismatch')
        for S in (8,16,32):
            if len({tuple(r['token_ids']) for r in saved if r['sequence_length']==S})!=96:raise ValueError('Calibration token collision')
        model,loading=Qwen3ForCausalLM.from_pretrained(SNAPSHOT,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError(str(loading))
        model.eval();config=model.config.to_dict();fields=('num_hidden_layers','hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','vocab_size')
        if tuple(config[k] for k in fields)!=(28,1024,3072,16,8,128,151936):raise ValueError('Architecture mismatch')
        if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('CPU-only calibration required')
        if model.lm_head.weight.data_ptr()!=model.model.embed_tokens.weight.data_ptr():raise ValueError('Tied output/embedding mismatch')
        weights={name:value.detach().numpy().astype(np.float64) for name,value in model.state_dict().items() if name!='lm_head.weight'}
        report.update(architecture={k:config[k] for k in fields},loading_info=loading)
        native_ranges={};exact_ranges={};union_ranges={};group_ranges={};checks=[];hook_checks=[]
        def native(ids):
            with torch.no_grad():
                return model(input_ids=torch.tensor([ids],dtype=torch.long),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),position_ids=torch.arange(len(ids))[None,:],use_cache=False).logits[:,-1].detach().numpy().astype(np.float64)
        with NativeObserver(model) as observer:
            def run(ids,label,group,check_hook=False):
                observer.active=False;bare=native(ids) if check_hook else None
                observer.begin();expected=native(ids);nrecords,_=observer.end()
                records={}
                def observe(name,value):merge(records,{name:stats(value)})
                embeddings=weights['model.embed_tokens.weight'][np.asarray([ids],dtype=np.int64)]
                actual=observed.forward(config,weights,embeddings,observe=observe)
                if check_hook:
                    plain=oracle.forward(config,weights,embeddings)
                    invariant={'id':label,'native_logits_bitwise_invariant':bool(np.array_equal(bare,expected)),'exact_logits_bitwise_invariant':bool(np.array_equal(actual,plain))}
                    hook_checks.append(invariant)
                    if not all(invariant[k] for k in ('native_logits_bitwise_invariant','exact_logits_bitwise_invariant')):raise AssertionError('Instrumentation changed logits')
                metric=check_logits(actual,expected);checks.append(metric)
                for target,source in ((native_ranges,nrecords),(exact_ranges,records),(union_ranges,nrecords),(union_ranges,records)):merge(target,source)
                group_record=group_ranges.setdefault(group,{'native':{},'exact':{}});merge(group_record['native'],nrecords);merge(group_record['exact'],records)
                record={'id':label,'group':group,'sequence_length':len(ids),'token_ids':list(ids),'context_sha256':hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest(),
                    'exact_oracle_vs_native':metric,'native_top1':int(expected.argmax()),'oracle_top1':int(actual.argmax()),'native_sites':len(nrecords),'exact_sites':len(records)}
                print(json.dumps({'id':label,'length':len(ids),'semantic_pass':metric['passed'],'max_abs':metric.get('max_abs_error')}),flush=True)
                return expected,actual,record
            prefills=saved
            if args.smoke:prefills=[next(r for r in saved if r['sequence_length']==S) for S in (8,16,32)]
            shapes=set()
            for row in prefills:
                S=row['sequence_length'];_,_,record=run(row['token_ids'],row['id']+f'_S{S}',f'prefill_S{S}',S not in shapes);shapes.add(S);report['cases'].append(record)
            decodes=corpus['decode_prompt_ids'][:1] if args.smoke else corpus['decode_prompt_ids']
            for index,name in enumerate(decodes):
                row=next(r for r in saved if r['id']==name and r['sequence_length']==16)
                def callback(ids,kind,step):return run(ids,name+f'_{kind}_{step}',f'decode_{kind}',check_hook=index==0 and kind=='teacher' and step==0)
                report['decode'].append(dict(id=name,**decode_cases(row['token_ids'],callback)))
        required={f'{layer}.{part}.inverse_sqrt' for layer in range(28) for part in ('input','post','q','k')}|{f'{layer}.silu' for layer in range(28)}|{f'{layer}.softmax.centered' for layer in range(28)}|{'final.inverse_sqrt'}
        coverage=required<=native_ranges.keys() and required<=exact_ranges.keys()
        finite=all(not r['nonfinite'] for r in union_ranges.values());positive=all(union_ranges[n]['minimum']>0 for n in required if n.endswith('inverse_sqrt'))
        report.update(observed_native_ranges=native_ranges,observed_exact_ranges=exact_ranges,calibration_union_ranges=union_ranges,group_ranges=group_ranges,
            required_input_site_count=len(required),required_input_sites_covered=coverage,finite_trajectories=finite,positive_inverse_sqrt_inputs=positive,
            hook_invariance_checks=hook_checks,all_exact_semantic_checks_passed=all(c['passed'] for c in checks),total_forward_contexts=len(checks),all_final_logits_checked_per_context=151936,
            teacher_forced_top1_agreements=sum(r['exact_oracle_vs_native']['top1_agreement'] for d in report['decode'] for r in d['teacher_records']),teacher_forced_decisions=4*len(report['decode']),
            free_running_complete_sequence_agreements=sum(d['complete_sequence_agreement'] for d in report['decode']),free_running_sequences=len(report['decode']),
            no_jax_imported=not any(n.split('.')[0] in ('jax','jaxlib','jaxite_word') for n in sys.modules),
            complete=not args.smoke and len(report['cases'])==288 and len(report['decode'])==16,audit_completed=True,
            limitations=['Authored calibration corpus only; no held-out validation ranges/logits/outputs are opened. Decode seeds overlap calibration prefills.',
                'Native float32 and exact float64 trajectories do not certify approximate/CKKS/refresh inputs or unseen prompts.',
                'No polynomial selection, approximate forward, hidden benchmark calibration, language-task accuracy, HE or TPU execution.',
                'Final norm observes last position only; all needed positions are observed for earlier 28-layer norms/SiLU/attention.',
                'Selected source archive and package versions are not a complete OS/binary reproducibility image.'])
        report['passed']=report['all_exact_semantic_checks_passed'] and coverage and finite and positive and report['no_jax_imported']
    except BaseException as exc:
        primary=exc;report['errors'].append({'type':type(exc).__name__,'message':str(exc)});report['passed']=False
        if not isinstance(exc,(KeyboardInterrupt,SystemExit)):traceback.print_exc()
    finally:
        if model is not None:del model
        gc.collect()
        try:
            after={name:digest(path) for name,path in source_paths.items()};report['source_hashes_after']=after
            report['source_files_unchanged']=bool(before) and before==after
            report['asset_sha256_after']={name:digest(SNAPSHOT/name) for name in assets}
            report['assets_unchanged']=bool(assets) and assets==report['asset_sha256_after']
            if not report['source_files_unchanged'] or not report['assets_unchanged']:report['passed']=False
        except BaseException as exc:
            report['passed']=False;report['errors'].append({'phase':'final_provenance','type':type(exc).__name__,'message':str(exc)})
            if primary is None:primary=exc
            else:primary.add_note('Final provenance failed: '+repr(exc))
        report['elapsed_s']=time.perf_counter()-began
        try:write(args.output/'report.json',report)
        except BaseException as exc:
            if primary is not None:primary.add_note('Final evidence publication failed: '+repr(exc))
            else:raise
    if primary is not None:raise primary
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
