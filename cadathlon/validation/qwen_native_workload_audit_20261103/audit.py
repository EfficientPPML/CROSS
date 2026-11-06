"""Fixed public Qwen workload: native domains and independent exact semantics.

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
    def __init__(self,model,manifest):
        self.model,self.manifest=model,manifest;self.handles=[];self.records={};self.missing=set();self.active=False
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
                self.observe(label,variance,self.manifest['nonlinear'][label]['domain'])
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
                if self.active:self.observe(label,args[0],self.manifest['nonlinear'][label]['domain'])
            self.handles.append(layer.mlp.act_fn.register_forward_pre_hook(silu))
        self.handles.append(self.model.model.norm.register_forward_pre_hook(norm('final.inverse_sqrt',last=True)))
        def attention(module,query,key,value,attention_mask,scaling,dropout=0.,**kwargs):
            if self.active:
                if module.training or dropout!=0 or query.shape[-2]!=key.shape[-2]:raise AssertionError('expected eager eval full prefill')
                keys=m.repeat_kv(key,module.num_key_value_groups)
                scores=(torch.matmul(query,keys.transpose(2,3))*scaling).detach().cpu().numpy()
                sequence=scores.shape[-1]
                if attention_mask is not None:
                    am=attention_mask[:,:,:,:sequence].detach().cpu().numpy()
                    expected=np.tril(np.ones((sequence,sequence),dtype=bool))
                    if not np.array_equal(np.broadcast_to(am==0,scores.shape),np.broadcast_to(expected,scores.shape)):
                        raise AssertionError('native mask differs from unpadded causal contract')
                rows,centered=causal_centered(scores);rule=self.manifest['softmax'][str(module.layer_idx)]
                prefix=f'{module.layer_idx}.softmax'
                self.observe(prefix+'.raw_valid',np.concatenate([a.ravel() for a in rows]))
                self.observe(prefix+'.centered',np.concatenate([a.ravel() for a in centered]),rule['centered_domain'])
                self.observe(prefix+'.range_reduced_exp_input',np.concatenate([a.ravel() for a in centered])/2**rule['squarings'],(-.5,.5))
                self.observe(f'{module.layer_idx}.q_rope.native_output',query)
                self.observe(f'{module.layer_idx}.k_rope.native_output',key)
                for count in range(1,sequence+1):
                    if str(count) not in rule['reciprocal_iterations_by_keys']:self.missing.add((module.layer_idx,count))
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

def tokenize_fixtures(tokenizer,source):
    records=[];seen=defaultdict(set)
    for fixture in source['fixtures']:
        full=tokenizer(fixture['text'],add_special_tokens=False)['input_ids'];S=fixture['sequence_length']
        if len(full)<max(32,S):raise ValueError('Raw fixture is too short: '+fixture['id'])
        ids=tuple(map(int,full[:S]));bucket=(fixture['kind'],S)
        if len(ids)!=S or ids in seen[bucket]:raise ValueError('Duplicate or incorrect length token fixture')
        seen[bucket].add(ids);records.append(dict(fixture,raw_token_count=len(full),token_ids=list(ids)))
    assert {k:len(v) for k,v in seen.items()}=={('prefill',8):16,('prefill',16):16,('prefill',32):16,('decode',16):8}
    return records

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
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    args.output.mkdir(exist_ok=False,parents=True);began=time.perf_counter();report={'scope':__doc__,'passed':False,'complete':False,'smoke':args.smoke,'cases':[],'decode':[],'errors':[]}
    primary=None;before={};source_paths={};model=None
    try:
        write(args.output/'report.json',report)
        import torch
        from transformers import AutoTokenizer,Qwen3ForCausalLM
        from transformers.models.qwen3 import modeling_qwen3
        torch.set_num_threads(4);torch.set_num_interop_threads(1)
        oracle=load(HERE/'reference.py','frozen_qwen_reference')
        # previous_audit imports support in main only? Its top-level support
        # import is deliberately avoided below by reading EXPECTED_ASSETS via AST.
        assets={name:digest(SNAPSHOT/name) for name in EXPECTED_ASSETS}
        if assets!=EXPECTED_ASSETS:raise ValueError('Pinned Qwen asset mismatch')
        operators=json.loads((HERE/'operators.json').read_text());manifest=operators['manifest']
        sources=[*sorted(HERE.glob('*.py')),*sorted(HERE.glob('*.json')),HERE/'problem.md']
        import transformers
        package=Path(transformers.__file__).parent
        software=[*sorted((package/'models/qwen3').glob('*.py')),*[package/name for name in ('activations.py','cache_utils.py','modeling_attn_mask_utils.py','modeling_rope_utils.py','modeling_utils.py','modeling_outputs.py','configuration_utils.py','tokenization_utils_base.py','tokenization_utils_fast.py')]]
        source_paths={str(p.relative_to(HERE)):p for p in sources}
        source_paths.update({'software/transformers/'+str(p.relative_to(package)):p for p in software})
        before={name:digest(path) for name,path in source_paths.items()}
        with tarfile.open(args.output/'sources.tar.gz','w:gz') as archive:
            for name,path in source_paths.items():
                data=path.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
        report.update(source_hashes_before=before,source_archive_sha256=digest(args.output/'sources.tar.gz'),
            asset_sha256=assets,model_revision='c1899de289a04d12100db370d81485cdf75e47ca',native_dtype='float32',cpu_affinity=sorted(os.sched_getaffinity(0)),
            versions={name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','safetensors','tokenizers')},semantic_gates=GATES,
            contract_sha256=digest(HERE/'operators.json'),fixture_source_sha256=digest(HERE/'fixtures.json'))
        tokenizer=AutoTokenizer.from_pretrained(SNAPSHOT,local_files_only=True)
        fixtures=tokenize_fixtures(tokenizer,json.loads((HERE/'fixtures.json').read_text()))
        write(args.output/'tokenized_fixtures.json',fixtures)
        report['tokenized_fixtures_sha256']=digest(args.output/'tokenized_fixtures.json')
        model,loading=Qwen3ForCausalLM.from_pretrained(SNAPSHOT,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise ValueError(str(loading))
        model.eval();config=model.config.to_dict()
        if tuple(config[k] for k in ('num_hidden_layers','hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','vocab_size'))!=(28,1024,3072,16,8,128,151936):raise ValueError('Architecture mismatch')
        if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('Native audit must remain CPU only')
        if model.lm_head.weight.data_ptr()!=model.model.embed_tokens.weight.data_ptr():raise ValueError('Expected tied embeddings/output')
        weights={name:value.detach().numpy().astype(np.float64) for name,value in model.state_dict().items() if name!='lm_head.weight'}
        report.update(architecture={k:config[k] for k in ('num_hidden_layers','hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','vocab_size')},loading_info=loading)
        aggregate={};missing=set();subset_aggregate={};checks=[];hook_checks=[]
        def native(ids):
            with torch.no_grad():
                return model(input_ids=torch.tensor([ids],dtype=torch.long),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),position_ids=torch.arange(len(ids))[None,:],use_cache=False).logits[:,-1].detach().numpy().astype(np.float64)
        with NativeObserver(model,manifest) as observer:
            def run(ids,label,group,check_hook=False):
                observer.active=False
                bare=native(ids) if check_hook else None
                observer.begin();expected=native(ids);observations,gaps=observer.end()
                if check_hook:
                    same=bool(np.array_equal(bare,expected));hook_checks.append({'case':label,'bitwise_equal_all_logits':same})
                    if not same:raise AssertionError('Instrumentation changed native logits')
                embeddings=weights['model.embed_tokens.weight'][np.asarray([ids],dtype=np.int64)]
                actual=oracle.forward(config,weights,embeddings);metric=check_logits(actual,expected);checks.append(metric)
                merge(aggregate,observations);merge(subset_aggregate.setdefault(group,{}),observations);missing.update(gaps)
                violations={k:v for k,v in observations.items() if v['below'] or v['above'] or v['nonfinite']}
                record={'id':label,'group':group,'sequence_length':len(ids),'token_ids':ids,'context_sha256':hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest(),
                    'exact_oracle_vs_native':metric,'native_top1':int(expected.argmax()),'oracle_top1':int(actual.argmax()),'domain_violations':violations,
                    'missing_softmax_key_counts':sorted({k for layer,k in gaps})}
                print(json.dumps({'id':label,'length':len(ids),'semantic_pass':metric['passed'],'domain_failure_sites':len(violations),'max_abs':metric.get('max_abs_error')}),flush=True)
                return expected,actual,record
            prefills=[f for f in fixtures if f['kind']=='prefill']
            if args.smoke:prefills=[next(f for f in prefills if f['sequence_length']==S) for S in (8,16,32)]
            seen_shapes=set()
            for fixture in prefills:
                S=fixture['sequence_length'];_,_,record=run(fixture['token_ids'],fixture['id'],f'prefill_S{S}',S not in seen_shapes);seen_shapes.add(S);report['cases'].append(record)
            decodes=[f for f in fixtures if f['kind']=='decode']
            if args.smoke:decodes=decodes[:1]
            for fixture in decodes:
                def callback(ids,kind,step):return run(ids,fixture['id']+f'_{kind}_{step}',f'decode_{kind}',check_hook=(fixture['id']=='Q6_00' and kind=='teacher' and step==0))
                report['decode'].append(dict(id=fixture['id'],**decode_cases(fixture['token_ids'],callback)))
        report.update(observed_native_ranges=aggregate,group_ranges=subset_aggregate,
            missing_softmax_contracts=[{'layer':layer,'valid_keys':count} for layer,count in sorted(missing)],
            fixed_domain_coverage_passed=not any(v['below'] or v['above'] or v['nonfinite'] for v in aggregate.values()) and not missing,
            hook_invariance_checks=hook_checks,all_exact_semantic_checks_passed=all(c['passed'] for c in checks),
            total_forward_contexts=len(checks),all_final_logits_checked_per_context=151936,
            teacher_forced_top1_agreements=sum(r['exact_oracle_vs_native']['top1_agreement'] for d in report['decode'] for r in d['teacher_records']),
            teacher_forced_decisions=4*len(report['decode']),free_running_complete_sequence_agreements=sum(d['complete_sequence_agreement'] for d in report['decode']),
            free_running_sequences=len(report['decode']),no_jax_imported=not any(n.split('.')[0] in ('jax','jaxlib','jaxite_word') for n in sys.modules),
            complete=not args.smoke and len(report['cases'])==48 and len(report['decode'])==8,
            audit_completed=True,encrypted_execution=False,approximate_forward_evaluated=False,
            limitations=['Public fixtures are not hidden organizer prompts; no benchmark baseline/calibration or language-task accuracy is established.',
                'Native float32 versus exact float64 NumPy checks are semantic comparisons, not approximate-circuit or HE agreement gates.',
                'Native input ranges do not bound approximate/CKKS trajectories. All observed fixed-domain failures and missing key-count contracts remain unchanged.',
                'No padding/cache/chat template: full-prefill recomputation, positions0..S-1; four decisions use sequence lengths16,17,18,19.',
                'Teacher contexts use native preceding tokens; oracle free-running contexts use the exact NumPy oracle preceding tokens. Native model is also evaluated on each free-running context for same-input semantic checks.',
                'Final RMSNorm contract audits only the last output position, which is the required vocabulary-logit input. Earlier-layer RMS/QK/SiLU contracts cover every needed token position.',
                'Source archive covers audit/reference/contract and selected relevant Transformers Python modules; software versions identify dependencies, not a complete OS/binary reproducibility image.'])
        report['passed']=report['audit_completed'] and report['all_exact_semantic_checks_passed'] and report['no_jax_imported']
    except BaseException as exc:
        primary=exc;report['errors'].append({'type':type(exc).__name__,'message':str(exc)});report['passed']=False
        if not isinstance(exc,(KeyboardInterrupt,SystemExit)):traceback.print_exc()
    finally:
        if model is not None:del model
        gc.collect()
        try:
            after={name:digest(path) for name,path in source_paths.items()}
            report['source_hashes_after']=after;report['source_files_unchanged']=bool(before) and before==after
            if not report['source_files_unchanged']:report['passed']=False
        except BaseException as exc:
            report['passed']=False;report['errors'].append({'phase':'final_source_check','type':type(exc).__name__,'message':str(exc)})
            if primary is None:primary=exc
            else:primary.add_note('Final source check failed: '+repr(exc))
        report['elapsed_s']=time.perf_counter()-began
        try:write(args.output/'report.json',report)
        except BaseException as exc:
            if primary is not None:primary.add_note('Final evidence publication failed: '+repr(exc))
            else:raise
    if primary is not None:raise primary
    return 0 if report['passed'] else 1

# Parse the frozen previous audit's literal asset map without importing its
# runtime support module (which is outside this native-only closure).
import ast
for node in ast.parse((HERE/'previous_audit.py').read_text()).body:
    if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='EXPECTED_ASSETS' for t in node.targets):EXPECTED_ASSETS=ast.literal_eval(node.value);break
else:raise ValueError('Pinned asset map not found')

if __name__=='__main__':raise SystemExit(main())
