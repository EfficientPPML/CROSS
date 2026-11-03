"""Native-only fixed-contract MRPC validation audit; no approximate or HE forward."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, io, json, math, os, random, sys, tarfile, time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = Path('/home/jianming_gatech/CROSS')
HARNESS = Path('/home/jianming_gatech/BERT_validation/harness')
MODEL = Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c')
MANIFEST = REPO/'cadathlon/validation/compose_pretrained_bert_arithmetic.json'

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def write(path, value):
    p = Path(path); temporary=p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temporary.replace(p)

def summarize(values, domain):
    values=np.asarray(values); finite=np.isfinite(values); good=values[finite]
    lo,hi=domain
    return dict(domain=list(domain),count=int(values.size),minimum=float(good.min()) if good.size else None,
        maximum=float(good.max()) if good.size else None,below=int(np.sum(values<lo)),
        above=int(np.sum(values>hi)),nonfinite=int(np.sum(~finite)))

def merge(rows):
    result={}
    for row in rows:
        for name,value in row.items():
            if name not in result:
                result[name]=dict(value,examples=0,examples_with_violations=0)
                for k in ('count','below','above','nonfinite'): result[name][k]=0
                result[name]['minimum']=result[name]['maximum']=None
            out=result[name]
            assert out['domain']==value['domain']
            for k in ('count','below','above','nonfinite'): out[k]+=value[k]
            out['examples']+=1
            out['examples_with_violations']+=bool(value['below'] or value['above'] or value['nonfinite'])
            if value['minimum'] is not None:
                out['minimum']=value['minimum'] if out['minimum'] is None else min(out['minimum'],value['minimum'])
                out['maximum']=value['maximum'] if out['maximum'] is None else max(out['maximum'],value['maximum'])
    return result

class Observer:
    def __init__(self, model, manifest):
        self.model=model; self.manifest=manifest; self.handles=[]; self.pending={}; self.rows={}
        self.mask=None
        for i,layer in enumerate(model.bert.encoder.layer):
            for phase,module in [('attention',layer.attention.output.LayerNorm),('mlp',layer.output.LayerNorm)]:
                self.handles.append(module.register_forward_pre_hook(self.norm_hook(i,phase)))
            self.handles.append(layer.intermediate.dense.register_forward_hook(self.gelu_hook(i)))
            self.handles.append(layer.attention.self.query.register_forward_hook(self.projection_hook(i,'q')))
            self.handles.append(layer.attention.self.key.register_forward_hook(self.projection_hook(i,'k')))
            self.handles.append(layer.attention.self.register_forward_hook(self.scores_hook(i)))
        self.handles.append(model.bert.pooler.dense.register_forward_hook(self.pooler_hook))

    def observe_tokens(self,label,values,domain):
        self.rows[label+'.valid_queries']=summarize(values[self.mask],domain)
        self.rows[label+'.padded_queries']=summarize(values[~self.mask],domain)

    def norm_hook(self,layer,phase):
        def hook(module,args):
            # Public native activations converted to float64 only for stable audit moments.
            x=args[0][0].detach().numpy().astype(np.float64)
            centered=x-x.mean(axis=-1,keepdims=True)
            variance=np.mean(centered*centered,axis=-1)+module.eps
            self.observe_tokens(f'{layer}.{phase}.inverse_sqrt',variance,self.manifest['norms'][f'{layer}.{phase}']['domain'])
        return hook

    def gelu_hook(self,layer):
        def hook(module,args,output):
            self.observe_tokens(f'{layer}.gelu',output[0].detach().numpy(),self.manifest['gelu'][str(layer)]['domain'])
        return hook

    def projection_hook(self,layer,kind):
        def hook(module,args,output): self.pending[(layer,kind)]=output
        return hook

    def scores_hook(self,layer):
        def hook(module,args,output):
            import torch
            q=self.pending.pop((layer,'q')); k=self.pending.pop((layer,'k'))
            q=q.view(1,128,module.num_attention_heads,module.attention_head_size).transpose(1,2)
            k=k.view(1,128,module.num_attention_heads,module.attention_head_size).transpose(1,2)
            # Same float32 matmul/scaling as eager BERT; mask only public key positions.
            scores=(torch.matmul(q,k.transpose(-1,-2))/math.sqrt(module.attention_head_size))[0].detach().numpy()
            valid=scores[:,:,self.mask].astype(np.float64)
            centered=valid-valid.mean(axis=-1,keepdims=True)
            rule=self.manifest['softmax'][str(layer)]
            for scope,mask in [('valid_queries',self.mask),('padded_queries',~self.mask)]:
                values=centered[:,mask,:]
                self.rows[f'{layer}.softmax.centered_scores.{scope}']=summarize(values,rule['centered_domain'])
                self.rows[f'{layer}.softmax.base_exp.{scope}']=summarize(values/(2**rule['squarings']),[-.5,.5])
        return hook

    def pooler_hook(self,module,args,output):
        self.rows['pooler.tanh']=summarize(output.detach().numpy(),self.manifest['tanh']['domain'])

    def close(self):
        for handle in self.handles: handle.remove()
        self.handles.clear(); self.pending.clear()

def subset_indices(total,size,base_seed):
    # Exactly the upstream run_submission -> generate_input seed route.
    seed=int(np.random.default_rng(base_seed).integers(0,0x7fffffff))
    return (list(range(total)) if size==total else sorted(random.Random(seed).sample(range(total),size))),seed

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=408); parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();report=dict(scope=__doc__,passed=False,completed_examples=0)
    observer=None; before={}
    try:
        write(args.output/'report.json',report)
        import torch, pyarrow.parquet as pq
        from transformers import AutoTokenizer, BertForSequenceClassification
        import transformers.models.bert.modeling_bert as modeling
        import transformers.models.bert.tokenization_bert as tokenization
        import transformers.models.bert.tokenization_bert_fast as tokenization_fast
        import transformers.tokenization_utils_base as tokenization_base
        import transformers.activations as activations
        torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
        dataset=json.loads((HERE/'dataset_manifest.json').read_text())
        path=Path('/home/jianming_gatech/cross_validation_assets/mrpc')/dataset['revision']/dataset['file']
        if sha(path)!=dataset['sha256']: raise ValueError('dataset bytes differ')
        table=pq.read_table(path).to_pylist()
        if len(table)!=408 or len({row['idx'] for row in table})!=408: raise ValueError('unexpected split size/identities')
        if any(type(row['label']) is not int or row['label'] not in (0,1) for row in table):raise ValueError('invalid labels')
        manifest=json.loads(MANIFEST.read_text())['operator_manifest']
        assets=json.loads((REPO/'cadathlon/validation/pretrained_bert_domain_audit.json').read_text())['asset_sha256']
        if {name:sha(MODEL/name) for name in assets}!=assets: raise ValueError('pinned assets differ')
        source_paths=[Path(__file__),HERE/'test_audit.py',HERE/'dataset_manifest.json',MANIFEST,
            REPO/'cadathlon/validation/plan_pretrained_bert_operators.json',
            REPO/'cadathlon/validation/compose_pretrained_bert_arithmetic.py',
            REPO/'cadathlon/validation/check_pretrained_bert_domains.py',
            REPO/'cadathlon/agent_pset/p2_bert/src/stages.py',
            REPO/'cadathlon/agent_pset/p2_bert/src/benchmark.py']
        source_paths += [HARNESS/name for name in ('generate_input.py','generate_dataset.py','params.py','run_submission.py','utils.py','mrpc/mrpc.py','mrpc/model.py','mrpc/test.py')]
        source_paths += [Path(m.__file__) for m in (modeling,tokenization,tokenization_fast,tokenization_base,activations)]
        source_paths += [MODEL/name for name in assets if name!='model.safetensors']
        before={str(p):sha(p) for p in source_paths}
        with tarfile.open(args.output/'sources.tar.gz','w:gz') as archive:
            for p in source_paths:
                data=p.read_bytes();info=tarfile.TarInfo(str(p).lstrip('/'));info.size=len(data);archive.addfile(info,io.BytesIO(data))
        with tarfile.open(args.output/'sources.tar.gz') as archive:
            members=archive.getmembers()
            assert len(members)==len(before)
            assert all(hashlib.sha256(archive.extractfile(m).read()).hexdigest()==before['/'+m.name] for m in members)
        report.update(dataset=dataset,model_revision=MODEL.name,model_assets=assets,contracts=manifest,
            native_dtype='torch.float32 (upstream default)',sequence_length=128,padding='max_length',truncation=True,
            attention_implementation='eager',batch_size=1,threads=args.threads,cpu_affinity=sorted(os.sched_getaffinity(0)),
            source_hashes_before=before,source_archive_sha256=sha(args.output/'sources.tar.gz'),source_archive_verified=True,
            versions={name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','tokenizers','safetensors','pyarrow')},
            limitations=['Native activation-domain audit, not approximate inference, encrypted inference, or runtime calibration.',
              'Original full-model domains came from three public pairs and are held fixed here; failures are not retuned.',
              'Valid-query observations match public padding-compacted arithmetic plans; padded-query observations are reported separately.',
              'Internal approximate Newton/normalization trajectories and CKKS error are not evaluated.',
              '100/200 sets are explicit reproducible base-seed42 harness subsets, not universal subset guarantees.',
              'No plaintext hidden-state arrays, token arrays, sentence text, logits, or weights are persisted in the report.'])
        model,loading=BertForSequenceClassification.from_pretrained(MODEL,local_files_only=True,use_safetensors=True,
            torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        if loading['missing_keys'] or loading.get('mismatched_keys') or loading.get('error_msgs') or set(loading['unexpected_keys'])-{'bert.embeddings.position_ids'}:
            raise ValueError(f'incomplete pinned model: {loading}')
        model.eval();tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        if model.config.hidden_size!=768 or model.config.num_hidden_layers!=12 or model.num_labels!=2:raise ValueError('architecture differs')
        observer=Observer(model,manifest);records=[];token_digest=hashlib.sha256()
        for position,row in enumerate(table[:args.limit]):
            tokens=tokenizer(row['sentence1'],row['sentence2'],max_length=128,padding='max_length',truncation=True,return_tensors='pt')
            token_record={name:value.tolist() for name,value in sorted(tokens.items())}
            token_bytes=json.dumps(dict(target_idx=row['idx'],**token_record),sort_keys=True,separators=(',',':')).encode()+b'\n'
            token_digest.update(token_bytes)
            observer.rows={};observer.mask=tokens['attention_mask'][0].numpy().astype(bool)
            with torch.no_grad():logits=model(**tokens).logits[0]
            if not bool(torch.isfinite(logits).all()):raise ValueError('nonfinite native logits')
            prediction=int(logits[1]>=logits[0]); expected_count=12*10+1
            if len(observer.rows)!=expected_count or observer.pending:raise ValueError('incomplete observer coverage')
            records.append(dict(position=position,target_idx=row['idx'],label=row['label'],prediction=prediction,
                correct=prediction==row['label'],valid_tokens=int(observer.mask.sum()),tokenization_sha256=hashlib.sha256(token_bytes).hexdigest(),domains=observer.rows))
            report['completed_examples']=position+1
            if position%25==0 or position+1==args.limit:
                print(json.dumps(dict(completed=position+1,elapsed_s=time.monotonic()-started,correct=sum(r['correct'] for r in records))),flush=True)
                write(args.output/'progress.json',dict(completed=position+1,elapsed_s=time.monotonic()-started))
        subsets={}
        sizes=(100,200,408) if args.limit==408 else (args.limit,)
        for size in sizes:
            indices,seed=subset_indices(args.limit,size,42)
            selected=[records[i] for i in indices];domains=merge([r['domains'] for r in selected])
            violations={name:value for name,value in domains.items() if value['below'] or value['above'] or value['nonfinite']}
            valid_failures={k:v for k,v in violations.items() if '.padded_queries' not in k}
            subsets[str(size)]=dict(size=size,base_seed=42,derived_input_seed=seed,positions=indices,
                target_ids=[r['target_idx'] for r in selected],correct=sum(r['correct'] for r in selected),
                accuracy=sum(r['correct'] for r in selected)/size,domains=domains,violations=violations,
                valid_query_contracts_cover_native_inputs=not valid_failures,
                valid_query_failing_operators=len(valid_failures),padded_query_failing_operators=len(violations)-len(valid_failures),
                examples_with_valid_query_domain_violation=sum(any(v['below'] or v['above'] or v['nonfinite'] for k,v in r['domains'].items() if '.padded_queries' not in k) for r in selected))
        report.update(subsets=subsets,tokenization_sha256=token_digest.hexdigest(),
            input_token_counts=dict(minimum=min(r['valid_tokens'] for r in records),maximum=max(r['valid_tokens'] for r in records)),
            per_example=[{k:v for k,v in r.items() if k!='domains'} for r in records],
            audited_all_official_validation_examples=args.limit==408,
            no_jax_imported=not any(k=='jax' or k.startswith('jax.') for k in sys.modules),passed=True)
        if not report['no_jax_imported']: raise ValueError('unexpected JAX import')
    except BaseException as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}');raise
    finally:
        if observer is not None:observer.close()
        after={name:sha(name) for name in before};report.update(source_hashes_after=after,
            source_files_unchanged=bool(before) and before==after,elapsed_s=time.monotonic()-started)
        report['passed']=bool(report['passed'] and before==after)
        write(args.output/'report.json',report)
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
