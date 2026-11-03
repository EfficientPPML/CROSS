"""Training-only native domain collection. Labels are never read or used."""
from pathlib import Path
import hashlib,io,json,math,os,sys,tarfile,time,importlib.metadata
import numpy as np
from audit import Observer,sha,write,MODEL,MANIFEST,REPO

HERE=Path(__file__).resolve().parent

def accumulate(result, rows):
    for name,row in rows.items():
        if name not in result:
            result[name]=dict(row,examples=0,examples_with_violations=0)
            for key in ('count','below','above','nonfinite'):result[name][key]=0
            result[name]['minimum']=result[name]['maximum']=None
        out=result[name];assert out['domain']==row['domain']
        for key in ('count','below','above','nonfinite'):out[key]+=row[key]
        out['examples']+=1
        out['examples_with_violations']+=bool(row['below'] or row['above'] or row['nonfinite'])
        if row['minimum'] is not None:
            out['minimum']=row['minimum'] if out['minimum'] is None else min(out['minimum'],row['minimum'])
            out['maximum']=row['maximum'] if out['maximum'] is None else max(out['maximum'],row['maximum'])

def domain(lo,hi,positive=False):
    if not math.isfinite(lo) or not math.isfinite(hi) or lo>hi:raise ValueError('invalid native range')
    if positive:
        if lo<=0:raise ValueError('nonpositive variance')
        return [.8*lo,1.2*hi]
    return [math.floor(lo-.1*(hi-lo)),math.ceil(hi+.1*(hi-lo))]

def main():
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False);start=time.monotonic();before={};obs=None
    report=dict(scope=__doc__,passed=False,completed_examples=0,training_labels_loaded=False)
    try:
        write(out/'report.json',report)
        import torch,pyarrow.parquet as pq
        from transformers import AutoTokenizer,BertForSequenceClassification
        torch.set_num_threads(8);torch.set_num_interop_threads(1)
        data=json.loads((HERE/'dataset_manifest.json').read_text());protocol=json.loads((HERE/'selection_protocol.json').read_text())
        path=Path('/home/jianming_gatech/cross_validation_assets/mrpc')/data['revision']/data['file']
        assert sha(path)==data['sha256']
        # Deliberately exclude the label column from the Parquet reader.
        rows=pq.read_table(path,columns=['sentence1','sentence2','idx']).to_pylist()
        assert len(rows)==3668 and len({r['idx'] for r in rows})==3668 and all('label' not in r for r in rows)
        previous=json.loads(Path('/tmp/bert-mrpc408-domain-audit-20261103/full408-v1/report.json').read_text())
        assets=previous['model_assets'];assert {n:sha(MODEL/n) for n in assets}==assets
        selected=[Path(__file__),HERE/'audit.py',HERE/'test_audit.py',HERE/'test_calibrate.py',HERE/'dataset_manifest.json',HERE/'selection_protocol.json']
        selected += [Path(n) for n in previous['source_hashes_before'] if Path(n).name not in ('audit.py','test_audit.py','dataset_manifest.json')]
        before={str(p.resolve()):sha(p) for p in selected}
        with tarfile.open(out/'sources.tar.gz','w:gz') as archive:
            for name in before:
                raw=Path(name).read_bytes();info=tarfile.TarInfo(name.lstrip('/'));info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
        with tarfile.open(out/'sources.tar.gz') as archive:
            assert len(archive.getmembers())==len(before)
            assert all(hashlib.sha256(archive.extractfile(m).read()).hexdigest()==before['/'+m.name] for m in archive.getmembers())
        manifest=json.loads(MANIFEST.read_text())['operator_manifest']
        model,loading=BertForSequenceClassification.from_pretrained(MODEL,local_files_only=True,use_safetensors=True,
            torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        if loading['missing_keys'] or loading.get('mismatched_keys') or loading.get('error_msgs') or set(loading['unexpected_keys'])-{'bert.embeddings.position_ids'}:raise ValueError('incomplete model')
        model.eval();tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        obs=Observer(model,manifest);aggregate={};tokenhash=hashlib.sha256();min_tokens=128;max_tokens=0
        report.update(dataset=data,selection_protocol=protocol,source_hashes_before=before,
            source_archive_sha256=sha(out/'sources.tar.gz'),source_archive_verified=True,model_assets=assets,
            cpu_affinity=sorted(os.sched_getaffinity(0)),native_dtype='float32',threads=8,
            versions={n:importlib.metadata.version(n) for n in ('torch','numpy','transformers','tokenizers','safetensors','pyarrow')})
        for i,row in enumerate(rows):
            tokens=tokenizer(row['sentence1'],row['sentence2'],max_length=128,padding='max_length',truncation=True,return_tensors='pt')
            encoded=dict(target_idx=row['idx'],**{n:v.tolist() for n,v in sorted(tokens.items())})
            tokenhash.update(json.dumps(encoded,sort_keys=True,separators=(',',':')).encode()+b'\n')
            obs.rows={};obs.mask=tokens['attention_mask'][0].numpy().astype(bool)
            with torch.no_grad():model.bert(**tokens)
            assert len(obs.rows)==121 and not obs.pending
            accumulate(aggregate,obs.rows);min_tokens=min(min_tokens,int(obs.mask.sum()));max_tokens=max(max_tokens,int(obs.mask.sum()))
            report['completed_examples']=i+1
            if i%100==0 or i+1==len(rows):
                progress=dict(completed=i+1,elapsed_s=time.monotonic()-start)
                write(out/'progress.json',progress);print(json.dumps(progress),flush=True)
        proposed={}
        for name,row in aggregate.items():
            if name.endswith('.padded_queries') or '.softmax.base_exp.' in name:continue
            label=name.removesuffix('.valid_queries')
            proposed[label]=dict(observed_range=[row['minimum'],row['maximum']],
                domain=domain(row['minimum'],row['maximum'],positive='.inverse_sqrt' in name),
                observations=row['count'],examples=row['examples'])
        report.update(aggregate_native_ranges=aggregate,proposed_domains=proposed,tokenization_sha256=tokenhash.hexdigest(),
            valid_token_range=[min_tokens,max_tokens],passed=True,
            no_jax_imported=not any(n=='jax' or n.startswith('jax.') for n in sys.modules),
            candidate_coefficients_selected=False,approximate_forward_evaluated=False,encrypted_execution=False,
            limitation='Prospective domains derive only from training native ranges and the fixed historical rule; neither validation coverage nor propagated approximation/CKKS error is established.')
        assert report['no_jax_imported']
    except BaseException as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}');raise
    finally:
        if obs is not None:obs.close()
        after={n:sha(n) for n in before};report.update(source_hashes_after=after,source_files_unchanged=bool(before) and before==after,elapsed_s=time.monotonic()-start)
        report['passed']=bool(report['passed'] and before==after);write(out/'report.json',report)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
