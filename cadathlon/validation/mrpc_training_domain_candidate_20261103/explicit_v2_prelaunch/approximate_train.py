"""Strict fixed-v2 approximate training trajectories; labels are never loaded."""
from pathlib import Path
import argparse,hashlib,io,json,os,sys,tarfile,time,importlib.util,importlib.metadata
import numpy as np
from audit import sha,write,MODEL,REPO
from calibrate import accumulate
from approximate import Arithmetic,DomainViolation
HERE=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--limit',type=int,default=3668)
    args=parser.parse_args();args.output.mkdir(exist_ok=False);start=time.monotonic();before={}
    report=dict(scope=__doc__,passed=False,completed_examples=0,training_labels_loaded=False,approximate_trajectory_coverage=False)
    try:
        write(args.output/'report.json',report)
        import torch,pyarrow.parquet as pq
        from transformers import AutoTokenizer,BertForSequenceClassification
        torch.set_num_threads(8);torch.set_num_interop_threads(1)
        native=json.loads((HERE/'train3668-v1/report.json').read_text());assets=native['model_assets']
        assert {n:sha(MODEL/n) for n in assets}==assets
        data=json.loads((HERE/'dataset_manifest.json').read_text());path=Path('/home/jianming_gatech/cross_validation_assets/mrpc')/data['revision']/data['file'];assert sha(path)==data['sha256']
        rows=pq.read_table(path,columns=['sentence1','sentence2','idx']).to_pylist();assert len(rows)==3668 and all('label' not in r for r in rows)
        if not 1<=args.limit<=3668:raise ValueError('invalid training subset count')
        manifest_path=HERE/'candidate-v2/candidate_manifest.json';manifest=json.loads(manifest_path.read_text())
        selected=[Path(__file__),HERE/'approximate.py',HERE/'test_approximate.py',HERE/'calibrate.py',HERE/'audit.py',HERE/'dataset_manifest.json',HERE/'select_v2.py',
            HERE/'candidate-v2/selection_protocol.json',HERE/'candidate-v2/report.json',manifest_path,HERE/'train3668-v1/report.json',
            REPO/'cadathlon/agent_pset/p2_bert/src/reference.py']
        selected += [Path(n) for n in native['source_hashes_before'] if '/site-packages/transformers/' in n]
        before={str(p.resolve()):sha(p) for p in selected}
        with tarfile.open(args.output/'sources.tar.gz','w:gz') as archive:
            for name in before:
                raw=Path(name).read_bytes();info=tarfile.TarInfo(name.lstrip('/'));info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
        with tarfile.open(args.output/'sources.tar.gz') as archive:
            assert len(archive.getmembers())==len(before)
            assert all(hashlib.sha256(archive.extractfile(m).read()).hexdigest()==before['/'+m.name] for m in archive.getmembers())
        module_path=REPO/'cadathlon/agent_pset/p2_bert/src/reference.py'
        spec=importlib.util.spec_from_file_location('reference',module_path);reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        model,loading=BertForSequenceClassification.from_pretrained(MODEL,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        if loading['missing_keys'] or loading.get('mismatched_keys') or loading.get('error_msgs') or set(loading['unexpected_keys'])-{'bert.embeddings.position_ids'}:raise ValueError('incomplete model')
        model.eval();config=model.config.to_dict();weights={n:v.detach().numpy().astype(np.float64) for n,v in model.state_dict().items()};tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        engine=Arithmetic(manifest);aggregate={};failures=[];numeric_failures=[];tokenhash=hashlib.sha256();successful=0;numeric_pass=0;tie_pass=0;worst=0.
        report.update(candidate_manifest_sha256=sha(manifest_path),dataset=data,model_assets=assets,source_hashes_before=before,
            source_archive_sha256=sha(args.output/'sources.tar.gz'),source_archive_verified=True,
            cpu_affinity=sorted(os.sched_getaffinity(0)),threads=8,native_reference_dtype='float32',approximate_dtype='float64',
            original_logit_gate='abs(approx-native)<=0.02+0.01*abs(native)',original_tie_rule='class1 iff logit1>=logit0',
            versions={n:importlib.metadata.version(n) for n in ('numpy','torch','transformers','tokenizers','safetensors','pyarrow')})
        for i,row in enumerate(rows[:args.limit]):
            tokens=tokenizer(row['sentence1'],row['sentence2'],max_length=128,padding='max_length',truncation=True,return_tensors='pt')
            encoded=dict(target_idx=row['idx'],**{n:v.tolist() for n,v in sorted(tokens.items())});tokenhash.update(json.dumps(encoded,sort_keys=True,separators=(',',':')).encode()+b'\n')
            with torch.no_grad():expected=model(**tokens).logits[0].numpy().astype(np.float64)
            embeddings=reference.embed(config,weights,tokens['input_ids'].numpy(),tokens['token_type_ids'].numpy())[0]
            mask=tokens['attention_mask'][0].numpy().astype(bool);engine.observations={}
            try:
                with np.errstate(over='raise',invalid='raise',divide='raise'):actual=engine.forward(config,weights,embeddings[mask])
            except DomainViolation as error:
                failures.append(dict(position=i,target_idx=row['idx'],operator=error.label,observation=error.row,
                    logits_evaluated=False,policy='Stopped before this out-of-domain nonlinear operation; no clipping or gate claim.'))
            else:
                if actual.shape!=(2,) or not np.isfinite(actual).all():raise ValueError('invalid approximate logits')
                successful+=1;error=float(np.max(np.abs(actual-expected)));worst=max(worst,error)
                numeric=bool(np.all(np.abs(actual-expected)<=.02+.01*np.abs(expected)));ties=bool((actual[1]>=actual[0])==(expected[1]>=expected[0]))
                numeric_pass+=numeric;tie_pass+=ties
                if not numeric or not ties:numeric_failures.append(dict(position=i,target_idx=row['idx'],max_abs_logit_error=error,logit_gate_pass=numeric,native_prediction_agreement=ties))
            accumulate(aggregate,engine.observations);report['completed_examples']=i+1
            if i%25==0 or i+1==args.limit:
                progress=dict(completed=i+1,domain_failures=len(failures),numeric_failures=len(numeric_failures),elapsed_s=time.monotonic()-start)
                write(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        digest=tokenhash.hexdigest()
        if args.limit==3668 and digest!=native['tokenization_sha256']:raise ValueError('training preprocessing differs from calibration')
        report.update(domain_observations=aggregate,domain_failures=failures,numeric_failures=numeric_failures,
            approximate_trajectory_coverage=not failures,complete_model_outputs=successful,logit_gate_passes=numeric_pass,native_prediction_agreements=tie_pass,
            worst_complete_logit_error=worst if successful else None,tokenization_sha256=digest,
            all_training_examples_evaluated=args.limit==3668,no_jax_imported=not any(n=='jax' or n.startswith('jax.') for n in sys.modules),
            validation_evaluated=False,labels_used=False,encrypted_execution=False,passed=not failures and not numeric_failures,
            limitation='Training-only float64 arithmetic analogue with native float32 logit reference; no label accuracy, unseen-input guarantee, CKKS error or runtime calibration. Padding is compacted using public masks and original positions before encrypted-graph analogue.')
        assert report['no_jax_imported']
    except BaseException as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}');raise
    finally:
        after={n:sha(n) for n in before};report.update(source_hashes_after=after,source_files_unchanged=bool(before) and before==after,elapsed_s=time.monotonic()-start)
        report['passed']=bool(report['passed'] and before==after);write(args.output/'report.json',report)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
