"""Held-out native/approximate MRPC check after complete training-domain audit."""
from pathlib import Path
import argparse,hashlib,io,json,os,sys,tarfile,time,importlib.util,importlib.metadata
import numpy as np
from audit import sha,write,MODEL,REPO
from calibrate import accumulate
from approximate import Arithmetic,DomainViolation
from ranked_metrics import evaluate
HERE=Path(__file__).resolve().parent

def require_training_coverage(path,manifest):
    r=json.loads(Path(path).read_text())
    if (r.get('completed_examples')!=3668 or r.get('complete_model_outputs')!=3668
            or r.get('source_archive_verified') is not True or r.get('validation_evaluated') is not False
            or r.get('all_training_examples_evaluated') is not True
            or r.get('approximate_trajectory_coverage') is not True or r.get('source_files_unchanged') is not True
            or r.get('no_jax_imported') is not True or r.get('labels_used') is not False
            or r.get('training_labels_loaded') is not False or r.get('domain_failures')!=[]
            or r.get('candidate_manifest_sha256')!=sha(manifest)):
        raise ValueError('complete, unchanged, label-free training-domain evidence for this exact candidate is required')
    before=r.get('source_hashes_before');after=r.get('source_hashes_after')
    if not isinstance(before,dict) or not before or before!=after:
        raise ValueError('complete training source hashes are required')
    if any(not isinstance(name,str) or not Path(name).is_absolute() or not isinstance(digest,str)
            or len(digest)!=64 for name,digest in before.items()):
        raise ValueError('malformed training source inventory')
    archive_path=Path(path).parent/'sources.tar.gz'
    if sha(archive_path)!=r.get('source_archive_sha256'):
        raise ValueError('training source archive digest differs')
    with tarfile.open(archive_path) as archive:
        members=archive.getmembers()
        if (len(members)!=len(before) or any(not member.isfile() for member in members)
                or {'/'+member.name for member in members}!=set(before)):
            raise ValueError('training source archive closure differs')
        for member in members:
            if hashlib.sha256(archive.extractfile(member).read()).hexdigest()!=before['/'+member.name]:
                raise ValueError('training source archive bytes differ')
    if any(sha(name)!=digest for name,digest in before.items()):
        raise ValueError('current arithmetic or dependencies differ from training evidence')
    # A local-toy prediction diagnostic may fail while ranked label accuracy
    # remains acceptable. Do not reinterpret its stronger passed flag here.
    return r

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False);start=time.monotonic();before={}
    report=dict(scope=__doc__,passed=False,completed_examples=0,encrypted_execution=False)
    try:
        write(args.output/'report.json',report)
        manifest_path=HERE/'candidate-v2/candidate_manifest.json'
        training_path=HERE/'approximate-train3668-v2/report.json';training=require_training_coverage(training_path,manifest_path)
        # No dataset/model import or file read occurs before the prerequisite.
        import torch,pyarrow.parquet as pq
        from transformers import AutoTokenizer,BertForSequenceClassification
        torch.set_num_threads(8);torch.set_num_interop_threads(1)
        assets=training['model_assets'];assert {n:sha(MODEL/n) for n in assets}==assets
        data_path=HERE/'validation_dataset_manifest.json';data=json.loads(data_path.read_text())
        path=Path('/home/jianming_gatech/cross_validation_assets/mrpc')/data['revision']/data['file'];assert sha(path)==data['sha256']
        rows=pq.read_table(path,columns=['sentence1','sentence2','idx']).to_pylist()
        assert len(rows)==408 and len({r['idx'] for r in rows})==408
        selected=[Path(__file__),HERE/'test_validate_candidate.py',HERE/'ranked_metrics.py',HERE/'test_ranked_metrics.py',
            HERE/'approximate.py',HERE/'test_approximate.py',HERE/'calibrate.py',HERE/'audit.py',data_path,
            HERE/'candidate-v2/selection_protocol.json',HERE/'candidate-v2/report.json',manifest_path,training_path,
            REPO/'cadathlon/agent_pset/p2_bert/src/reference.py',REPO/'cadathlon/agent_pset/p2_bert/doc/problem.md']
        selected += [Path(n) for n in training['source_hashes_before'] if '/site-packages/transformers/' in n]
        before={str(p.resolve()):sha(p) for p in selected}
        with tarfile.open(args.output/'sources.tar.gz','w:gz') as archive:
            for name in before:
                raw=Path(name).read_bytes();info=tarfile.TarInfo(name.lstrip('/'));info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
        with tarfile.open(args.output/'sources.tar.gz') as archive:
            assert len(archive.getmembers())==len(before)
            assert all(hashlib.sha256(archive.extractfile(m).read()).hexdigest()==before['/'+m.name] for m in archive.getmembers())
        spec=importlib.util.spec_from_file_location('reference',REPO/'cadathlon/agent_pset/p2_bert/src/reference.py');reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        model,loading=BertForSequenceClassification.from_pretrained(MODEL,local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        if loading['missing_keys'] or loading.get('mismatched_keys') or loading.get('error_msgs') or set(loading['unexpected_keys'])-{'bert.embeddings.position_ids'}:raise ValueError('incomplete model')
        model.eval();config=model.config.to_dict();weights={n:v.detach().numpy().astype(np.float64) for n,v in model.state_dict().items()};tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        manifest=json.loads(manifest_path.read_text());engine=Arithmetic(manifest)
        aggregate={};failures=[];numeric=[];tokenhash=hashlib.sha256();native_logits=[];approximate_logits=[];coverage=[]
        report.update(candidate_manifest_sha256=sha(manifest_path),training_report_sha256=sha(training_path),dataset=data,model_assets=assets,
            source_hashes_before=before,source_archive_sha256=sha(args.output/'sources.tar.gz'),source_archive_verified=True,
            cpu_affinity=sorted(os.sched_getaffinity(0)),threads=8,native_reference_dtype='float32',approximate_dtype='float64',
            versions={n:importlib.metadata.version(n) for n in ('numpy','torch','transformers','tokenizers','safetensors','pyarrow')})
        for i,row in enumerate(rows):
            tokens=tokenizer(row['sentence1'],row['sentence2'],max_length=128,padding='max_length',truncation=True,return_tensors='pt')
            encoded=dict(target_idx=row['idx'],**{n:v.tolist() for n,v in sorted(tokens.items())});tokenhash.update(json.dumps(encoded,sort_keys=True,separators=(',',':')).encode()+b'\n')
            with torch.no_grad():expected=model(**tokens).logits[0].numpy().astype(np.float64)
            if expected.shape!=(2,) or not np.isfinite(expected).all():raise ValueError('invalid native baseline logits')
            native_logits.append(expected)
            embeddings=reference.embed(config,weights,tokens['input_ids'].numpy(),tokens['token_type_ids'].numpy())[0]
            mask=tokens['attention_mask'][0].numpy().astype(bool);engine.observations={}
            try:
                with np.errstate(over='raise',invalid='raise',divide='raise'):actual=engine.forward(config,weights,embeddings[mask])
            except DomainViolation as error:
                failures.append(dict(position=i,target_idx=row['idx'],operator=error.label,observation=error.row,logits_evaluated=False))
                approximate_logits.append(np.full(2,np.nan));coverage.append(False)
            else:
                if actual.shape!=(2,) or not np.isfinite(actual).all():raise ValueError('invalid approximate logits')
                approximate_logits.append(actual);coverage.append(True)
                logit_gate=bool(np.all(np.abs(actual-expected)<=.02+.01*np.abs(expected)))
                agreement=bool((actual[1]>=actual[0])==(expected[1]>=expected[0]))
                if not logit_gate or not agreement:numeric.append(dict(position=i,target_idx=row['idx'],max_abs_logit_error=float(np.max(np.abs(actual-expected))),local_toy_logit_gate_pass=logit_gate,prediction_agreement=agreement))
            accumulate(aggregate,engine.observations);report['completed_examples']=i+1
            if i%25==0 or i==407:
                progress=dict(completed=i+1,domain_failures=len(failures),screening_differences=len(numeric),elapsed_s=time.monotonic()-start)
                write(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
        # Labels are loaded only after every approximate output is fixed.
        assert sha(path)==data['sha256']
        labels=pq.read_table(path,columns=['label']).column('label').to_numpy();assert sha(path)==data['sha256']
        ranked=evaluate(np.array([r['idx'] for r in rows]),labels,np.asarray(native_logits),np.asarray(approximate_logits),np.asarray(coverage))
        report.update(ranked=ranked,domain_observations=aggregate,domain_failures=failures,screening_differences=numeric,
            domain_coverage_passed=not failures,labels_loaded_after_outputs=True,tokenization_sha256=tokenhash.hexdigest(),
            approximate_logits_or_hidden_arrays_persisted=False,no_jax_imported=not any(n=='jax' or n.startswith('jax.') for n in sys.modules),
            asset_hashes_after={n:sha(MODEL/n) for n in assets},passed=not failures and ranked['all_ranked_subsets_pass'],
            limitation='Held-out plaintext arithmetic validation on official MRPC dev; no encrypted execution, full file-stage result, arbitrary-input bound, or timing calibration. No validation statistic changes the frozen candidate.')
        assert report['no_jax_imported'] and report['asset_hashes_after']==assets
    except BaseException as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}');raise
    finally:
        after={n:sha(n) for n in before};report.update(source_hashes_after=after,source_files_unchanged=bool(before) and before==after,elapsed_s=time.monotonic()-start)
        report['passed']=bool(report['passed'] and before==after);write(args.output/'report.json',report)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
