"""Trusted pinned full BertLayer reference on existing original-position embeddings."""
from pathlib import Path
import hashlib,importlib.metadata,json,os,time
import numpy as np
HERE=Path(__file__).resolve().parent
MODEL=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c')
FIXTURE=Path('/tmp/bert-first-attention-owner-20261103/client_fixtures.npz')
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    began=time.monotonic();pinned=json.loads((HERE/'verified_model_manifest.json').read_text())
    source_paths=[Path(__file__),HERE/'verified_model_manifest.json',HERE/'fixture_preparation.json',HERE/'joined_source_origin.json']
    before={str(path):sha(path) for path in source_paths}
    assets={name:sha(MODEL/name) for name in pinned['files']}
    assert assets=={name:row['sha256'] for name,row in pinned['files'].items()}
    expected=json.loads((HERE/'fixture_preparation.json').read_text());assert sha(FIXTURE)==expected['fixture_npz_sha256']
    with np.load(FIXTURE,allow_pickle=False) as source:inputs={name:source[name].copy() for name in source.files}
    import torch
    from transformers import BertForSequenceClassification
    torch.set_num_threads(4)
    model,loading=BertForSequenceClassification.from_pretrained(MODEL,local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float64,attn_implementation='eager',output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
    assert set(loading['unexpected_keys'])<={'bert.embeddings.position_ids'}
    model.eval();outputs={};records=[]
    with torch.no_grad():
        for K in (15,128):
            embeddings=torch.as_tensor(inputs[f'K{K}_embeddings'][:K][None],dtype=torch.float64)
            positions=inputs[f'K{K}_original_positions'];assert len(positions)==K
            mask=torch.zeros((1,1,1,K),dtype=torch.float64)
            attention=model.bert.encoder.layer[0].attention(embeddings,attention_mask=mask)[0]
            difference=float(np.max(np.abs(attention[0].numpy()-inputs[f'K{K}_native'])))
            assert difference<1e-11
            result=model.bert.encoder.layer[0](embeddings,attention_mask=mask)[0][0].numpy().copy()
            assert result.shape==(K,768) and np.isfinite(result).all()
            outputs[f'K{K}_native_block']=result
            records.append(dict(K=K,shape=list(result.shape),dtype=result.dtype.str,
                source_position_indices=list(map(int,positions)),attention_matches_prior_native_max_abs=difference,
                block_output_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
    target=HERE/'native_block_fixtures.npz';assert not target.exists();np.savez(target,**outputs)
    after={str(path):sha(path) for path in source_paths}
    report=dict(passed=before==after,scope=__doc__,encrypted_execution=False,original_position_embeddings_preserved=True,
        native_dtype='float64',reference='transformers.BertLayer.forward',cases=records,
        source_hashes_before=before,source_hashes_after=after,source_files_unchanged=before==after,
        checkpoint_revision=pinned['revision'],asset_sha256=assets,input_fixture_sha256=sha(FIXTURE),
        output_fixture_sha256=sha(target),cpu_affinity=sorted(os.sched_getaffinity(0)),threads=4,
        versions={n:importlib.metadata.version(n) for n in ('torch','transformers','numpy')},elapsed_s=time.monotonic()-began)
    (HERE/'native_block_preparation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','scope','cases','output_fixture_sha256','elapsed_s')},indent=2))
    if not report['passed']:raise ValueError('trusted preparation source changed')
if __name__=='__main__':main()
