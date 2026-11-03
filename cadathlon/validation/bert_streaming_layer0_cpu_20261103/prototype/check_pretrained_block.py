"""Pinned full-width BERT layer0 layout semantics; exact nonlinear oracle only."""
import argparse,hashlib,importlib.metadata,json,math,os
from pathlib import Path
import resource,sys,time
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
ROOT=Path(os.environ.get('CROSS_ROOT','/home/jianming_gatech/CROSS')).resolve()
sys.path[:0]=[str(ROOT/'cadathlon/common'),str(Path(__file__).resolve().parent)]
from support import load_module
from row_attention import RowPackedAttention


def digest(path):
    with path.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def dense_attention(q,k,v,mask,heads=12):
    s,h=q.shape;d=h//heads
    q,k,v=[value.reshape(s,heads,d).transpose(1,0,2) for value in (q,k,v)]
    scores=q@k.transpose(0,2,1)/math.sqrt(d)
    scores=np.where(mask[None,None,:],scores,-np.inf)
    probabilities=np.exp(scores-scores.max(-1,keepdims=True));probabilities/=probabilities.sum(-1,keepdims=True)
    return (probabilities@v).transpose(1,0,2).reshape(s,h)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    audit=load_module(ROOT/'cadathlon/validation/check_pretrained_bert_domains.py')
    reference=load_module(ROOT/'cadathlon/agent_pset/p2_bert/src/reference.py')
    hashes={name:digest(args.snapshot/name) for name in audit.EXPECTED_ASSETS}
    if hashes!=audit.EXPECTED_ASSETS:raise ValueError('pinned checkpoint asset mismatch')
    sources=[Path(__file__),Path(__file__).with_name('row_attention.py'),ROOT/'cadathlon/agent_pset/p2_bert/src/reference.py',ROOT/'cadathlon/validation/check_pretrained_bert_domains.py']
    before={str(path):digest(path) for path in sources};started=time.perf_counter()
    import torch
    from transformers import AutoTokenizer,BertForSequenceClassification
    torch.set_num_threads(4)
    model,loading=BertForSequenceClassification.from_pretrained(args.snapshot,local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float64,attn_implementation='eager',output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
    assert set(loading['unexpected_keys'])<={'bert.embeddings.position_ids'}
    model.eval();config=model.config.to_dict();weights={name:value.detach().numpy() for name,value in model.state_dict().items()}
    assert config['hidden_size']==768 and config['num_attention_heads']==12 and config['intermediate_size']==3072
    tokenizer=AutoTokenizer.from_pretrained(args.snapshot,local_files_only=True)
    public=list(audit.EXAMPLES)+[('The committee reviews the proposal and discusses the evidence. '*20,
                                'Members discuss the proposal before reaching a decision. '*20)]
    tokens=tokenizer([a for a,b in public],[b for a,b in public],max_length=128,truncation=True,padding='max_length',return_tensors='pt')
    embeddings=reference.embed(config,weights,tokens['input_ids'].numpy(),tokens['token_type_ids'].numpy())
    with torch.no_grad():native_embeddings=model.bert.embeddings(input_ids=tokens['input_ids'],token_type_ids=tokens['token_type_ids']).numpy()
    assert np.max(np.abs(embeddings-native_embeddings))<1e-12
    masks=tokens['attention_mask'].numpy().astype(bool)
    cases=[(f'public_pair_{i}',embeddings[i:i+1],masks[i:i+1]) for i in range(4)]
    boundary=np.zeros((1,128),bool);boundary[0,[0,15,16,31,32,63,64,95,111,112,127]]=True
    last=np.zeros((1,128),bool);last[0,-1]=True
    cases.extend([('public_boundary_mask',embeddings[:1],boundary),('public_last_key_only',embeddings[:1],last),
                  ('public_partial_tile17',embeddings[:1,:17],masks[:1,:17])])
    reports=[];prefix='bert.encoder.layer.0.'
    for name,x,mask in cases:
        print(json.dumps({'starting':name,'sequence':x.shape[1],'valid_keys':int(mask.sum())}),flush=True)
        q,k,v=[reference.linear(x,weights,prefix+'attention.self.'+kind)[0] for kind in ('query','key','value')]
        simulator=RowPackedAttention(diagnostic_domain=(-9,12))
        context=simulator.attention(q,k,v,mask[0])[None]
        dense=dense_attention(q,k,v,mask[0])[None]
        post_attention=reference.layer_norm(x+reference.linear(context,weights,prefix+'attention.output.dense'),weights,
            prefix+'attention.output.LayerNorm',config['layer_norm_eps'])
        intermediate=reference.gelu(reference.linear(post_attention,weights,prefix+'intermediate.dense'))
        actual=reference.layer_norm(post_attention+reference.linear(intermediate,weights,prefix+'output.dense'),weights,
            prefix+'output.LayerNorm',config['layer_norm_eps'])
        expected=reference.encoder_block(config,weights,x,mask,0)
        with torch.no_grad():
            native_x=torch.from_numpy(x.copy());native_mask=torch.from_numpy(mask.copy())
            additive=torch.zeros(native_mask[:,None,None,:].shape,dtype=torch.float64).masked_fill(~native_mask[:,None,None,:],torch.finfo(torch.float64).min)
            native=model.bert.encoder.layer[0](native_x,attention_mask=additive)[0].numpy()
            native_context=model.bert.encoder.layer[0].attention.self(native_x,attention_mask=additive)[0].numpy()
        comparisons={'context_vs_dense':float(np.max(np.abs(context-dense))),
            'context_vs_transformers':float(np.max(np.abs(context-native_context))),
            'full_block_vs_numpy':float(np.max(np.abs(actual-expected))),
            'full_block_vs_transformers':float(np.max(np.abs(actual-native)))}
        passed=all(value<1e-9 for value in comparisons.values()) and np.isfinite(actual).all()
        padded=~mask
        report=dict(case=name,sequence=int(x.shape[1]),valid_keys=int(mask.sum()),padded_queries=int(padded.sum()),
            passed=bool(passed),maximum_absolute_errors=comparisons,
            padded_query_output_max_abs=float(np.max(np.abs(actual[padded]))) if padded.any() else None,
            observed_ranges=simulator.ranges,layout_plan=simulator.plan(x.shape[1]))
        counts=report['layout_plan']['actual_simulated_counts'];expected_counts=report['layout_plan']['expected_counts']
        assert all(counts[key]==value for key,value in expected_counts.items())
        reports.append(report);print(json.dumps({'completed':name,'passed':bool(passed),'errors':comparisons}),flush=True)
    after={str(path):digest(path) for path in sources}
    report=dict(scope='Plaintext full-width pinned BERT encoder layer0 attention/layout parity; no encrypted execution or MRPC evaluation.',
        model_revision='f150c1d609d1e50dd5e2e5408661cfac8339277c',asset_sha256=hashes,
        layer=0,hidden=768,heads=12,head_dimension=64,intermediate=3072,examples=public,
        cases=reports,passed=all(row['passed'] for row in reports),
        source_hashes_before=before,source_hashes_after=after,source_files_unchanged=before==after,
        versions={name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','safetensors')},
        elapsed_s=time.perf_counter()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations=['Exact exp/division, GELU and LayerNorm are plaintext oracle functions, not CKKS circuits.',
            'Prior centered-score interval[-9,12] was selected on valid query tokens of three short pairs; padded-query and other-input violations remain visible here.',
            'Unnormalized exp accumulation differs from prior normalized repeated squaring; observed large magnitudes may make it unsuitable for CKKS refresh.',
            'Each Q/K/V projection and remaining dense layer is NumPy/Torch; full-width BSGS execution/precision is not validated.',
            'Full-slot period65536 has no sparse alias averaging; period128 refresh proof does not apply.'])
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['passed'] and report['source_files_unchanged'] else 1

if __name__=='__main__':raise SystemExit(main())
