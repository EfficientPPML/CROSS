"""Trusted public test preprocessing; never imported by the encrypted owner."""
from pathlib import Path
import argparse,hashlib,json,os,time
import numpy as np
HERE=Path(__file__).resolve().parent
PAIRS=(
    ('The committee approved the proposal.','The committee approved the proposal.'),
    ('The committee reviews the proposal and discusses the evidence. '*20,
     'Members discuss the proposal before reaching a decision. '*20),
)

def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);args=p.parse_args()
    start=time.monotonic();pinned=json.loads((HERE/'verified_model_manifest.json').read_text())
    assert args.snapshot.name==pinned['revision']
    hashes={name:digest(args.snapshot/name) for name in pinned['files']}
    assert hashes=={name:value['sha256'] for name,value in pinned['files'].items()}
    import torch
    from transformers import AutoTokenizer,BertForSequenceClassification
    torch.set_num_threads(4)
    model,loading=BertForSequenceClassification.from_pretrained(args.snapshot,local_files_only=True,use_safetensors=True,
        torch_dtype=torch.float64,attn_implementation='eager',output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
    assert set(loading['unexpected_keys'])<={'bert.embeddings.position_ids'}
    model.eval();tokenizer=AutoTokenizer.from_pretrained(args.snapshot,local_files_only=True)
    cfg=model.config
    assert (cfg.hidden_size,cfg.num_attention_heads,cfg.intermediate_size,cfg.layer_norm_eps)==(768,12,3072,1e-12)
    state=model.state_dict();arrays={};rows=[]
    for kind in ('query','key','value','output'):
        prefix=f'bert.encoder.layer.0.attention.self.{kind}' if kind!='output' else 'bert.encoder.layer.0.attention.output.dense'
        for suffix in ('weight','bias'):arrays[kind+'_'+suffix]=state[prefix+'.'+suffix].numpy().copy()
    for suffix,key in (('weight','gamma'),('bias','beta')):
        arrays[key]=state['bert.encoder.layer.0.attention.output.LayerNorm.'+suffix].numpy().copy()
    with torch.no_grad():
        for index,pair in enumerate(PAIRS):
            token=tokenizer(*pair,max_length=128,truncation=True,padding='max_length',return_tensors='pt')
            mask=token['attention_mask'][0].numpy().astype(bool);positions=np.flatnonzero(mask);K=len(positions)
            assert K==(15 if index==0 else 128)
            embeddings=model.bert.embeddings(input_ids=token['input_ids'],token_type_ids=token['token_type_ids'])
            additive=(1-token['attention_mask'][:,None,None,:].to(torch.float64))*torch.finfo(torch.float64).min
            native=model.bert.encoder.layer[0].attention(embeddings,attention_mask=additive)[0]
            compact=embeddings[:,mask,:]
            compact_native=model.bert.encoder.layer[0].attention(compact,attention_mask=torch.zeros((1,1,1,K),dtype=torch.float64))[0]
            parity=float(torch.max(torch.abs(compact_native-native[:,mask,:])))
            assert parity<1e-12
            frame=np.zeros((128,768));frame[:K]=compact[0].numpy()
            arrays[f'K{K}_embeddings']=frame;arrays[f'K{K}_native']=compact_native[0].numpy().copy()
            arrays[f'K{K}_original_positions']=positions.astype(np.int64)
            rows.append(dict(K=K,query_tiles=(K+15)//16,original_positions=positions.tolist(),native_compaction_max_abs=parity,
                tokenization_sha256=hashlib.sha256(json.dumps({k:v.tolist() for k,v in token.items()},sort_keys=True).encode()).hexdigest()))
    path=HERE/'client_fixtures.npz';np.savez(path,**arrays)
    report=dict(passed=True,scope='Trusted public fixed-pair embedding and native first-attention preparation only; no HE.',
        pairs=[list(x) for x in PAIRS],cases=rows,asset_sha256=hashes,revision=pinned['revision'],fixture_npz_sha256=digest(path),
        arrays={k:dict(shape=list(v.shape),dtype=v.dtype.str,sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()},
        elapsed_s=time.monotonic()-start,cpu_affinity=sorted(os.sched_getaffinity(0)),native_dtype='float64',threads=4)
    (HERE/'fixture_preparation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','scope','cases','elapsed_s','fixture_npz_sha256')},indent=2))

if __name__=='__main__':main()
