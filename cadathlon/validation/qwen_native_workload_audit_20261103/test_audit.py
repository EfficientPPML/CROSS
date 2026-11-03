import json
from pathlib import Path
import numpy as np
import pytest
import audit


def test_fixture_counts_and_real_tokenized_uniqueness():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(audit.SNAPSHOT,local_files_only=True)
    fixtures=audit.tokenize_fixtures(tokenizer,json.loads((audit.HERE/'fixtures.json').read_text()))
    assert len(fixtures)==56
    assert all(len(f['token_ids'])==f['sequence_length'] and f['raw_token_count']>=32 for f in fixtures)


def test_native_observation_is_bitwise_inert_and_restored():
    import torch
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from transformers.models.qwen3 import modeling_qwen3 as module
    torch.manual_seed(871)
    config=Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=24,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,tie_word_embeddings=True)
    config._attn_implementation='eager';model=Qwen3ForCausalLM(config).eval()
    manifest={'nonlinear':{},'softmax':{}}
    for layer in range(2):
        for suffix in ('input','post','q','k'):manifest['nonlinear'][f'{layer}.{suffix}.inverse_sqrt']={'domain':[1e-12,1e6]}
        manifest['nonlinear'][f'{layer}.silu']={'domain':[-1e6,1e6]}
        manifest['softmax'][str(layer)]={'centered_domain':[-1e6,1e6],'squarings':5,'reciprocal_iterations_by_keys':{'1':6,'2':7,'3':8,'4':8}}
    manifest['nonlinear']['final.inverse_sqrt']={'domain':[1e-12,1e6]}
    ids=torch.tensor([[2,6,8,11,4,3,7,12]])
    with torch.no_grad():expected=model(input_ids=ids,use_cache=False).logits.clone()
    original=module.eager_attention_forward
    with audit.NativeObserver(model,manifest) as observer:
        observer.begin()
        with torch.no_grad():actual=model(input_ids=ids,use_cache=False).logits
        records,gaps=observer.end()
        assert torch.equal(expected,actual)
        assert records['final.inverse_sqrt']['count']==1
        assert records['0.q.inverse_sqrt']['count']==16
        assert records['0.k.inverse_sqrt']['count']==8
        assert gaps==[(layer,k) for layer in range(2) for k in range(5,9)]
    assert module.eager_attention_forward is original
    with pytest.raises(KeyboardInterrupt):
        with audit.NativeObserver(model,manifest):raise KeyboardInterrupt()
    assert module.eager_attention_forward is original
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())


def test_causal_centering_excludes_all_future_keys():
    a=np.arange(2*4*4,dtype=float).reshape(1,2,4,4);b=a.copy()
    b[...,np.triu(np.ones((4,4),dtype=bool),1)]=1e20
    rows1,c1=audit.causal_centered(a);rows2,c2=audit.causal_centered(b)
    assert all(np.array_equal(x,y) for x,y in zip(rows1,rows2))
    assert all(np.array_equal(x,y) for x,y in zip(c1,c2))
    assert sum(x.size for x in rows1)==20
    assert all(np.allclose(x.mean(-1),0) for x in c1)


def test_decode_keeps_teacher_and_free_contexts_separate():
    calls=[]
    def callback(ids,kind,step):
        calls.append((list(ids),kind,step))
        native=np.array([0.,2.,1.]);oracle=np.array([3.,0.,1.])
        return native,oracle,{'step':step}
    initial=list(range(16));out=audit.decode_cases(initial,callback)
    assert initial==list(range(16)) and len(calls)==8
    assert [len(ids) for ids,_,_ in calls]==[16,17,18,19]*2
    assert calls[1][0][-1]==1 and calls[5][0][-1]==0
    assert out['native_tokens']==[1]*4 and out['oracle_free_tokens']==[0]*4
    assert not out['complete_sequence_agreement']


def test_semantic_gates_are_unchanged_and_not_domain_coverage():
    assert audit.GATES=={'max_abs':1e-3,'nrmse':1e-5,'exact_top1':True}
    x=np.ones((1,151936));x[0,8]=2
    assert audit.check_logits(x,x)['passed']
    y=x.copy();y[0,8]+=1e-2
    assert not audit.check_logits(y,x)['passed']
    row=audit.stats([-2.,0.,3.],[-1.,1.]);assert row['below']==row['above']==1
