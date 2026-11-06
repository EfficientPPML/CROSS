from pathlib import Path
import json
import numpy as np
import pytest
import arithmetic as a
import numeric
P=Path(__file__).resolve().parent;M=json.loads((P/'candidate-v1/manifest.json').read_text())

def test_every_frozen_nonlinear_matches_public_evaluation_and_exact_depth():
    for label,r in M['nonlinear'].items():
        x=np.linspace(*r['domain'],129);engine=a.QwenArithmetic(M);value=a.Value(x,7)
        if label.endswith('inverse_sqrt'):
            got=engine.inverse_sqrt(value,r,label)
            wanted=numeric.polynomial(x,r) if r['method']=='polynomial' else numeric.rsqrt(x,r)
        else:got=engine.silu(value,r,label);wanted=numeric.polynomial(x,r) if r['method']=='polynomial' else numeric.rational_silu(x,r)
        assert np.array_equal(got.data,wanted),label
        assert got.depth==7+r['depth'],label

def test_all_reciprocal_contracts_use_frozen_rounded_affines():
    for k,r in M['reciprocal_by_keys'].items():
        x=np.linspace(*r['domain'],129);engine=a.QwenArithmetic(M);out=engine.inverse(a.Value(x,5),r,'inv'+k)
        assert np.array_equal(out.data,numeric.reciprocal(x,r));assert out.depth==5+r['depth']

def test_domain_failure_stops_before_arithmetic_and_does_not_clip():
    r=M['nonlinear']['final.inverse_sqrt'];x=np.array([r['domain'][0]/2]);original=x.copy();engine=a.QwenArithmetic(M)
    with pytest.raises(a.InvalidCircuit) as caught:engine.inverse_sqrt(a.Value(x),r,'bad')
    assert caught.value.label=='bad' and caught.value.record['below']==1
    assert not engine.counts and np.array_equal(x,original)

def test_nonfinite_result_is_not_substituted():
    engine=a.QwenArithmetic(M)
    with np.errstate(over='ignore'):
        with pytest.raises(a.InvalidCircuit):engine.multiply(a.Value(np.array([1e308])),a.Value(np.array([1e308])),'overflow')

def test_symbolic_full_graph_depth_matches_published_count_without_model_values():
    # This deliberately substitutes zero-shaped symbolic payloads after every
    # primitive. It tests only graph traversal/depth, never model accuracy.
    class Symbolic(a.QwenArithmetic):
        def node(self,data,depth,label):return a.Value(np.zeros_like(data,dtype=float),depth)
        def observe(self,label,data,domain):pass
    c={'hidden_size':4,'intermediate_size':6,'num_hidden_layers':28,'num_attention_heads':2,'num_key_value_heads':1,'head_dim':2,'vocab_size':8,'rms_norm_eps':1e-6,'rope_theta':10000.}
    w={}
    for i in range(28):
        p=f'model.layers.{i}.'
        for name in ('input_layernorm','post_attention_layernorm'):w[p+name+'.weight']=np.ones(4)
        for name,shape in {'q_proj':(4,4),'k_proj':(2,4),'v_proj':(2,4),'o_proj':(4,4)}.items():w[p+'self_attn.'+name+'.weight']=np.zeros(shape)
        for name in ('q_norm','k_norm'):w[p+'self_attn.'+name+'.weight']=np.ones(2)
        for name,shape in {'gate_proj':(6,4),'up_proj':(6,4),'down_proj':(4,6)}.items():w[p+'mlp.'+name+'.weight']=np.zeros(shape)
    w['model.norm.weight']=np.ones(4);w['model.embed_tokens.weight']=np.zeros((8,4))
    for S in (8,16,19,32):
        engine=Symbolic(M);out,_,blocks,operators=a.approximate_forward(c,w,np.zeros((1,S,4)),M,engine=engine)
        assert out.depth==M['depth_by_sequence'][str(S)]['final_depth']
        assert len(blocks)==28 and len(operators)==421
        assert [x['depth_after'] for x in engine.phase_costs]==[x['depth_after'] for x in M['depth_by_sequence'][str(S)]['layers']]
