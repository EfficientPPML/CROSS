"""Actual preceding score loops plus public probability stand-ins and densePV."""
import math
import numpy as np
import pytest
import test_head_routing as base
from encrypted_qwen_attention_routing import QwenAttentionRouting

def owner(positions,callback=None):
    return QwenAttentionRouting(base.Arithmetic(),(base.P/'candidate_manifest.json').read_bytes(),
        (base.P/'config.json').read_bytes(),positions=positions,progress=callback)

@pytest.mark.parametrize('size',[1,8,16,19,32])
@pytest.mark.parametrize('shifted',[False,True])
def test_scores_to_probability_standin_to_dense_pv(size,shifted):
    positions=tuple((2*i+3) if shifted else i for i in range(size));o=owner(positions)
    rng=np.random.default_rng(500+size);q=rng.normal(size=(size,16,128));k=rng.normal(size=(size,8,128));v=rng.normal(size=(size,8,128))
    keys=[];values=[]
    for t in range(o.layout.tile_count):
        n=o.layout.tile_rows(t)
        ki=base.tile(n,1024,level=26);ki.polynomial.reshape(16,4096)[:n,:1024]=k[16*t:16*t+n].reshape(n,1024)
        keys.append(o.expand_kv(o.rope(ki,tile=t,heads=8),tile=t))
        vi=base.tile(n,1024,level=26);vi.polynomial.reshape(16,4096)[:n,:1024]=v[16*t:16*t+n].reshape(n,1024)
        expanded=o.expand_kv(vi,tile=t)
        # A fresh stand-in retains active expanded V; hostile padding must be
        # discarded even if a caller did not supply canonical zero padding.
        data=expanded.polynomial.copy().reshape(16,4096);data[:,2048:]=-1e6;data[n:]=1e6
        values.append(base.Value(data.ravel(),expanded.level,expanded._ckks_scale))
    for t in range(o.layout.tile_count):
        n=o.layout.tile_rows(t);qi=base.tile(n,2048,level=26);qi.polynomial.reshape(16,4096)[:n,:2048]=q[16*t:16*t+n].reshape(n,2048)
        scores=o.scores(o.rope(qi,tile=t,heads=16),tuple(keys),query_tile=t)
        packed=scores.output.polynomial.reshape(16,4096)
        p=np.zeros((n,16,size));standin=np.full((16,4096),1e6)
        for row,pos in enumerate(o.layout.tile_positions(t)):
            legal=sum(kpos<=pos for kpos in positions)
            for head in range(16):
                # This is explicit test-fixture preparation, not server code or
                # a replacement for the missing encrypted frozen softmax.
                selected=packed[row,32*head:32*head+legal];exp=np.exp(selected-selected.max());prob=exp/exp.sum()
                p[row,head,:legal]=prob;standin[row,32*head:32*head+legal]=prob
        probabilities=base.Value(standin.ravel(),scores.level,math.nextafter(scores.scale,math.inf))
        source=probabilities.polynomial.copy();result=o.attend(probabilities,tuple(values),query_tile=t)
        expected=np.zeros((16,4096));expected[:n,:2048]=np.einsum('rhk,khd->rhd',p,np.repeat(v,2,axis=1)).reshape(n,2048)
        np.testing.assert_allclose(result.output.polynomial,expected.ravel(),rtol=0,atol=5e-14)
        np.testing.assert_array_equal(probabilities.polynomial,source)
        assert result.level==18 and result.query_positions==o.layout.tile_positions(t)
        assert result.counts==dict(rotations=sum(n-int(j%16<n)+7+15+int(j!=0) for j in range(size)),
            weighted_sums=2*size+1,ciphertext_products=size,additions=8*size-1)

def test_bad_value_order_count_and_low_capacity_fail_before_work():
    o=owner(tuple(range(19)));p=base.tile(16,512);v=base.tile(16,2048)
    for values in [(v,),[v,v]]:
        with pytest.raises(ValueError):o.attend(p,values,query_tile=0)
        assert not o.arithmetic.calls
    p.level=4
    with pytest.raises(ValueError):o.attend(p,(v,v),query_tile=0)
    assert not o.arithmetic.calls

@pytest.mark.parametrize('when',['probability','value','final'])
def test_callback_source_metadata_corruption_is_rejected(when):
    held={}
    def cb(label,value,meta):
        if ((when=='probability' and label=='key0_probability_anchors') or
            (when=='value' and label=='key0_value_broadcast') or
            (when=='final' and label=='attended_output_mask')):held['source']._ckks_nsd=True
    o=owner((0,1),cb);held['source']=base.tile(2,512)
    with pytest.raises(ValueError):o.attend(held['source'],(base.tile(2,2048),),query_tile=0)
    assert o.last_run is None and not o._active

def test_current_broadcast_probability_is_protected_across_value_callback():
    held={}
    def cb(label,value,meta):
        if label=='key0_value_broadcast':held['last_add']._ckks_nsd=1.0
    o=owner((0,1),cb);add=o.arithmetic.add
    def recording(x,y):out=add(x,y);held['last_add']=out;return out
    o.arithmetic.add=recording
    with pytest.raises(ValueError):o.attend(base.tile(2,512),(base.tile(2,2048),),query_tile=0)
    assert o.arithmetic.calls['multiply']==0 and o.last_run is None

def test_new_pv_rotation_inventory_is_bound_before_entry():
    o=owner(tuple(range(32)));key=next(k for k in o.required_rotation_indices() if k not in base.QwenHeadRouting.required_rotation_indices(o))
    o.ctx._param_cache.coef_maps.pop(key)
    with pytest.raises(ValueError):o.attend(base.tile(16,512),(base.tile(16,2048),base.tile(16,2048)),query_tile=0)
    assert not o.arithmetic.calls
