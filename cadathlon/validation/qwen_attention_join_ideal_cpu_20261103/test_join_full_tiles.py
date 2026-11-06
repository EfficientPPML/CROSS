"""Predeclared full-tile genuine-weight joins and bounded callback retention."""
import numpy as np
import pytest
from test_join import build,inputs,expected_counts
from join_ideal import dense_reference
import encrypted_qwen_attention_join as runtime
from ideal_metrics import record

@pytest.fixture(autouse=True)
def clear_test_only_quarantine():
    runtime._QUARANTINE.clear()
    yield
    runtime._QUARANTINE.clear()

@pytest.mark.parametrize('sequence',[16,32])
def test_genuine_pinned_full_tiles(monkeypatch,sequence):
    positions=tuple(2*i for i in range(sequence));holder={};peaks={}
    def progress(label,value,metadata):
        owner=holder['owner'];phase=owner._phase
        peaks[phase]=max(peaks.get(phase,0),len(owner._phase_refs))
        if phase.endswith(('.softmax','.pv')):assert len(owner._phase_refs)<=1
    owner,a,refresh,w,g=build(monkeypatch,positions,progress=progress);holder['owner']=owner
    rows,tiles=inputs(positions);expected,domains=dense_reference(rows,w,g,0,positions)
    assert all(r['passed'] for r in domains),domains
    # Perturb only invalid physical lanes of identity-refresh stand-ins. This
    # exercises real public masks, not a guarantee for actual refresh noise.
    def padding(source,result):result.polynomial.reshape(16,4096)[:,2048:]=.125
    refresh.inject=padding
    result=owner.evaluate(tiles)
    worst=0.
    for tile,value in enumerate(result.outputs):
        wanted=np.zeros((16,4096));wanted[:,:2048]=expected[tile*16:tile*16+16]
        np.testing.assert_allclose(value.polynomial,wanted.ravel(),atol=1e-8,rtol=0.)
        worst=max(worst,float(np.max(np.abs(value.polynomial-wanted.ravel()))))
        assert value.level==20
    assert a.calls==expected_counts(owner)
    assert all(size<=max(n.contract.degree for n in owner.norms)+6 for phase,size in peaks.items() if phase.endswith('_norm'))
    assert not owner._phase_refs and not owner._refs
    record(layer=0,sequence=sequence,positions=positions,domains=domains,max_abs_error=worst,
        counts=a.calls,metadata=result.metadata,refreshes=result.refresh_events,padding_probe=True)
