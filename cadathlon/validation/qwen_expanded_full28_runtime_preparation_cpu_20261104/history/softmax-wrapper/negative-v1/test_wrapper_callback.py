"""Specific unbatched progress-callback wrapper mutations; synthetic slots only."""
from pathlib import Path
import gc, hashlib, json, os, sys, weakref
import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
SOURCE=Path(os.environ.get('SOFTMAX_REVIEW_SOURCE',str(HERE/'source'))).resolve()
sys.path[:0]=[str(SOURCE),str(SOURCE/'checkout/cadathlon/common')]
from ideal_primitives import Arithmetic, Value
from ideal_refresh import IdentityRefresh
from encrypted_qwen_causal_softmax import QwenCausalSoftmax, QwenCausalSoftmaxContract

RAW=(SOURCE/'inputs/candidate_manifest.json').read_bytes()
CERT=(SOURCE/'inputs/certificates.json').read_bytes()
CONFIG=(SOURCE/'inputs/config.json').read_bytes()
KEYS=json.loads((SOURCE/'inputs/actual_key_plan.json').read_bytes())['combined_keys']

def construct(callback=None):
    a=Arithmetic();cache=a.ctx._param_cache
    cache.raw_rotation_keys={k:{'a':object(),'b':object()}for k in KEYS}
    cache.coef_maps={k:object()for k in KEYS}
    refresh=IdentityRefresh(a)
    owner=QwenCausalSoftmax(a,QwenCausalSoftmaxContract(RAW,CERT,3),CONFIG,positions=(5,19),refresher=refresh,progress=callback)
    # K2 lives in row1; all unused rows, features and future keys are hostile.
    data=np.full((16,4096),29.)
    for h in range(16):data[1,32*h:32*h+2]=[-.37+.01*h,.52+.01*h]
    return a,refresh,owner,Value(data.ravel(),21)

@pytest.mark.parametrize('method',['refresh','check'])
def test_callback_wrapper_replacement_rejected_before_invocation(method):
    holder={};calls=[];observed={};payload_refs=[]
    def callback(label,value,metadata):
        if label!='K2.normalized_t.refresh':return
        owner=holder['owner'];binding=owner.refresh_binding
        observed['replaced_at']=label;observed['refreshes_at_replacement']=len(holder['refresh'].calls)
        payload_refs.append(weakref.ref(value.polynomial))
        if method=='refresh':
            def replacement(source,*,public_bound):
                calls.append(dict(method='refresh',bound=public_bound,source_level=source.level))
                return value # Valid callback-provided L25 metadata, wrong later mathematical value.
        else:
            def replacement():calls.append(dict(method='check'))
        setattr(binding,method,replacement)
    a,r,o,source=construct(callback);holder.update(owner=o,refresh=r)
    source_payload=source.polynomial;original=source_payload.copy();caught=None;result=None
    try:result=o.evaluate_tile(source,query_tile=0)
    except BaseException as error:caught=error
    observed.update(rejected=isinstance(caught,ValueError),error_type=type(caught).__name__if caught else None,
        replacement_invocations=len(calls),actual_refresh_calls=len(r.calls),normal_return=result is not None,
        active_cleared=o._active is False,last_run_clear=o.last_run is None,
        source_payload_retained=source.polynomial is source_payload,source_unchanged=bool(np.array_equal(source.polynomial,original)))
    (HERE/f'{method}-observation.json').write_text(json.dumps(observed,indent=2)+'\n')
    assert isinstance(caught,ValueError), 'callback wrapper mutation was not rejected by the actual unbatched owner'
    assert not calls, 'a substituted wrapper accessor ran before identity rejection'
    assert len(r.calls)==observed['refreshes_at_replacement']==1
    assert not o._active and o.last_run is None and source.polynomial is source_payload
    np.testing.assert_array_equal(source.polynomial,original)
    gc.collect();assert payload_refs[0]()is not None

def test_clean_owner_control():
    _,r,o,source=construct();result=o.evaluate_tile(source,query_tile=0)
    assert result.output.level==24 and np.isfinite(result.output.polynomial).all()
    assert result.query_positions==(5,19)and result.key_positions==(5,19)and result.direct_value_queries==(0,)
    view=result.output.polynomial.reshape(16,4096)
    assert np.count_nonzero(view[0])==0 and np.count_nonzero(view[2:])==0 and np.count_nonzero(view[1,512:])==0
    assert all(np.count_nonzero(view[1,32*h+2:32*h+32])==0 for h in range(16))
    assert len(r.calls)==3+2*o.contract.squarings and not o._active and o.last_run is not None

def test_genuine_callback_primary_and_retained_payload_control():
    error=RuntimeError('synthetic genuine body primary');refs=[]
    def callback(label,value,metadata):
        if label=='K2.normalized_t.refresh':
            refs.append(weakref.ref(value.polynomial));raise error
    _,r,o,source=construct(callback);original=source.polynomial
    with pytest.raises(RuntimeError)as caught:o.evaluate_tile(source,query_tile=0)
    assert caught.value is error and not o._active and o.last_run is None and len(r.calls)==1
    assert source.polynomial is original
    gc.collect();assert refs[0]()is not None # Retained error traceback owns the in-flight source/output.
