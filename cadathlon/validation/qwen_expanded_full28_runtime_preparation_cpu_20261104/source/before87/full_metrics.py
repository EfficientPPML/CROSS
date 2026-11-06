"""Fail-closed finite all-slot ideal diagnostics, never payload serialization."""
import numpy as np

def check_logits(blocks,expected,*,gate=1e-8):
    if type(gate)is not float or gate!=1e-8:raise ValueError('fixed ideal gate required')
    result=dict(passed=False,gate=gate,finite=False,shape_valid=False,physical_slots_checked=0,max_abs_error=None,all_logits=151936,predicted_token=None,reference_token=None)
    if type(blocks)is not tuple or len(blocks)!=38 or type(expected)is not np.ndarray or expected.dtype!=np.dtype('float64') or expected.shape!=(151936,):return result
    if any(type(x)is not np.ndarray or x.dtype!=np.dtype('float64') or x.shape!=(65536,) for x in blocks):return result
    result['shape_valid']=True;result['physical_slots_checked']=38*65536
    if not np.isfinite(expected).all() or not all(np.isfinite(x).all() for x in blocks):return result
    result['finite']=True;errors=[];logits=[]
    for i,x in enumerate(blocks):
        start=i*4096;stop=min(start+4096,151936);oracle=np.zeros(65536);oracle[:stop-start]=expected[start:stop];logits.append(x[:stop-start])
        with np.errstate(over='ignore',invalid='ignore'):error=np.max(np.abs(x-oracle))
        if not np.isfinite(error):return result
        errors.append(float(error))
    result.update(max_abs_error=max(errors),predicted_token=int(np.argmax(np.concatenate(logits))),reference_token=int(np.argmax(expected)))
    result['passed']=result['max_abs_error']<=gate
    return result
