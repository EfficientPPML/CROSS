"""New focused fixtures, all computation deferred until an authorized test run.

Only fixed public seed9105 toy weights are generated. Refresh payloads are ideal
identity; bootstrap/RNS/noise are absent. Binary64 metadata is explicitly
synthetic and independently calculated, never evidence from an actual return.
"""
from pathlib import Path
from types import SimpleNamespace
import math

ROOT = Path(__file__).resolve().parent / 'runtime'

# Derived from original evaluation order, not imported from oracle.mapping():
# each layer creates V,Q,K; key transposition prepares key before Q*K prepares Q.
DENSE_REFRESH_KEYS = (
    ('polynomial_before','5:inverse_sqrt'), ('linear_before','6'),
    ('polynomial_before','7:inverse_sqrt'), ('polynomial_before','8:inverse_sqrt'),
    ('polynomial_before','9:inverse_sqrt'), ('rope_after','3'), ('rope_after','2'),
    ('linear_before','10'), ('polynomial_before','12:inverse_sqrt'), ('linear_before','13'),
    ('polynomial_before','14:inverse_sqrt'), ('polynomial_before','15:inverse_sqrt'),
    ('polynomial_before','16:inverse_sqrt'), ('rope_after','5'), ('rope_after','4'),
    ('linear_before','17'), ('polynomial_before','19:inverse_sqrt'), ('linear_before','20'),
    ('polynomial_before','21:inverse_sqrt'),
)
EXPECTED_SELECTED_TRACE_INDICES = (0,31,35,36,48,49,52,56,57,70,71,80,81,89,90,93,94,103,104,107,111,112,124,125,128,132,133,146,147,156,157,165,166,169,170,179,180,183,187,188,200,201,204,208,209,212,215)


def independent_weighted(profile, scale, level):
    q = profile['q_towers']
    divisor = q[2*level] * q[2*level+1]
    # Keep the original binary64 evaluation order; no reassociation/Decimal.
    target = profile['scale'] * divisor
    encoded = target / scale
    product = scale * encoded
    return product / divisor


def meta(value):
    return dict(level=value.level, scale=value._ckks_scale, nsd=1)


def observation(profile, index, bound, before, final_scale):
    def weighted(row, weight):
        return dict(before=dict(row),after=dict(level=row['level']-1,scale=independent_weighted(profile,row['scale'],row['level']),nsd=1),weight=float(weight),bias=0.)
    calls=[]
    if bound != 1.:
        calls.append(weighted(before,1./bound))
    calls.append(weighted(dict(level=26,scale=final_scale,nsd=1),bound/64))
    return dict(index=index,public_bound=float(bound),before=dict(before),weighted_calls=calls,after=dict(calls[-1]['after']))


def different_return_inputs(profile):
    # Bounded deterministic scalar search, evaluated only in the future fixture.
    # It finds two independently computed legal scale outputs, including an ULP
    # deviation from canonical. No call to the candidate enclosure or recurrence.
    choices={}
    for i in range(1,4096):
        value=math.ldexp(1.+i/4096,56)
        returned=independent_weighted(profile,value,26)
        choices.setdefault(returned,value)
        if profile['scale'] in choices and any(x!=profile['scale'] for x in choices):
            return choices[profile['scale']], next(v for k,v in choices.items() if k!=profile['scale'])
    raise AssertionError('fixture must cover a non-nominal independently rounded return')


def fixed_math(np):
    from q5_checkpoint_contract import numerical_plan
    from reference.literal_graph import build,selected
    from reference.dense_reference import references
    plan=numerical_plan();ref=ROOT/'reference';ns=build(np,ref)
    selected(ref/'frozen/cadathlon/agent_pset/p4_qwen/src/reference.py',['toy_weights'],ns)
    config=plan['case']['config'];weights=ns['toy_weights'](config,9105)
    embeddings=weights['model.embed_tokens.weight'][np.asarray(plan['case']['token_ids'],dtype=np.int64)]
    rules=ns['public_toy_approximations']('qwen')
    dense=references(np,config,weights,embeddings,rules)
    assert dense['complete'], 'new independent dense evaluation must complete'
    layout=ns['PackedPairLayout'](2,4,32,65536)
    arithmetic=ns['Arithmetic'](plan['raw_profile'],{})
    # Exact original _qwen phase labels, independent of expected trace indices.
    # BoundedArithmetic also records refresh-start/complete phases; those
    # are not the diagnostic layer ends and must not be exported twice.
    layer_phases={f'packed Qwen layer {layer}: {phase} ready'
                  for layer in range(config['num_hidden_layers'])for phase in ('attention','MLP')}
    captures=[];original_record=arithmetic.record
    def record(kind,label,value,**extra):
        result=original_record(kind,label,value,**extra)
        # Retain only the47 new targets, not every full-size intermediate.
        if kind in ('input','output','refresh_before','refresh_after') or (kind=='phase' and label in layer_phases) or (kind=='normalize_after' and str(label)=='6'):
            captures.append((len(arithmetic.trace)-1,np.array(value.slots,copy=True)))
        return result
    arithmetic.record=record;observations=[]
    final_inputs=different_return_inputs(plan['profile'])
    def refresh(value,*,public_bound):
        index=len(observations)
        record('refresh_before',str(index),value,public_bound=public_bound)
        observed=observation(plan['profile'],index,public_bound,meta(value),final_inputs[index%2])
        observations.append(observed)
        # Synthetic actual-return metadata, ideal identity slots. No bootstrap.
        result=ns['Value'](value.slots.copy(),observed['after']['level'],observed['after']['scale'])
        return record('refresh_after',str(index),result,public_bound=public_bound)
    arithmetic.refresher.refresh=refresh
    graph=ns['Graph'](layout,arithmetic)
    initial=ns['Value'](layout.pack_client_embeddings(embeddings)[0],49,plan['profile']['scale'])
    record('input','original_embeddings',initial)
    result=graph.evaluate(initial,'qwen',config,weights,rules)
    record('output','logits',result)
    rows=[]
    for index,(trace_index,_) in enumerate(captures):
        trace=arithmetic.trace[trace_index]
        row={k:trace[k]for k in ('kind','label','stack','level','scale','nsd')}
        row.update(index=index,trace_index=trace_index)
        if trace['kind'].startswith('refresh_'):row['public_bound']=trace['public_bound']
        rows.append(row)
    return SimpleNamespace(np=np,plan=plan,reference_root=ref,weights=weights,embeddings=embeddings,rules=rules,dense=dense,arithmetic=arithmetic,captures=captures,rows=rows,observations=observations,final_inputs=final_inputs,result=result)


class RecordingArray:
    """Metadata/readiness stand-in, no RNS array allocation or contents."""
    dtype='uint32'
    def __init__(self,n):self.shape=(1,2,256,512,n);self.barriers=0
    def block_until_ready(self):self.barriers+=1;return self


class RecordingCiphertext:
    def __init__(self,profile,level=49,scale=None):
        self.batch=1;self.num_elements=2;self.degree=131072;self.precision=32
        self.r=256;self.c=512;self.degree_layout=(256,512);self._ckks_nsd=1
        self.num_moduli=2*(level+1);self.moduli=tuple(profile['q_towers'][:self.num_moduli])
        self.polynomial=RecordingArray(self.num_moduli);self.moduli_array=object();self.ntt_ctx=object()
        self._ckks_scale=profile['scale'] if scale is None else scale
