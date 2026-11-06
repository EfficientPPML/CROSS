"""Observed binary64 metadata recurrence; no ciphertext/noise certificate.

The actual final refresh alignment is observed at the original evaluator call.
Client replay uses all19 observed returns, never the clear nominal return scale.
"""
from fractions import Fraction
from functools import lru_cache
import math,sys
from q5_checkpoint_contract import numerical_plan,positive,metadata_row,all_checkpoints,canonical

@lru_cache(maxsize=1)
def constants():
    p=numerical_plan()['profile'];return tuple(p['q_towers']),p['scale']

def divisor(level):
    if type(level)is not int or not 1<=level<=49:raise ValueError('source level1..49 required')
    return math.prod(constants()[0][2*level:2*(level+1)])

def weighted_scale(scale,level):
    scale=positive(scale);D=divisor(level);target=constants()[1]*D
    return positive(positive(scale*positive(target/scale))/D)

def product_scale(left,right,level):
    return positive(positive(positive(left)*positive(right))/divisor(level))

def refresh_enclosure(observed_input_level):
    # Original49 chain/bootstrap depth plus one normalization rescale leads to
    #26 BEFORE the last original weighted alignment. Observe, do not assign it.
    if type(observed_input_level)is not int or observed_input_level!=26:raise ValueError('observed refresh alignment input26 required')
    if(sys.float_info.radix,sys.float_info.mant_dig,sys.float_info.rounds)!=(2,53,1):raise ValueError('nearest binary64 required')
    D=divisor(observed_input_level);target=positive(constants()[1]*D)
    T=Fraction(target);u=Fraction(1,2**53);lo=T*(1-u)**2;hi=T*(1+u)**2
    value=float(lo)
    while Fraction(value)<lo:value=math.nextafter(value,math.inf)
    products=[]
    while Fraction(value)<=hi:
        products.append(value)
        if len(products)>16:raise ValueError('unexpectedly broad finite enclosure')
        value=math.nextafter(value,math.inf)
    return {'input_level':observed_input_level,'output_level':25,'divisor':D,'target':target,
            'accepted_scales':sorted({x/D for x in products}),
            'scope':'Conditional binary64 enclosure only; no CKKS, bootstrap-accuracy or reachability claim.'}

def check_alignment(observation):
    if type(observation)is not dict or set(observation)!={'before','after','weight','bias'}:raise ValueError('closed observed weighted call required')
    before=metadata_row(observation['before']);after=metadata_row(observation['after'])
    if type(observation['weight'])is not float or type(observation['bias'])is not float or observation['bias']!=0.:raise ValueError('original scalar weighted call required')
    level=before['level'];want=weighted_scale(before['scale'],level)
    if after!={'level':level-1,'scale':want,'nsd':1}:raise ValueError('actual weighted metadata recurrence differs')
    return observation

def check_refresh(observation,index):
    if type(observation)is not dict or set(observation)!={'index','public_bound','before','weighted_calls','after'}:raise ValueError('closed refresh observation required')
    expected=[r for r in numerical_plan()['checkpoints'] if r['kind']=='refresh_before'][index]
    if type(observation['index'])is not int or observation['index']!=index or type(observation['public_bound'])is not float or observation['public_bound']!=expected['public_bound']:raise ValueError('original ordered refresh bound required')
    before=metadata_row(observation['before']);after=metadata_row(observation['after']);calls=observation['weighted_calls'];bound=observation['public_bound']
    if before['level']!=expected['planned_level'] or type(calls)is not list or len(calls)!=(1 if bound==1. else 2):raise ValueError('actual refresh input/count differs')
    for call in calls:check_alignment(call)
    if bound!=1. and (calls[0]['before']!=before or calls[0]['weight']!=1./bound):raise ValueError('actual original preconditioning differs')
    final=calls[-1]
    if final['weight']!=bound/64 or final['after']!=after:raise ValueError('original64-alias bound restoration differs')
    certificate=refresh_enclosure(final['before']['level']);S=final['before']['scale'];D=certificate['divisor'];T=certificate['target']
    encoded=positive(T/S);product=positive(S*encoded);output=positive(product/D)
    # The relative-rounding enclosure is used ONLY after actual observed
    #intermediate normality/finite checks; no universal scale assumption.
    if any(x<sys.float_info.min for x in (encoded,product,output)):raise ValueError('observed alignment normality precondition failed')
    if after['level']!=25 or after['scale']!=output or after['scale']not in certificate['accepted_scales']:raise ValueError('actual refresh return outside original enclosure')
    return after

def check_refreshes(observations,rows):
    all_checkpoints(rows)
    if type(observations)is not list or len(observations)!=19:raise ValueError('complete ordered19 actual refresh observations required')
    before=[r for r in rows if r['kind']=='refresh_before'];after=[r for r in rows if r['kind']=='refresh_after']
    for index,obs in enumerate(observations):
        actual=check_refresh(obs,index)
        if obs['before']!={k:before[index][k]for k in ('level','scale','nsd')} or actual!={k:after[index][k]for k in ('level','scale','nsd')}:raise ValueError('export/observed refresh metadata differs')
    return tuple(obs['after']['scale']for obs in observations)

def client_replay(np,reference_root,weights,embeddings,rows,observations):
    """Client-only original graph over plaintext metadata doubles.

    Caller authenticates all47 artifacts and terminal proof BEFORE this call;
    this function grants no private access and reads no ciphertext body.
    Dense numerical independence is separately provided by q5_checkpoint_oracle.
    """
    scales=check_refreshes(observations,rows);plan=numerical_plan()
    from reference.literal_graph import build
    ns=build(np,reference_root);layout=ns['PackedPairLayout'](2,4,32,65536)
    arithmetic=ns['Arithmetic'](plan['raw_profile'],{});position=0
    def actual_refresh(value,*,public_bound):
        nonlocal position
        if position>=19:raise ValueError('additional replay refresh')
        observation=observations[position]
        if public_bound!=observation['public_bound'] or {'level':value.level,'scale':value._ckks_scale,'nsd':value._ckks_nsd}!=observation['before']:raise ValueError('independent actual-input metadata replay differs')
        label=str(position);arithmetic.record('refresh_before',label,value,public_bound=public_bound)
        result=ns['Value'](value.slots.copy(),observation['after']['level'],scales[position]);position+=1
        return arithmetic.record('refresh_after',label,result,public_bound=public_bound)
    arithmetic.refresher.refresh=actual_refresh
    graph=ns['Graph'](layout,arithmetic);rules=ns['public_toy_approximations']('qwen')
    value=ns['Value'](layout.pack_client_embeddings(embeddings)[0],49,plan['profile']['scale'])
    arithmetic.record('input','original_embeddings',value)
    result=graph.evaluate(value,'qwen',plan['case']['config'],weights,rules)
    arithmetic.record('output','logits',result)
    if position!=19 or len(arithmetic.trace)!=216 or arithmetic.counts!=plan['ideal_counts'] or arithmetic.counter!=plan['semantic_counts']:raise ValueError('complete original metadata graph required')
    for expected,actual in zip(plan['full_trace'],arithmetic.trace,strict=True):
        for key in ('index','kind','label','stack','level','nsd'):
            if canonical(actual[key])!=canonical(expected[key]):raise ValueError('original replay event order differs')
    for row in rows:
        actual=arithmetic.trace[row['trace_index']]
        for key in ('kind','label','stack','level','scale','nsd'):
            if canonical(actual[key])!=canonical(row[key]):raise ValueError('independent actual metadata replay differs: '+key)
    return {'complete':True,'checkpoints':47,'refresh_returns':19,'primitive_counts':arithmetic.counts,
            'actual_scales_consumed':True,'nominal_refresh_substitution':False,'ciphertext_accuracy_claim':False}
