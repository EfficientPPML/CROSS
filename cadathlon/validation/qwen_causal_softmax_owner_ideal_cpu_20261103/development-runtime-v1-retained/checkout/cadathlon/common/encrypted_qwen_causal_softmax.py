"""Experimental literal candidate3 causal softmax with public row branches.

Refresh and nonlinear domains are conditional caller contracts. No range
inspection, codec, key generation, context ownership or coefficient selection.
K1 is a typed direct-V request, not a fabricated probability ciphertext.
"""
from dataclasses import dataclass,field
import hashlib,json,math
import numpy as np
from encrypted_qwen_head_routing import QwenHeadRouting,typed,MANIFEST,SLOTS
from encrypted_qwen_rms_refresh import QwenInverseRefreshBinding,accessor
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale

CERTIFICATE='ad386cb499682a0abb54f81a26d67136b2df0da26383b4b140184344de904a37'
DELTA=1.1467305003699405e18
Q_SHA='fc2b841869ef2b2b556209323ed033c5a177344d1de395fc94e8a5865df39a49'
P_SHA='c03a649ab504238a4f24d57d7fe8e49f2a445c5d842579494cd243a7043d1d49'
def digest(raw):return hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class QwenCausalSoftmaxContract:
    manifest_bytes:bytes=field(repr=False)
    certificate_bytes:bytes=field(repr=False)
    layer:int
    centered_domain:tuple=field(init=False)
    squarings:int=field(init=False)
    coefficients:tuple=field(init=False)
    initial_rule:tuple=field(init=False)
    key_rules:tuple=field(init=False)
    events:tuple=field(init=False)
    def __post_init__(self):
        if type(self.manifest_bytes)is not bytes or digest(self.manifest_bytes)!=MANIFEST:raise ValueError('exact candidate3 bytes required')
        if type(self.certificate_bytes)is not bytes or digest(self.certificate_bytes)!=CERTIFICATE:raise ValueError('exact conditional public certificate required')
        if type(self.layer)is not int or not 0<=self.layer<28:raise ValueError('typed frozen layer0..27 required')
        m=json.loads(self.manifest_bytes);certificate=json.loads(self.certificate_bytes);site=m['softmax'][str(self.layer)]
        if certificate['manifest_sha256']!=MANIFEST or certificate['passed']is not True:raise ValueError('certificate binding differs')
        exp=m['exp']
        if exp['strategy']!='power' or exp['degree']!=8 or exp['domain']!=[-.5,.5]:raise ValueError('exact exp8 power required')
        def rule(row):return (tuple(row['domain']),row['rounded_first_weight'],row['rounded_first_bias'],row['iterations'])
        initial=rule(m['initial_reciprocal']);keys=[]
        for k in range(1,33):
            record=certificate['softmax_K'][k-1]
            if record['K']!=k or not record['existing_inverse_refresh_bound_certified']:raise ValueError('K certificate differs')
            bound=record['existing_inverse_refresh_bound'];keys.append((*rule(m['reciprocal_by_keys'][str(k)]),float(bound) if bound is not None else 0.))
        if initial[-1]!=6 or site['squarings']not in (5,6):raise ValueError('original counts required')
        # The inherited validator consumes the bound field; actual locations
        # and entry levels are enforced by this literal owner at every event.
        events=(('normalized_t',2.,18),('completed_initial',4.,3),('probability',2.,2),
            *((f'K{k}',keys[k-1][-1],6) for k in range(2,33)))
        for name,value in dict(centered_domain=tuple(site['centered_domain']),squarings=site['squarings'],
            coefficients=tuple(exp['evaluation_coefficients']),initial_rule=initial,key_rules=tuple(keys),events=events).items():object.__setattr__(self,name,value)

FIELDS=tuple(QwenCausalSoftmaxContract.__dataclass_fields__)
def contract_state(c):return tuple(typed(getattr(c,n,None)) for n in FIELDS)
def verify_contract(c):
    if not isinstance(c,QwenCausalSoftmaxContract):raise TypeError('frozen causal-softmax contract required')
    expected=QwenCausalSoftmaxContract(c.manifest_bytes,c.certificate_bytes,c.layer)
    if contract_state(c)!=contract_state(expected):raise ValueError('typed causal-softmax contract changed')
    return contract_state(c)

@dataclass(frozen=True)
class QwenProbabilityTile:
    output:object
    query_positions:tuple
    key_positions:tuple
    direct_value_queries:tuple
    query_tile:int
    layer:int
    counts:dict
    checkpoints:dict
    plan:dict

class QwenCausalSoftmax(QwenHeadRouting):
    def __init__(self,arithmetic,contract,config_bytes,*,positions,refresher=None,progress=None):
        self.contract=contract;state=verify_contract(contract);self.refresher=refresher
        super().__init__(arithmetic,contract.manifest_bytes,config_bytes,positions=positions,progress=progress)
        self._softmax_bound=(contract,refresher);self._contract=state;self._arithmetic_accessors=self._accessors()
        if len(self.layout.positions)==1:
            if refresher is not None:raise ValueError('K1 direct-V branch has no refresher')
            self.refresh_binding=None
        else:self.refresh_binding=QwenInverseRefreshBinding(self,refresher)
        self._refresh_binding=self.refresh_binding;self._softmax_guard()
    def required_rotation_indices(self):
        return () if len(self.layout.positions)==1 else tuple(sorted({s%SLOTS for b in range(5) for s in (1<<b,-(1<<b))}))
    def _accessors(self):
        a=self.arithmetic
        return (*tuple(accessor(a,n) for n in ('rotate','weighted_sum','multiply','add','_level')),
            accessor(a.rotation,'rotate'),accessor(a._plaintext_multiply,'multiply'),accessor(a._rescale,'rescale'))
    def _softmax_guard(self):
        if (any(a is not b for a,b in zip((self.contract,self.refresher),self._softmax_bound)) or
            contract_state(self.contract)!=self._contract or self.refresh_binding is not self._refresh_binding):raise ValueError('construction-bound softmax changed')
        super()._guard()
        if any(a is not b for now,old in zip(self._accessors(),self._arithmetic_accessors) for a,b in zip(now,old)):raise ValueError('arithmetic accessor changed')
        c=self.ctx;k=c._param_cache
        if (typed(c.max_level)!=typed(49) or typed(c.dnum)!=typed(8) or typed(c.scaling_factor)!=typed(DELTA) or
            digest(json.dumps(list(k.q_towers)).encode())!=Q_SHA or digest(json.dumps(list(k.p_towers)).encode())!=P_SHA):raise ValueError('fixed common p60/d8 profile required')
        if self.refresh_binding is not None:self.refresh_binding.check()
    def evaluate_tile(self,scores,*,query_tile,capture_checkpoints=False):
        self._softmax_guard()
        if self._active:raise ValueError('serial softmax cannot be reentered')
        if type(capture_checkpoints)is not bool:raise TypeError('exact checkpoint bool required')
        positions=self.layout.tile_positions(query_tile);a=self.arithmetic;c=self.contract
        if len(self.layout.positions)==1:
            if scores is not None:raise ValueError('K1 bypass requires no score ciphertext')
            plan=dict(stages=[],refreshes=[],counts={},output_level=None,output_scale=None)
            self.last_run=plan
            return QwenProbabilityTile(None,positions,self.layout.positions,(0,),query_tile,c.layer,{}, {},plan)
        self._validate(scores,0)
        if a._level(scores)!=21:raise ValueError('actual score input level21 required')
        q=tuple(self.ctx._param_cache.q_towers);refs={};checkpoints={};stages=[];events=[]
        counts=dict(rotations=0,ordinary_additions=0,weighted_sums=0,ciphertext_products=0,refreshes=0)
        def retain(name,value):refs[name]=(value,value.polynomial,self._fingerprint(value));return value
        def state(value):return (value,value.polynomial,self._fingerprint(value))
        def check_states(values):
            if any(v.polynomial is not payload or self._fingerprint(v)!=old for v,payload,old in values):raise ValueError('retained softmax ciphertext changed')
        retain('scores',scores)
        def guard():self._softmax_guard();check_states(tuple(refs.values()))
        def metadata(value,level,scale):
            self._validate(value,0)
            if (type(a._level(value))is not int or a._level(value)!=level or typed(value._ckks_scale)!=typed(scale) or
                type(value.num_moduli)is not int or value.num_moduli!=2*(level+1) or typed(tuple(value.moduli))!=typed(q[:2*(level+1)])):raise ValueError('softmax primitive metadata differs')
        def note(label,value,level,scale,kind,owners=()):
            guard();check_states(owners);metadata(value,level,scale);self._forget(value);current=state(value)
            if self.progress is not None:self.progress(label,value,dict(level=level,scale=scale,counts=dict(counts)))
            guard();check_states((*owners,current));self._forget(value)
            for prior,_,_ in refs.values():
                if prior is not scores:self._forget(prior)
            stages.append(dict(label=label,kind=kind,level=level,scale=scale))
            if capture_checkpoints:checkpoints[label]=retain('checkpoint.'+label,value)
            return value
        def weighted(label,values,weights,bias=0.):
            guard();owners=tuple(state(v) for v in values);level=min(a._level(v) for v in values)
            scale=weighted_sum_scale([v._ckks_scale for v in values],DELTA,drop_divisor(q,level))['output_scale'];counts['weighted_sums']+=1
            return note(label,a.weighted_sum(values,weights,bias=bias),level-1,scale,'weighted',owners)
        def mul(label,x,y):
            guard();owners=(state(x),state(y));level=min(a._level(x),a._level(y));scale=multiply_scale(x._ckks_scale,y._ckks_scale,drop_divisor(q,level));counts['ciphertext_products']+=1
            return note(label,a.multiply(x,y),level-1,scale,'multiply',owners)
        def rotate(value,shift):
            guard();old=state(value);level=a._level(value);scale=value._ckks_scale;counts['rotations']+=1
            out=a.rotate(value,shift%SLOTS);metadata(out,level,scale);guard();check_states((old,));return out
        def add(x,y,subtract=False):
            guard();owners=(state(x),state(y));level=min(a._level(x),a._level(y));scale=x._ckks_scale;counts['ordinary_additions']+=1
            out=a.add(x,y,subtract=subtract);metadata(out,level,scale);guard();check_states(owners);return out
        def reduce(value):
            for bit in range(5):value=add(value,rotate(value,1<<bit))
            return value
        def broadcast(value):
            for bit in range(5):value=add(value,rotate(value,-(1<<bit)))
            return value
        def refresh(label,value,bound,expected_level):
            guard()
            if a._level(value)!=expected_level:raise ValueError('frozen refresh entry level differs')
            retain('refresh_source',value);counts['refreshes']+=1
            out=self.refresh_binding.refresh(value,public_bound=float(bound))
            out=note(label,out,25,positive_scale(out._ckks_scale),'refresh',(state(value),))
            events.append(dict(label=label,input_level=expected_level,output_level=25,actual_output_scale=out._ckks_scale,public_bound=float(bound)))
            refs.pop('refresh_source');return out
        def inverse(prefix,denominator,rule,bound=None):
            _,w,b,n=rule;retain('denominator',denominator)
            y=retain('inverse',weighted(prefix+'.first',[denominator],[w],b))
            for update in range(2,n+1):
                if bound is not None and update==7:y=retain('inverse',refresh(prefix+'.refresh_before7',y,bound,6))
                product=mul(prefix+f'.{update}.product',denominator,y)
                correction=weighted(prefix+f'.{update}.correction',[product],[-1.],2.)
                y=retain('inverse',mul(prefix+f'.{update}.update',y,correction))
            return y
        self._active=True;self.last_run=None
        try:
            aggregate=None
            for row in range(len(positions)):
                k=query_tile*16+row+1
                if k==1:continue
                prefix=f'K{k}';mask=np.zeros((16,4096));anchors=np.zeros((16,4096));dummy=np.zeros((16,4096));dummy[:,::32]=1.
                for head in range(16):mask[row,32*head:32*head+k]=1.;anchors[row,32*head]=1.;dummy[row,32*head]=0.
                mask=mask.ravel();anchors=anchors.ravel();dummy=dummy.ravel();alpha=float(1/k)
                masked=retain('masked_source',weighted(prefix+'.source_branch_mask',[scores],[mask]))
                mean=retain('mean',broadcast(weighted(prefix+'.mean_anchor_alpha',[reduce(masked)],[anchors*alpha])))
                centered=add(masked,mean,True)
                t=retain('mapped_t',weighted(prefix+'.normalized_t_branch_mask',[centered],[mask*2.**(1-c.squarings)]))
                t=retain('mapped_t',refresh(prefix+'.normalized_t.refresh',t,2.,18));terms={1:t}
                def power(n):
                    if n not in terms:terms[n]=retain('power'+str(n),mul(prefix+'.power'+str(n),power(n//2),power((n+1)//2)))
                    return terms[n]
                base=retain('base',weighted(prefix+'.base_exp',[power(n) for n in range(1,9)],c.coefficients[1:],c.coefficients[0]))
                for name in tuple(refs):
                    if name.startswith('power') or name in ('mapped_t','mean','masked_source'):refs.pop(name)
                terms.clear()
                masked_exp=weighted(prefix+'.initial_mean_mask_alpha',[base],[mask*alpha])
                denominator=broadcast(weighted(prefix+'.initial_mean_anchor_dummy1',[reduce(masked_exp)],[anchors],dummy))
                y=inverse(prefix+'.initial_inverse',denominator,c.initial_rule)
                y=retain('inverse',refresh(prefix+'.completed_initial_inverse.refresh',y,4.,3))
                probability=retain('probability',weighted(prefix+'.initial_probability_alpha_mask',[mul(prefix+'.initial_quotient',base,y)],[mask*alpha]))
                refs.pop('base');refs.pop('denominator');refs.pop('inverse')
                for iteration in range(c.squarings):
                    tag=prefix+f'.round{iteration}'
                    probability=retain('probability',refresh(tag+'.probability.refresh',probability,2.,a._level(probability)))
                    masked_probability=weighted(tag+'.mask',[probability],[mask])
                    numerator=retain('numerator',mul(tag+'.square',masked_probability,masked_probability))
                    denominator=broadcast(weighted(tag+'.denominator_anchor_dummy1',[reduce(numerator)],[anchors],dummy))
                    krule=c.key_rules[k-1];y=inverse(tag+'.inverse',denominator,krule[:4],krule[4])
                    probability=retain('probability',mul(tag+'.quotient',numerator,y))
                    for name in ('numerator','denominator','inverse'):refs.pop(name)
                probability=retain('probability',refresh(prefix+'.final_probability.refresh',probability,2.,a._level(probability)))
                branch=weighted(prefix+'.final_branch_mask',[probability],[mask])
                aggregate=retain('aggregate',branch if aggregate is None else add(aggregate,branch));refs.pop('probability');guard()
            output=note('output',aggregate,24,aggregate._ckks_scale,'assembled');guard()
            plan=dict(stages=stages,refreshes=events,counts=dict(counts),input_level=21,input_scale=scores._ckks_scale,output_level=24,output_scale=output._ckks_scale)
            self.last_run=plan
            return QwenProbabilityTile(output,positions,self.layout.positions,(0,) if query_tile==0 else (),query_tile,c.layer,dict(counts),checkpoints,plan)
        finally:self._active=False
