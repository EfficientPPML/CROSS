"""Frozen BERT-v2 pooler tanh over genuine static arithmetic and explicit refresh.

The caller supplies values in [-34,34]. B2 refresh and polynomial/reciprocal
accuracy remain conditional obligations, not ciphertext range inspections.
"""
from dataclasses import dataclass,field
import hashlib,json,math
import numpy as np
from encrypted_factored_score_routing import FactoredScoreLayout,FactoredScoreRouting
from encrypted_fullslot_refresh_binding import FullSlotRefreshBinding,typed
from encrypted_integer_affine import integer_affine,backward_scale_plan
from encrypted_scaled_newton import integer_newton_updates
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale,static_newton_forward
MANIFEST_SHA256='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
COEFFICIENT_SHA256='a107f176bf28ea6e8f80af0613687c02ed6478078395b67d94b5e04b077204a7'
@dataclass(frozen=True)
class BertTanhContract:
    manifest_bytes:bytes=field(repr=False)
    domain:tuple=field(init=False)
    coefficients:tuple=field(init=False)
    power_coefficients:tuple=field(init=False)
    seed:float=field(init=False)
    def __post_init__(self):
        if type(self.manifest_bytes)is not bytes or hashlib.sha256(self.manifest_bytes).hexdigest()!=MANIFEST_SHA256:raise ValueError('exact frozen BERT-v2 manifest required')
        m=json.loads(self.manifest_bytes);r=m['tanh'];p=r['base_polynomial'];inv=m['tanh_reciprocal']
        c=np.asarray(p['coefficients'],dtype='<f8')
        if (r['domain']!=[-34,34] or r['base_degree']!=12 or r['doublings']!=6 or r['reciprocal_iterations']!=5
                or p['domain']!=[-1.,1.] or c.shape!=(13,) or hashlib.sha256(c.tobytes()).hexdigest()!=COEFFICIENT_SHA256
                or inv['domain']!=[1.,2.1] or inv['iterations']!=5 or inv['seed']!=.6451612903225806):raise ValueError('frozen tanh circuit differs')
        object.__setattr__(self,'domain',(-34.,34.));object.__setattr__(self,'coefficients',tuple(map(float,c)))
        object.__setattr__(self,'power_coefficients',tuple(map(float,np.polynomial.chebyshev.cheb2poly(c))))
        object.__setattr__(self,'seed',inv['seed'])
def contract_state(c):return tuple(typed(getattr(c,n,None)) for n in ('manifest_bytes','domain','coefficients','power_coefficients','seed'))
def tanh_scale_plan(q_moduli,*,input_level,input_scale,canonical_scale,contract,refresh_scales=None):
    if not isinstance(contract,BertTanhContract) or contract_state(contract)!=contract_state(BertTanhContract(contract.manifest_bytes)):raise ValueError('frozen tanh contract changed')
    q=tuple(q_moduli)
    if (len(q)!=100 or len(set(q))!=100 or any(type(x)is not int or x<=1 for x in q)
            or type(input_level)is not int or not 8<=input_level<=49):raise ValueError('common depth49 profile and eight entry levels required')
    delta=positive_scale(canonical_scale);source=positive_scale(input_scale);stages=[];events=[]
    refresh_scales={} if refresh_scales is None else dict(refresh_scales)
    if any(type(i)is not int or not 1<=i<=6 for i in refresh_scales):raise ValueError("invalid public refresh scale index")
    refresh_scales={i:positive_scale(s) for i,s in refresh_scales.items()}
    def ws(name,values):
        L=min(v['level'] for v in values);s=weighted_sum_scale([v['scale'] for v in values],delta,drop_divisor(q,L))['output_scale']
        row=dict(name=name,kind='weighted_sum',level=L-1,scale=s);stages.append(row);return row
    def mul(name,a,b):
        L=min(a['level'],b['level']);row=dict(name=name,kind='ciphertext_product',level=L-1,scale=multiply_scale(a['scale'],b['scale'],drop_divisor(q,L)))
        stages.append(row);return row
    t=ws('base.map',[dict(level=input_level,scale=source)]);powers={1:t}
    def power(n):
        if n not in powers:powers[n]=mul(f'base.power{n}',power(n//2),power((n+1)//2))
        return powers[n]
    indices=[i for i in range(1,len(contract.power_coefficients)) if contract.power_coefficients[i]!=0]
    current=ws('base.output',[power(i) for i in indices]);assert current['level']==input_level-6
    rounds=[]
    for i in range(1,7):
        label=f'round{i}';before=current['level'];refresh=before<13
        if refresh:
            assert before>=2
            current=dict(name=label+'.refresh',kind='refresh',level=25,scale=refresh_scales.get(i,delta));stages.append(current)
            events.append(dict(round=i,input_level=before,output_level=25,public_bound=2.))
        squared=mul(label+'.square',current,current)
        denom=dict(name=label+'.denominator',kind='integer_affine',level=squared['level'],scale=squared['scale']);stages.append(denom)
        inverse_level=denom['level']-9;product_level=inverse_level-1
        target=delta*drop_divisor(q,inverse_level)/current['scale']
        backward=backward_scale_plan(q[:2*(denom['level']+1)],input_scale=denom['scale'],first_output_level=denom['level']-1,updates=5,final_scale=target)
        forward=static_newton_forward(q,input_scale=denom['scale'],first_output_level=denom['level']-1,updates=5,first_target_scale=backward['initial_target_scale'])
        for row in forward['rows']:stages.append(dict(name=label+f".inverse{row['update']}",kind='static_inverse',level=row['level'],scale=row['scale']))
        inverse=dict(level=inverse_level,scale=forward['rows'][-1]['scale'])
        product=mul(label+'.product',current,inverse)
        current=dict(name=label+'.output',kind='integer_affine',level=product['level'],scale=product['scale']);stages.append(current)
        assert current['level']==product_level>=2
        rounds.append(dict(round=i,refresh_before=refresh,inverse_scale_plan=forward,nominal_inverse_target_plan=backward,output_level=current['level'],output_scale=current['scale']))
    return dict(input_level=input_level,input_scale=source,canonical_scale=delta,stages=stages,rounds=rounds,refresh_events=events,
                output_level=current['level'],output_scale=current['scale'],range_verified=False,base_input_divisor=64,base_depth=6,
                physical_padding='Pointwise tanh; following classifier masks inactive rows/features.')
class BertRationalTanh:
    def __init__(self,arithmetic,refresher,contract,*,progress=None):
        if not isinstance(contract,BertTanhContract):raise TypeError('frozen tanh contract required')
        if contract_state(contract)!=contract_state(BertTanhContract(contract.manifest_bytes)):raise ValueError('changed tanh contract')
        if progress is not None and not callable(progress):raise TypeError('progress must be callable or None')
        self.arithmetic,self.refresher,self.contract,self.progress=arithmetic,refresher,contract,progress
        self.binding=FullSlotRefreshBinding(arithmetic,refresher)
        self._metadata=FactoredScoreRouting(arithmetic,FactoredScoreLayout(0,0))
        self._bound=(arithmetic,refresher,contract,progress,self.binding,self._metadata);self._contract=contract_state(contract)
        self._active=False;self.last_plan=None;self._guard()
    def _guard(self):
        if any(a is not b for a,b in zip((self.arithmetic,self.refresher,self.contract,self.progress,self.binding,self._metadata),self._bound)):raise ValueError('construction-bound tanh owner changed')
        if contract_state(self.contract)!=self._contract or self.arithmetic.strategy!='power':raise ValueError('fixed tanh method/contract changed')
        self.binding.check();self._metadata._guard_context()
    def _validate(self,value,minimum_levels):
        self._metadata._validate_ciphertext(value,minimum_levels=minimum_levels,canonical=False)
        if typed(getattr(value,'degree_layout',None))!=typed(self.arithmetic.ctx.degree_layout):raise ValueError('tanh ciphertext layout differs')
    def evaluate_tile(self,value,*,capture_checkpoints=False):
        self._guard()
        if self._active:raise ValueError('tanh owner cannot be reentered')
        if type(capture_checkpoints)is not bool:raise ValueError('capture_checkpoints must be bool')
        self._validate(value,8);a=self.arithmetic
        plan=tanh_scale_plan(tuple(a.ctx._param_cache.q_towers),input_level=a._level(value),input_scale=value._ckks_scale,canonical_scale=a.scale,contract=self.contract)
        refresh_scales={}
        expected={x['name']:x for x in plan['stages']};fingerprint=self._metadata._value_fingerprint
        source_state=fingerprint(value);retained={};captured={};checkpoints={};self.last_plan=None;self._active=True
        counts=dict(ciphertext_products=0,weighted_operations=0,integer_affines=0,refreshes=0)
        def guard():
            self._guard()
            if fingerprint(value)!=source_state or any(fingerprint(v)!=state for v,state in (*retained.values(),*captured.values())):raise ValueError('tanh source/retained branch changed')
        def note(name,result):
            guard();self._validate(result,0);row=expected[name]
            if a._level(result)!=row['level'] or typed(result._ckks_scale)!=typed(row['scale']):raise ValueError('actual tanh metadata differs from plan')
            for key in tuple(vars(result)):
                if key.startswith('_public_'):delattr(result,key)
            state=fingerprint(result)
            if capture_checkpoints:checkpoints[name]=result;captured[name]=(result,state)
            if self.progress is not None:self.progress(name,result,dict(counts))
            guard()
            if fingerprint(result)!=state:raise ValueError('tanh callback changed current result')
            for current in (result,*(v for v,_ in retained.values()),*(v for v,_ in captured.values())):
                for key in tuple(vars(current)):
                    if key.startswith('_public_'):delattr(current,key)
            return result
        def keep(name,result):retained[name]=(result,fingerprint(result));return result
        def mul(name,left,right):
            guard();r=a.multiply(left,right);counts['ciphertext_products']+=1;return note(name,r)
        try:
            t=a.weighted_sum([value],[1./64.]);counts['weighted_operations']+=1;t=note('base.map',t);keep('base.t',t);powers={1:t}
            def power(n):
                if n not in powers:powers[n]=keep(f'base.power{n}',mul(f'base.power{n}',power(n//2),power((n+1)//2)))
                return powers[n]
            c=self.contract.power_coefficients;indices=[i for i in range(1,len(c)) if c[i]!=0]
            values=[power(i) for i in indices];guard()
            current=a.weighted_sum(values,[c[i] for i in indices],bias=c[0]);counts['weighted_operations']+=1
            current=note('base.output',current);retained.clear();powers.clear();del values,t
            for index in range(1,7):
                row=plan['rounds'][index-1];label=f'round{index}'
                if row['refresh_before']:
                    keep('refresh_source',current);current=self.binding.refresh(current,public_bound=2.);counts['refreshes']+=1
                    refresh_scales[index]=current._ckks_scale
                    plan=tanh_scale_plan(tuple(a.ctx._param_cache.q_towers),input_level=a._level(value),input_scale=value._ckks_scale,canonical_scale=a.scale,contract=self.contract,refresh_scales=refresh_scales)
                    expected={x['name']:x for x in plan['stages']};row=plan['rounds'][index-1]
                    current=note(label+'.refresh',current);retained.clear()
                keep('iterate',current)
                squared=mul(label+'.square',current,current)
                denom=integer_affine(a,squared,1,bias=1.);counts['integer_affines']+=1;denom=note(label+'.denominator',denom);keep('denominator',denom)
                inverse=None
                for update,inverse in integer_newton_updates(a,denom,normalization_factor=1,initial_estimate=self.contract.seed,updates=5,
                                                            first_output_scale=row['inverse_scale_plan']['initial_target_scale']):
                    counts['weighted_operations']+=update==1;counts['ciphertext_products']+=2*(update!=1);counts['integer_affines']+=update!=1
                    inverse=note(label+f'.inverse{update}',inverse)
                product=mul(label+'.product',current,inverse);keep('product',product)
                current=integer_affine(a,product,2);counts['integer_affines']+=1
                current=note(label+'.output',current);retained.clear()
            guard();self.last_plan=plan
            return current,checkpoints
        finally:
            retained.clear();captured.clear();self._active=False
