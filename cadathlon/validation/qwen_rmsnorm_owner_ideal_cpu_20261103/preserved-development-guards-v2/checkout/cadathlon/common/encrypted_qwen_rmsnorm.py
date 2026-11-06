"""Frozen candidate3 RMSNorm with explicit inverse-only continuation.

Public domains and ideal refresh envelopes are caller assumptions. This module
does not inspect ranges, encode/decode, load weights or own a context. Every
selected polynomial and original Newton update remains unchanged.
"""
from dataclasses import dataclass,field
import hashlib,json,math
import numpy as np
from encrypted_qwen_head_routing import QwenHeadRouting,typed,MANIFEST,CONFIG,SLOTS
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale

SITE_METADATA='19155627d1af13c3f68facdbf27e7b700b52f38072fea754d287c2791da3e4b8'
Q_SHA='fc2b841869ef2b2b556209323ed033c5a177344d1de395fc94e8a5865df39a49'
P_SHA='c03a649ab504238a4f24d57d7fe8e49f2a445c5d842579494cd243a7043d1d49'
DELTA=1.1467305003699405e18
def digest(raw):return hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class QwenRMSContract:
    manifest_bytes:bytes=field(repr=False)
    site_metadata_bytes:bytes=field(repr=False)
    site:str
    epsilon:float=1e-6
    domain:tuple=field(init=False)
    method:str=field(init=False)
    width:int=field(init=False)
    groups:int=field(init=False)
    coefficients:tuple=field(init=False)
    degree:int=field(init=False)
    iterations:int=field(init=False)
    first_weight:float=field(init=False)
    first_bias:float=field(init=False)
    gamma_sha256:str=field(init=False)
    events:tuple=field(init=False)
    output_level:int=field(init=False)
    def __post_init__(self):
        if type(self.manifest_bytes)is not bytes or digest(self.manifest_bytes)!=MANIFEST:raise ValueError('exact candidate3 bytes required')
        if type(self.site_metadata_bytes)is not bytes or digest(self.site_metadata_bytes)!=SITE_METADATA:raise ValueError('exact public113-site metadata required')
        if type(self.site)is not str:raise TypeError('exact public site string required')
        if typed(self.epsilon)!=typed(1e-6):raise ValueError('fixed Qwen epsilon required')
        sites=json.loads(self.site_metadata_bytes)
        if self.site not in sites:raise ValueError('unknown frozen normalization site')
        meta=sites[self.site];rule=json.loads(self.manifest_bytes)['nonlinear'][self.site]
        width=meta['group_width'];groups=16 if '.q.' in self.site else 8 if '.k.' in self.site else 1
        polynomial=rule['method']=='polynomial'
        if polynomial:
            if rule['strategy']!='chebyshev' or rule['evaluation_coefficients']!=rule['chebyshev_coefficients']:raise ValueError('unchanged balanced coefficients required')
            degree=rule['degree'];coefficients=tuple(rule['evaluation_coefficients'])
            if degree not in (64,128,256,512) or len(coefficients)!=degree+1:raise ValueError('fixed supported polynomial degree required')
            iterations=0;weight=bias=0.
        else:
            if rule['method']!='newton':raise ValueError('fixed normalization method required')
            degree=0;coefficients=();iterations=rule['iterations'];weight=rule['rounded_first_weight'];bias=rule['rounded_first_bias']
            if type(iterations)is not int or not 15<=iterations<=24:raise ValueError('original update count required')
        events=tuple((e.get('before_iteration',0),float(e['public_bound']),e['input_level']) for e in meta['refresh_events'])
        values=dict(domain=tuple(rule['domain']),method=rule['method'],width=width,groups=groups,coefficients=coefficients,
            degree=degree,iterations=iterations,first_weight=weight,first_bias=bias,gamma_sha256=meta['gamma_f64le_sha256'],
            events=events,output_level=meta['output_level'])
        for name,value in values.items():object.__setattr__(self,name,value)

FIELDS=tuple(QwenRMSContract.__dataclass_fields__)
def contract_state(c):return tuple(typed(getattr(c,n,None)) for n in FIELDS)
def verify_contract(c):
    if not isinstance(c,QwenRMSContract):raise TypeError('frozen RMSNorm contract required')
    expected=QwenRMSContract(c.manifest_bytes,c.site_metadata_bytes,c.site,c.epsilon)
    if contract_state(c)!=contract_state(expected):raise ValueError('typed normalization contract changed')
    return contract_state(c)

@dataclass(frozen=True)
class QwenRMSResult:
    output:object
    original_positions:tuple
    checkpoints:dict
    plan:dict

def rms_scale_plan(q,contract,*,input_scale,refresh_scales=()):
    """Exact metadata replay; refresh scales must be supplied, never invented."""
    verify_contract(contract);q=tuple(q);supplied=tuple(refresh_scales)
    if len(supplied)!=len(contract.events):raise ValueError('one actual scale for every frozen refresh event required')
    source=dict(level=25,scale=positive_scale(input_scale));stages=[];index=0
    def mul(name,x,y):
        level=min(x['level'],y['level']);out=dict(level=level-1,scale=multiply_scale(x['scale'],y['scale'],drop_divisor(q,level)))
        stages.append(dict(name=name,kind='multiply',**out));return out
    def weighted(name,*values):
        level=min(x['level'] for x in values);out=dict(level=level-1,scale=weighted_sum_scale([x['scale'] for x in values],DELTA,drop_divisor(q,level))['output_scale'])
        stages.append(dict(name=name,kind='weighted',**out));return out
    def refresh(name,value,event):
        nonlocal index
        if value['level']!=event[2]:raise ValueError('frozen refresh entry level differs')
        out=dict(level=25,scale=positive_scale(supplied[index]));index+=1
        stages.append(dict(name=name,kind='refresh',public_bound=event[1],**out));return out
    variance=weighted('variance',mul('square',source,source))
    if contract.method=='polynomial':
        t=weighted('map',variance);terms={1:t}
        def term(n):
            if n not in terms:
                product=mul(f'T{n}.product',term(n//2),term((n+1)//2))
                terms[n]=weighted(f'T{n}',product) if n%2==0 else weighted(f'T{n}',product,t)
            return terms[n]
        inverse=weighted('inverse',*[term(n) for n in range(1,contract.degree+1)])
        if contract.events:inverse=refresh('inverse.refresh',inverse,contract.events[0])
    else:
        events={e[0]:e for e in contract.events};inverse=weighted('inverse.first',variance)
        for update in range(2,contract.iterations+1):
            if update in events:inverse=refresh(f'inverse.refresh{update}',inverse,events[update])
            square=mul(f'inverse{update}.square',inverse,inverse)
            product=mul(f'inverse{update}.variance_product',variance,square)
            correction=weighted(f'inverse{update}.correction',product)
            inverse=mul(f'inverse{update}',inverse,correction)
    output=weighted('output',mul('normalized',source,inverse))
    if index!=len(contract.events) or output['level']!=contract.output_level:raise ValueError('frozen level schedule differs')
    return dict(stages=stages,input_level=25,input_scale=source['scale'],output_level=output['level'],output_scale=output['scale'],refresh_scales=supplied)

class QwenRMSNorm(QwenHeadRouting):
    def __init__(self,arithmetic,contract,config_bytes,*,gamma,positions,refresher=None,progress=None):
        self.contract=contract;state=verify_contract(contract)
        if type(gamma)is not np.ndarray or gamma.dtype!=np.dtype('float64') or gamma.shape!=(contract.width,) or not np.isfinite(gamma).all():raise ValueError('finite exact float64 gamma vector required')
        raw=np.asarray(gamma,dtype='<f8').tobytes()
        if digest(raw)!=contract.gamma_sha256:raise ValueError('public learned gamma digest differs')
        self.gamma=np.frombuffer(raw,dtype='<f8');self.refresher=refresher
        super().__init__(arithmetic,contract.manifest_bytes,config_bytes,positions=positions,progress=progress)
        self._rms_bound=(contract,self.gamma,refresher);self._contract=state;self._gamma_bytes=raw
        from encrypted_qwen_rms_refresh import QwenInverseRefreshBinding
        from encrypted_qwen_rms_refresh import accessor
        self._arithmetic_accessors=self._accessors()
        self.refresh_binding=QwenInverseRefreshBinding(self,refresher) if contract.events else None
        if not contract.events and refresher is not None:raise ValueError('no refresher supplied for a no-event frozen rule')
        self._refresh_binding=self.refresh_binding;self._rms_guard()
    def _accessors(self):
        from encrypted_qwen_rms_refresh import accessor
        a=self.arithmetic
        return (*tuple(accessor(a,n) for n in ('rotate','weighted_sum','multiply','add','_level')),
            accessor(a.rotation,'rotate'),accessor(a._plaintext_multiply,'multiply'),accessor(a._rescale,'rescale'))
    def required_rotation_indices(self):
        bits=self.contract.width.bit_length()-1
        return tuple(sorted({s%SLOTS for bit in range(bits) for s in (1<<bit,-(1<<bit))}))
    def _rms_guard(self):
        super()._guard()
        methods=self._accessors()
        if any(a is not b for now,old in zip(methods,self._arithmetic_accessors) for a,b in zip(now,old)):
            raise ValueError('construction-bound arithmetic accessor changed')
        if (any(a is not b for a,b in zip((self.contract,self.gamma,self.refresher),self._rms_bound)) or
            contract_state(self.contract)!=self._contract or self.gamma.tobytes()!=self._gamma_bytes or
            self.gamma.flags.writeable or self.refresh_binding is not self._refresh_binding):raise ValueError('construction-bound RMSNorm state changed')
        c=self.ctx;k=c._param_cache
        if (typed(c.max_level)!=typed(49) or typed(c.dnum)!=typed(8) or typed(c.scaling_factor)!=typed(DELTA) or
            digest(json.dumps(list(k.q_towers)).encode())!=Q_SHA or digest(json.dumps(list(k.p_towers)).encode())!=P_SHA):raise ValueError('fixed commonp60/d8 profile required')
        if self.refresh_binding is not None:self.refresh_binding.check()
    def evaluate_tile(self,value,*,tile,capture_checkpoints=False):
        self._rms_guard()
        if self._active:raise ValueError('serial RMSNorm cannot be reentered')
        if type(capture_checkpoints)is not bool:raise TypeError('exact checkpoint boolean required')
        positions=self.layout.tile_positions(tile);self._validate(value,0)
        a=self.arithmetic;c=self.contract
        if a._level(value)!=25:raise ValueError('frozen RMSNorm requires actual input level25')
        q=tuple(self.ctx._param_cache.q_towers);refs={};checkpoints={};stages=[];refresh_scales=[];counts=dict(rotations=0,additions=0,weighted_sums=0,ciphertext_products=0,refreshes=0)
        def retain(name,v):
            refs[name]=(v,v.polynomial,self._fingerprint(v));return v
        retain('source',value)
        def guard():
            self._rms_guard()
            if any(v.polynomial is not payload or self._fingerprint(v)!=state for v,payload,state in refs.values()):raise ValueError('retained RMSNorm input/basis/checkpoint changed')
        def metadata(v,level,scale):
            self._validate(v,0)
            if (type(a._level(v))is not int or a._level(v)!=level or typed(v._ckks_scale)!=typed(scale) or
                type(v.num_moduli)is not int or v.num_moduli!=2*(level+1) or typed(tuple(v.moduli))!=typed(q[:2*(level+1)])):raise ValueError('RMSNorm primitive metadata differs from public recurrence')
        def note(label,v,level,scale,kind):
            guard();metadata(v,level,scale);self._forget(v);payload=v.polynomial;state=self._fingerprint(v)
            if self.progress is not None:self.progress(label,v,dict(level=level,scale=scale,counts=dict(counts)))
            guard()
            if v.polynomial is not payload or self._fingerprint(v)!=state:raise ValueError('RMSNorm callback changed current result')
            self._forget(v)
            for prior,_,_ in refs.values():
                if prior is not value:self._forget(prior)
            stages.append(dict(name=label,kind=kind,level=level,scale=scale));return v
        def weighted(label,values,weights,bias=0.):
            guard();owners=tuple((v,v.polynomial) for v in values);level=min(a._level(v) for v in values)
            expected=weighted_sum_scale([v._ckks_scale for v in values],DELTA,drop_divisor(q,level))['output_scale'];counts['weighted_sums']+=1
            out=a.weighted_sum(values,weights,bias=bias)
            # Keep the complete input storage alive through both callback guards.
            result=note(label,out,level-1,expected,'weighted');assert owners;return result
        def mul(label,x,y):
            guard();owners=(x.polynomial,y.polynomial);level=min(a._level(x),a._level(y));expected=multiply_scale(x._ckks_scale,y._ckks_scale,drop_divisor(q,level));counts['ciphertext_products']+=1
            result=note(label,a.multiply(x,y),level-1,expected,'multiply');assert owners;return result
        def rotate(v,shift):
            guard();payload=v.polynomial;level=a._level(v);scale=v._ckks_scale;counts['rotations']+=1;out=a.rotate(v,shift%SLOTS)
            metadata(out,level,scale);guard();assert payload is not None;return out
        def add(x,y):
            guard();owners=(x.polynomial,y.polynomial);level=min(a._level(x),a._level(y));scale=x._ckks_scale;counts['additions']+=1;out=a.add(x,y)
            metadata(out,level,scale);guard();assert owners;return out
        def refreshed(label,y,event):
            guard()
            if a._level(y)!=event[2]:raise ValueError('frozen refresh entry level differs')
            retain('refresh_source',y);out=self.refresh_binding.refresh(y,public_bound=event[1]);counts['refreshes']+=1
            result=note(label,out,25,positive_scale(out._ckks_scale),'refresh');stages[-1]['public_bound']=event[1];refresh_scales.append(result._ckks_scale);refs.pop('refresh_source');return result
        def capture(name,v):
            if capture_checkpoints:checkpoints[name]=retain('checkpoint.'+name,v)
            return v
        self._active=True;self.last_run=None
        try:
            square=mul('square',value,value);reduced=square
            for bit in range(c.width.bit_length()-1):reduced=add(reduced,rotate(reduced,1<<bit))
            weights=np.zeros((16,4096));bias=np.zeros((16,4096));lo,hi=c.domain;dummy=lo+(hi-lo)/2
            bias[:,::c.width]=dummy
            for row in range(len(positions)):
                for group in range(c.groups):weights[row,group*c.width]=1/c.width;bias[row,group*c.width]=c.epsilon
            variance=weighted('variance',[reduced],[weights.ravel()],bias.ravel())
            for bit in range(c.width.bit_length()-1):variance=add(variance,rotate(variance,-(1<<bit)))
            retain('variance',variance);capture('variance',variance)
            if c.method=='polynomial':
                t=weighted('map',[variance],[2/(hi-lo)],-(hi+lo)/(hi-lo));terms={1:retain('T1',t)}
                def term(n):
                    if n not in terms:
                        product=mul(f'T{n}.product',term(n//2),term((n+1)//2))
                        out=weighted(f'T{n}',[product],[2.],-1.) if n%2==0 else weighted(f'T{n}',[product,t],[2.,-1.])
                        terms[n]=retain(f'T{n}',out)
                    return terms[n]
                inverse=weighted('inverse',[term(n) for n in range(1,c.degree+1)],list(c.coefficients[1:]),c.coefficients[0])
                # Basis storage remains protected through the final polynomial
                # callback, then may be released before inverse continuation.
                terms.clear()
                for name in tuple(refs):
                    if name.startswith('T'):refs.pop(name)
                if c.events:inverse=refreshed('inverse.refresh',inverse,c.events[0])
            else:
                events={event[0]:event for event in c.events};inverse=weighted('inverse.first',[variance],[c.first_weight],c.first_bias)
                retain('inverse',inverse)
                for update in range(2,c.iterations+1):
                    if update in events:inverse=refreshed(f'inverse.refresh{update}',inverse,events[update]);retain('inverse',inverse)
                    squared=mul(f'inverse{update}.square',inverse,inverse)
                    product=mul(f'inverse{update}.variance_product',variance,squared)
                    correction=weighted(f'inverse{update}.correction',[product],[-.5],1.5)
                    inverse=mul(f'inverse{update}',inverse,correction);retain('inverse',inverse)
            retain('inverse',inverse);capture('inverse',inverse)
            normalized=mul('normalized',value,inverse)
            gamma=np.zeros((16,4096))
            for group in range(c.groups):gamma[:len(positions),group*c.width:(group+1)*c.width]=self.gamma
            output=weighted('output',[normalized],[gamma.ravel()]);retain('output',output)
            expected=rms_scale_plan(q,c,input_scale=value._ckks_scale,refresh_scales=tuple(refresh_scales))
            if stages!=expected['stages']:raise ValueError('complete RMSNorm metadata replay differs')
            guard();self._forget(output)
            self.last_run=dict(expected,counts=dict(counts),site=c.site,domain=c.domain,ideal_domain_assumption=True)
            return QwenRMSResult(output,positions,checkpoints,self.last_run)
        finally:self._active=False
