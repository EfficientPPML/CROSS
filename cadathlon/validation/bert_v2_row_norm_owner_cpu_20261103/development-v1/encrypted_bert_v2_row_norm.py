"""Frozen BERT-v2 layer0 row LayerNorm, borrowing genuine encrypted primitives.

Only compact valid-token rows with explicit original positions are supported.
Domains are public assumptions, not inspected ciphertext ranges. No codec,
refresh, native normalization, model loader or context ownership is introduced.
"""
from dataclasses import dataclass,field
from collections.abc import Mapping
import hashlib,json,math
import numpy as np
from encrypted_polynomial import PolynomialApproximation
from encrypted_row_normalization import RowLayerNorm
from encrypted_streaming_linear import RowTileLayout
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale

SLOTS=65536
MANIFEST_SHA256='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
AFFINE_SHA256={
 '0.attention':('75757855791f359399f2b35e7fb16db0bc93d08a3476cf7fdd8c523027029d03','2c76aba351fd94b34a299abf2cf8e2d12d2b903bf63f00e66bea6ed20773cc0d'),
 '0.mlp':('e10771889f781626f1a16ea0ad106b3f88ee65df412cd0021b37bdf3c93c007e','5ca80117e32d77fbf91306d6bce9dc1a4b93548c158312825d493346857d6a2d')}
COEFFICIENT_SHA256={'0.attention':'8a5d76da10bcf0b005c7529a2cffec56d717d3c4f03b01658d28fb767f18ad46','0.mlp':'2e036ed2318ebc0a0939013421e434dcd2f831e6387ba508968fa92dc7f07c08'}

def typed(value):
    if isinstance(value,(list,tuple)):return (type(value),tuple(typed(x) for x in value))
    return (type(value),value)

@dataclass(frozen=True)
class V2RowNormContract:
    manifest_bytes:bytes=field(repr=False)
    site:str='0.attention'
    epsilon:float=1e-12
    domain:tuple=field(init=False)
    coefficients:tuple=field(init=False)
    degree:int=field(init=False)
    method:str=field(init=False)
    def __post_init__(self):
        if type(self.manifest_bytes)is not bytes or hashlib.sha256(self.manifest_bytes).hexdigest()!=MANIFEST_SHA256:
            raise ValueError('exact frozen BERT-v2 manifest bytes required')
        if type(self.site)is not str or self.site not in AFFINE_SHA256:raise ValueError('only frozen layer0 attention/MLP sites supported')
        if type(self.epsilon)is not float or self.epsilon!=1e-12:raise ValueError('pinned native epsilon1e-12 required')
        rule=json.loads(self.manifest_bytes)['norms'][self.site];poly=rule['polynomial']
        if rule['method']!='power' or poly['degree']!={'0.attention':12,'0.mlp':16}[self.site]:raise ValueError('fixed power polynomial differs')
        coefficients=tuple(poly['coefficients']);domain=tuple(poly['domain'])
        if hashlib.sha256(np.asarray(coefficients,dtype='<f8').tobytes()).hexdigest()!=COEFFICIENT_SHA256[self.site]:raise ValueError('fixed inverse coefficients differ')
        for name,value in [('domain',domain),('coefficients',coefficients),('degree',poly['degree']),('method',rule['method'])]:object.__setattr__(self,name,value)
    @property
    def approximation(self):return PolynomialApproximation('inverse_sqrt',self.domain,self.coefficients)

FIELDS=('manifest_bytes','site','epsilon','domain','coefficients','degree','method')
def contract_state(contract):return tuple(typed(getattr(contract,n,None)) for n in FIELDS)
def verify_contract(contract):
    if not isinstance(contract,V2RowNormContract):raise TypeError('frozen row norm contract required')
    expected=V2RowNormContract(contract.manifest_bytes,contract.site,contract.epsilon)
    if contract_state(contract)!=contract_state(expected):raise ValueError('row norm contract fields changed')
    return contract_state(contract)

@dataclass(frozen=True)
class V2RowNormResult:
    output:object
    checkpoints:dict
    plan:dict
    original_positions:tuple


def row_norm_scale_plan(q_moduli,*,input_level,input_scale,canonical_scale,contract):
    verify_contract(contract);q=tuple(q_moduli)
    if type(input_level)is not int or input_level<12 or len(q)!=2*(input_level+1) or len(set(q))!=len(q):raise ValueError('11-level CD2 norm plus one continuation level required')
    source=positive_scale(input_scale);delta=positive_scale(canonical_scale)
    weighted=[]
    def ws(name,scales,level):
        row=weighted_sum_scale(scales,delta,drop_divisor(q,level));row.update(name=name,input_level=level,output_level=level-1);weighted.append(row);return row['output_scale']
    mean=ws('mean',[source],input_level)
    if not math.isclose(source,mean,rel_tol=1e-12,abs_tol=0.):raise ValueError('input and computed mean exceed native HEAdd scale compatibility')
    centered=source # genuine subtraction preserves the left input scale
    squares=multiply_scale(centered,centered,drop_divisor(q,input_level-1))
    variance=ws('variance',[squares],input_level-2)
    t=ws('inverse_domain_map',[variance],input_level-3)
    powers={1:dict(power=1,level=input_level-4,scale=t)}
    def power(n):
        if n not in powers:
            a,b=power(n//2),power((n+1)//2);level=min(a['level'],b['level'])
            powers[n]=dict(power=n,level=level-1,scale=multiply_scale(a['scale'],b['scale'],drop_divisor(q,level)))
        return powers[n]
    coefficients=np.polynomial.chebyshev.cheb2poly(contract.coefficients)
    indices=[i for i in range(1,len(coefficients)) if coefficients[i]!=0]
    values=[power(i) for i in indices]
    inverse=ws('inverse_combine',[p['scale'] for p in values],min(p['level'] for p in values))
    inverse_level=input_level-9
    product=multiply_scale(centered,inverse,drop_divisor(q,inverse_level))
    output=ws('affine',[product],input_level-10)
    return dict(site=contract.site,input_level=input_level,input_scale=source,canonical_scale=delta,
        consumed_levels=11,reserved_output_levels=1,polynomial_depth=6,
        levels=dict(mean=input_level-1,centered=input_level-1,squares=input_level-2,variance=input_level-3,
                    inverse=inverse_level,normalized=input_level-10,output=input_level-11),
        scales=dict(mean=mean,centered=centered,squares=squares,variance=variance,inverse=inverse,normalized=product,output=output),
        weighted_sum_stages=weighted,power_stages=values,logical_rotations=42,
        domain=list(contract.domain),epsilon=contract.epsilon,range_verified=False,
        conditional_scope='Only supplied valid compact rows; original positions preserved; no padded-query or CKKS domain guarantee.')


def _affine_vector(value,expected_hash):
    raw=np.asarray(value)
    if raw.shape!=(768,) or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():raise ValueError('finite real pinned width768 vector required')
    with np.errstate(over='ignore',invalid='ignore'):lift=raw.astype(np.float64);f32=lift.astype('<f4')
    if not np.isfinite(lift).all() or not np.array_equal(lift,f32.astype(np.float64)) or hashlib.sha256(f32.tobytes()).hexdigest()!=expected_hash:
        raise ValueError('learned affine vector differs from the pinned checkpoint')
    return np.frombuffer(lift.astype('<f8').tobytes(),dtype='<f8')

def _array_state(value):return (type(value),value.dtype.str,value.shape,value.strides,value.flags.writeable,hashlib.sha256(value.tobytes()).digest())


class V2RowLayerNorm:
    """Serial row operator borrowing one initialized keyless CompiledArithmetic."""
    def __init__(self,arithmetic,manifest_bytes,gamma,beta,*,site='0.attention',epsilon=1e-12,progress=None):
        from encrypted_packed_qwen import CompiledArithmetic
        if not isinstance(arithmetic,CompiledArithmetic):raise TypeError('borrowed compiled arithmetic required')
        if progress is not None and not callable(progress):raise TypeError('progress must be callable or None')
        self.arithmetic=arithmetic;self.ctx=arithmetic.ctx;self.layout=RowTileLayout(SLOTS,4096)
        self.contract=V2RowNormContract(manifest_bytes,site,epsilon);self.approximation=self.contract.approximation
        self.gamma=_affine_vector(gamma,AFFINE_SHA256[site][0]);self.beta=_affine_vector(beta,AFFINE_SHA256[site][1])
        self.progress=progress;self.last_run=None;self._active=False
        # Reuse the measured exact prefix/broadcast primitives without creating
        # another compiled owner or taking ownership of the shared context.
        self._core=RowLayerNorm.__new__(RowLayerNorm);self._core.ctx=self.ctx;self._core.layout=self.layout;self._core.arithmetic=arithmetic
        self._bound=(arithmetic,self.ctx,self.ctx._param_cache,self.layout,self.contract,self.approximation,self.gamma,self.beta,self._core,progress)
        self._contract=verify_contract(self.contract);self._affine=(_array_state(self.gamma),_array_state(self.beta))
        self._context=self._context_state();self._approx=typed((self.approximation.name,self.approximation.domain,self.approximation.coefficients,self.approximation.input_power_basis))
        self._guard()
    @staticmethod
    def required_rotation_indices():return RowLayerNorm.required_rotation_indices(RowTileLayout(SLOTS,4096))
    def _context_state(self):
        c=self.ctx;k=c._param_cache;a=self.arithmetic
        return (tuple(typed(getattr(obj,n,None)) for obj,names in ((c,('degree','num_slots','composite_degree','dnum','degree_layout')),(k,('batch','composite_degree','dnum','degree_layout'))) for n in names),
                typed(tuple(k.q_towers)),typed(tuple(k.p_towers)),typed(a.scale),typed(getattr(a,'strategy',None)),id(getattr(a,'progress',None)))
    def _guard(self):
        now=(self.arithmetic,self.ctx,self.ctx._param_cache,self.layout,self.contract,self.approximation,self.gamma,self.beta,self._core,self.progress)
        if any(a is not b for a,b in zip(now,self._bound)):raise ValueError('construction-bound row norm owner changed')
        c=self.ctx;k=c._param_cache
        if (self.arithmetic.ctx is not c or self._core.ctx is not c or self._core.arithmetic is not self.arithmetic or self._core.layout is not self.layout
                or c.secret_key is not None or k.secret_key is not None or c.parameters.get('secret_key') is not None
                or self._context_state()!=self._context or contract_state(self.contract)!=self._contract
                or (_array_state(self.gamma),_array_state(self.beta))!=self._affine
                or typed((self.approximation.name,self.approximation.domain,self.approximation.coefficients,self.approximation.input_power_basis))!=self._approx
                or getattr(self.arithmetic,'refresher',None) is not None or self.arithmetic.strategy!='power'
                or typed((self.layout.slots,self.layout.features))!=typed((SLOTS,4096))
                or type(c.degree)is not int or c.degree!=2*SLOTS or type(c.num_slots)is not int or c.num_slots!=SLOTS
                or type(c.composite_degree)is not int or c.composite_degree!=2 or type(k.batch)is not int or k.batch!=1):
            raise ValueError('row norm context, layout or public contract changed')
    def _fingerprint(self,value):
        return (typed(self.arithmetic._level(value)),tuple(typed(getattr(value,n,None)) for n in ('_ckks_scale','_ckks_nsd','moduli','batch','num_elements','degree','degree_layout')),id(value.polynomial),id(getattr(value,'ntt_ctx',None)))
    def level_scale_plan(self,q_moduli,*,input_level,input_scale):
        self._guard();return row_norm_scale_plan(q_moduli,input_level=input_level,input_scale=input_scale,canonical_scale=self.arithmetic.scale,contract=self.contract)
    def normalize_tile(self,value,*,valid_rows,original_positions,capture_checkpoints=False):
        self._guard()
        if self._active:raise ValueError('row norm cannot be reentered')
        self.layout._valid_rows(valid_rows)
        if (type(original_positions)is not tuple or len(original_positions)!=valid_rows
                or any(type(x)is not int or not 0<=x<128 for x in original_positions)
                or tuple(sorted(set(original_positions)))!=original_positions):raise ValueError('explicit increasing original positions for every compact valid row required')
        if type(capture_checkpoints)is not bool:raise TypeError('capture must be explicit bool')
        if type(getattr(value,'_ckks_nsd',None))is not int or value._ckks_nsd!=1:raise ValueError('explicit integer NSD1 required')
        self._core._validate_ciphertext(value,11)
        plan=self.level_scale_plan(self.ctx._param_cache.q_moduli_at_level(self.arithmetic._level(value)),input_level=self.arithmetic._level(value),input_scale=value._ckks_scale)
        source=value;state=self._fingerprint(value);retained={};captured={};self._active=True;self.last_run=None
        def guard():
            self._guard()
            if self._fingerprint(source)!=state or any(self._fingerprint(v)!=s for v,s in retained.values()):raise ValueError('row norm input or retained ciphertext changed')
        def note(name,result):
            guard()
            if self.arithmetic._level(result)!=plan['levels'][name] or result._ckks_scale!=plan['scales'][name] or type(result._ckks_nsd)is not int or result._ckks_nsd!=1:raise ValueError('row norm metadata differs from actual public scale plan')
            for key in tuple(vars(result)):
                if key.startswith('_public_'):delattr(result,key)
            fingerprint=self._fingerprint(result)
            if self.progress is not None:self.progress('layer0 '+self.contract.site+' '+name,result,dict(level=plan['levels'][name],original_positions=original_positions))
            guard()
            if self._fingerprint(result)!=fingerprint:raise ValueError('row norm callback changed its result')
            if capture_checkpoints and name in ('mean','variance','inverse'):captured[name]=result
            return result
        try:
            a=self.arithmetic;anchors=np.zeros((16,4096));anchors[:valid_rows,0]=1/768
            mean=note('mean',self._core._broadcast(a.weighted_sum([self._core._prefix768(value)],[anchors.ravel()]),full=False))
            centered=note('centered',a.add(value,mean,subtract=True));del mean,value
            retained['centered']=(centered,self._fingerprint(centered))
            squares=note('squares',a.multiply(centered,centered));bias=np.zeros_like(anchors)
            bias[:valid_rows,0]=self.contract.epsilon;bias[valid_rows:,0]=sum(self.contract.domain)/2
            variance=note('variance',self._core._broadcast(a.weighted_sum([self._core._prefix768(squares)],[anchors.ravel()],bias=bias.ravel()),full=True));del squares
            inverse=note('inverse',a.evaluate(variance,self.approximation))
            normalized=note('normalized',a.multiply(centered,inverse));retained.clear();del centered
            output=note('output',a.weighted_sum([normalized],[self.layout.feature(self.gamma,valid_rows=valid_rows)],bias=self.layout.feature(self.beta,valid_rows=valid_rows)));del normalized
            checkpoints=dict(captured)
            self.last_run=dict(plan=plan,original_positions=original_positions,valid_rows=valid_rows,site=self.contract.site)
            return V2RowNormResult(output,checkpoints,plan,original_positions)
        finally:self._active=False

V2AttentionRowNorm=V2RowLayerNorm
