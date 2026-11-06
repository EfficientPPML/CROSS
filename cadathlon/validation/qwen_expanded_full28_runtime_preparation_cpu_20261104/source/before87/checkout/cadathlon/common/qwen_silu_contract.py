"""Pinned candidate3 SiLU contracts and binary64 metadata replay, no backend."""
from dataclasses import dataclass,field
import hashlib,json
from encrypted_qwen_head_routing import typed,MANIFEST
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale
CERTIFICATE='ad386cb499682a0abb54f81a26d67136b2df0da26383b4b140184344de904a37'
DELTA=1.1467305003699405e18

def digest(raw):return hashlib.sha256(raw).hexdigest()

@dataclass(frozen=True)
class QwenSiLUContract:
    manifest_bytes:bytes=field(repr=False)
    certificate_bytes:bytes=field(repr=False)
    layer:int
    method:str=field(init=False)
    domain:tuple=field(init=False)
    degree:int=field(init=False)
    strategy:str=field(init=False)
    coefficients:tuple=field(init=False)
    polynomial_domain:tuple=field(init=False)
    input_factor:float=field(init=False)
    doublings:int=field(init=False)
    inverse_rule:tuple=field(init=False)
    events:tuple=field(init=False)
    def __post_init__(self):
        if type(self.manifest_bytes)is not bytes or digest(self.manifest_bytes)!=MANIFEST:raise ValueError('exact candidate3 manifest required')
        if type(self.certificate_bytes)is not bytes or digest(self.certificate_bytes)!=CERTIFICATE:raise ValueError('exact conditional public certificate required')
        if type(self.layer)is not int or not 0<=self.layer<28:raise ValueError('exact layer0..27 required')
        m=json.loads(self.manifest_bytes);proof=json.loads(self.certificate_bytes);r=m['nonlinear'][f'{self.layer}.silu']
        if proof['passed']is not True or proof['manifest_sha256']!=MANIFEST:raise ValueError('certificate identity differs')
        polynomial=r['method']=='polynomial';p=r if polynomial else r['base']
        if polynomial:
            if r['strategy']!='chebyshev' or r['degree']not in (192,256) or r['evaluation_coefficients']!=r['chebyshev_coefficients']:raise ValueError('fixed balanced SiLU polynomial required')
            doublings=0;inverse=();events=();factor=1.
        else:
            if r['method']!='rational_tanh' or p['strategy']!='power' or p['degree']!=12:raise ValueError('fixed rational-tanh circuit required')
            d=r['doublings'];ir=r['reciprocal'];record=next(x for x in proof['rational_silu'] if x['site']==f'{self.layer}.silu')
            if type(d)is not int or d not in (6,7,9,11) or ir['iterations']!=5:raise ValueError('original doubling/reciprocal counts required')
            if (record['domain']!=r['domain'] or record['doublings']!=d or record['base_refresh_bound']!=2 or
                record['actual_power_hex']!=[float(x).hex() for x in p['evaluation_coefficients']] or
                len(record['rounds'])!=d or any(x['refresh_bound']!=2 for x in record['rounds'])):raise ValueError('exact conditional B2 certificate differs')
            doublings=d;factor=2.**-(d+1);inverse=(tuple(ir['domain']),ir['rounded_first_weight'],ir['rounded_first_bias'],ir['iterations'])
            events=tuple((f'double{j}',2.,18 if j==1 else 8) for j in range(1,d+1))
        values=dict(method=r['method'],domain=tuple(r['domain']),degree=p['degree'],strategy=p['strategy'],
            coefficients=tuple(p['evaluation_coefficients']),polynomial_domain=tuple(p['domain']),input_factor=factor,
            doublings=doublings,inverse_rule=inverse,events=events)
        for name,value in values.items():object.__setattr__(self,name,value)

FIELDS=tuple(QwenSiLUContract.__dataclass_fields__)
def contract_state(c):return tuple(typed(getattr(c,n,None)) for n in FIELDS)
def verify_contract(c):
    if type(c)is not QwenSiLUContract:raise TypeError('exact frozen SiLU contract required')
    expected=QwenSiLUContract(c.manifest_bytes,c.certificate_bytes,c.layer)
    if contract_state(c)!=contract_state(expected):raise ValueError('typed frozen SiLU contract changed')
    return contract_state(c)

def silu_scale_plan(q,contract,*,input_scale,refresh_scales=()):
    """Replay actual operations with supplied returned scales, never relabel."""
    verify_contract(contract);q=tuple(q);scales=tuple(refresh_scales)
    if len(scales)!=len(contract.events):raise ValueError('one actual scale per explicit event required')
    source=dict(level=24,scale=positive_scale(input_scale));stages=[]
    counts=dict(weighted_sums=0,ciphertext_products=0,refreshes=0)
    def weighted(label,*values):
        level=min(v['level'] for v in values)
        out=dict(level=level-1,scale=weighted_sum_scale([v['scale'] for v in values],DELTA,drop_divisor(q,level))['output_scale'])
        counts['weighted_sums']+=1;stages.append(dict(label=label,kind='weighted',**out));return out
    def mul(label,x,y):
        level=min(x['level'],y['level']);out=dict(level=level-1,scale=multiply_scale(x['scale'],y['scale'],drop_divisor(q,level)))
        counts['ciphertext_products']+=1;stages.append(dict(label=label,kind='multiply',**out));return out
    terms={1:weighted('map',source)}
    def term(n):
        if n not in terms:
            product=mul(f'basis{n}.product',term(n//2),term((n+1)//2))
            terms[n]=(product if contract.strategy=='power' else weighted(f'basis{n}',product) if n%2==0 else weighted(f'basis{n}',product,terms[1]))
        return terms[n]
    indices=[n for n in range(1,len(contract.coefficients)) if contract.strategy=='chebyshev' or contract.coefficients[n]!=0]
    value=weighted('output' if not contract.doublings else 'polynomial',*[term(n) for n in indices])
    for j,event in enumerate(contract.events,1):
        if value['level']!=event[2]:raise ValueError('explicit SiLU refresh entry differs')
        value=dict(level=25,scale=positive_scale(scales[j-1]));counts['refreshes']+=1
        stages.append(dict(label=f'double{j}.refresh',kind='refresh',public_bound=2.,**value))
        numerator=value;square=mul(f'double{j}.square',value,value);denominator=weighted(f'double{j}.denominator',square)
        inverse=weighted(f'double{j}.inverse.first',denominator)
        for update in range(2,contract.inverse_rule[3]+1):
            product=mul(f'double{j}.inverse{update}.product',denominator,inverse)
            correction=weighted(f'double{j}.inverse{update}.correction',product)
            inverse=mul(f'double{j}.inverse{update}',inverse,correction)
        value=weighted(f'double{j}.output',mul(f'double{j}.quotient',numerator,inverse))
    if contract.doublings:value=mul('output',source,weighted('sigmoid',value))
    if value['level']!=6:raise ValueError('frozen SiLU output level differs')
    return dict(layer=contract.layer,method=contract.method,input_level=24,input_scale=source['scale'],output_level=6,
        output_scale=value['scale'],stages=stages,counts=counts,refresh_scales=scales,
        scope='Actual binary64 metadata only; conditional domain and ideal-refresh certificate, no HE error guarantee.')
