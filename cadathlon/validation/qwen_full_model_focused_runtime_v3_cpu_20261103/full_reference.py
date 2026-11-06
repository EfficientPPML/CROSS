"""Independent dense NumPy candidate3 reference, only original embeddings input.

No packed owner or frozen evaluator is imported. Polynomials use NumPy's
Clenshaw/power evaluation and Newton uses the fixed rounded first affine.
No native/owner intermediate enters this circuit. All declared domains and
finite values fail closed; no clipping, coefficient change or fallback occurs.
"""
from dataclasses import dataclass
import hashlib,json,math
import numpy as np
MANIFEST='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
CONFIG='660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'

class DomainFailure(ValueError):
    def __init__(self,label,record):super().__init__('independent declared domain failure: '+label);self.label=label;self.record=record

def finite(value,label):
    value=np.asarray(value,dtype=np.float64)
    if not np.isfinite(value).all():raise FloatingPointError('independent nonfinite stage: '+label)
    return value

def observe(records,label,value,domain):
    value=np.asarray(value,dtype=np.float64);good=np.isfinite(value)
    row=dict(domain=list(domain),count=int(value.size),minimum=float(value[good].min()) if good.any() else None,
        maximum=float(value[good].max()) if good.any() else None,below=int(np.sum(value<domain[0])),above=int(np.sum(value>domain[1])),nonfinite=int(np.sum(~good)))
    records[label]=row
    if row['below'] or row['above'] or row['nonfinite']:raise DomainFailure(label,row)

def polynomial(value,rule,label,records,*,factor=1.):
    observe(records,label,np.asarray(value)*factor,rule['domain'])
    lo,hi=rule['domain'];t=finite(value*(2/(hi-lo)*factor)-(hi+lo)/(hi-lo),label+'.map')
    coefficients=rule['evaluation_coefficients']
    if rule['strategy']=='power':result=np.polynomial.polynomial.polyval(t,coefficients)
    elif rule['strategy']=='chebyshev':result=np.polynomial.chebyshev.chebval(t,coefficients)
    else:raise ValueError('unknown frozen polynomial strategy')
    return finite(result,label+'.polynomial')

def reciprocal(value,rule,label,records):
    observe(records,label,value,rule['domain'])
    y=finite(value*rule['rounded_first_weight']+rule['rounded_first_bias'],label+'.first')
    for i in range(1,rule['iterations']):y=finite(y*(2.-value*y),label+f'.update{i}')
    return y

def norm(value,gamma,epsilon,rule,label,records):
    variance=finite(np.sum(value*value,axis=-1,keepdims=True)*(1./value.shape[-1])+epsilon,label+'.variance')
    if rule['method']=='polynomial':inverse=polynomial(variance,rule,label,records)
    elif rule['method']=='newton':
        observe(records,label,variance,rule['domain'])
        inverse=finite(variance*rule['rounded_first_weight']+rule['rounded_first_bias'],label+'.first')
        for i in range(1,rule['iterations']):inverse=finite(inverse*(1.5-.5*(variance*(inverse*inverse))),label+f'.update{i}')
    else:raise ValueError('unknown frozen inverse-sqrt method')
    return finite(value*inverse*gamma,label+'.normalized')

def rope(value,positions,theta):
    # [S,H,D]; actual binary64 public frequency expression matches the contract.
    d=value.shape[-1];angles=np.asarray(positions)[:,None]*theta**(-np.arange(0,d,2)/d)[None,:]
    c=np.cos(angles)[:,None,:];s=np.sin(angles)[:,None,:]
    left=value[...,:d//2];right=value[...,d//2:]
    return finite(np.concatenate((left*c-right*s,right*c+left*s),axis=-1),'rope')

def causal(query,key,value,rule,manifest,label,records):
    sequence,heads,dimension=query.shape;rows=[]
    for row in range(sequence):
        if row==0:rows.append(value[0].copy());continue
        count=row+1;prefix=label+f'.query{row}';alpha=1./count
        scores=finite(np.einsum('hd,khd->hk',query[row],key[:count])*(1./math.sqrt(dimension)),prefix+'.scores')
        centered=finite(scores-np.sum(scores,axis=-1,keepdims=True)*alpha,prefix+'.centered')
        observe(records,prefix+'.centered',centered,rule['centered_domain'])
        base=polynomial(centered,manifest['exp'],prefix+'.exp',records,factor=2.**-rule['squarings'])
        denominator=finite(base.sum(axis=-1,keepdims=True)*alpha,prefix+'.initial_mean')
        p=finite((base*reciprocal(denominator,manifest['initial_reciprocal'],prefix+'.initial_inverse',records))*alpha,prefix+'.initial')
        for i in range(rule['squarings']):
            square=finite(p*p,prefix+f'.round{i}.square');denominator=finite(square.sum(axis=-1,keepdims=True),prefix+f'.round{i}.sum')
            p=finite(square*reciprocal(denominator,manifest['reciprocal_by_keys'][str(count)],prefix+f'.round{i}.inverse',records),prefix+f'.round{i}.probability')
        rows.append(finite(np.einsum('hk,khd->hd',p,value[:count]),prefix+'.attended'))
    return np.stack(rows)

def silu(value,rule,label,records):
    if rule['method']=='polynomial':return polynomial(value,rule,label,records)
    if rule['method']!='rational_tanh':raise ValueError('unknown frozen SiLU method')
    observe(records,label,value,rule['domain'])
    y=polynomial(value,rule['base'],label+'.tanh_base',records,factor=2.**-(rule['doublings']+1))
    for i in range(rule['doublings']):
        denominator=finite(1.+y*y,label+f'.denominator{i}')
        inverse=reciprocal(denominator,rule['reciprocal'],label+f'.inverse{i}',records)
        y=finite(2.*(y*inverse),label+f'.double{i}')
    return finite(value*(.5*y+.5),label+'.activated')

@dataclass(frozen=True)
class ReferenceResult:
    logits:np.ndarray
    blocks:tuple
    attention:tuple
    operators:dict
    domains:dict
    original_positions:tuple

def forward(embeddings,positions,*,manifest_bytes,config_bytes,matrix,gamma):
    for raw,pin in ((manifest_bytes,MANIFEST),(config_bytes,CONFIG)):
        if type(raw)is not bytes or hashlib.sha256(raw).hexdigest()!=pin:raise ValueError('exact frozen reference bytes required')
    if type(embeddings)is not np.ndarray or embeddings.dtype!=np.dtype('float64') or embeddings.ndim!=2 or embeddings.shape[1]!=1024 or not 1<=len(embeddings)<=32:raise ValueError('original float64[S,1024] embeddings required')
    if type(positions)is not tuple or len(positions)!=len(embeddings) or any(type(p)is not int or not 0<=p<40960 for p in positions) or any(a>=b for a,b in zip(positions,positions[1:])):raise ValueError('typed increasing original positions required')
    m=json.loads(manifest_bytes);config=json.loads(config_bytes);records={};operators={};blocks=[];attention=[]
    value=finite(embeddings.copy(),'original_embeddings');sequence=len(value)
    def linear(x,name,label):return finite(x@matrix(name).T,label)
    for layer in range(28):
        prefix=f'model.layers.{layer}.';attention_prefix=prefix+'self_attn.';label=str(layer)
        site=label+'.input.inverse_sqrt'
        normalized=norm(value,gamma(site),config['rms_norm_eps'],m['nonlinear'][site],site,records);operators[site]=normalized[None].copy()
        projections=[]
        for kind,heads in (('q',16),('k',8),('v',8)):
            projected=linear(normalized,attention_prefix+kind+'_proj.weight',label+'.'+kind)
            operators[label+'.'+kind+'_linear']=projected[None].copy();projections.append(projected.reshape(sequence,heads,128))
        query,key,projected=projections
        for kind,operand in (('q',query),('k',key)):
            site=label+'.'+kind+'.inverse_sqrt';normalized=norm(operand,gamma(site),config['rms_norm_eps'],m['nonlinear'][site],site,records)
            operators[site]=normalized.transpose(1,0,2)[None].copy();roped=rope(normalized,positions,config['rope_theta'])
            operators[label+'.'+kind+'_rope']=roped.transpose(1,0,2)[None].copy()
            if kind=='q':query=roped
            else:key=roped
        key=np.repeat(key,2,axis=1);projected=np.repeat(projected,2,axis=1)
        attended=causal(query,key,projected,m['softmax'][label],m,label+'.attention',records).reshape(sequence,2048)
        attention.append(attended.copy());operators[label+'.gqa']=attended[None].copy()
        branch=linear(attended,attention_prefix+'o_proj.weight',label+'.output');operators[label+'.output_linear']=branch[None].copy()
        residual=finite(value+branch,label+'.attention_residual')
        site=label+'.post.inverse_sqrt';normalized=norm(residual,gamma(site),config['rms_norm_eps'],m['nonlinear'][site],site,records);operators[site]=normalized[None].copy()
        gate=linear(normalized,prefix+'mlp.gate_proj.weight',label+'.gate');up=linear(normalized,prefix+'mlp.up_proj.weight',label+'.up')
        operators[label+'.gate_linear']=gate[None].copy();operators[label+'.up_linear']=up[None].copy()
        activated=silu(gate,m['nonlinear'][label+'.silu'],label+'.silu',records);operators[label+'.silu']=activated[None].copy()
        branch=linear(finite(activated*up,label+'.product'),prefix+'mlp.down_proj.weight',label+'.down');operators[label+'.gated_mlp']=branch[None].copy()
        value=finite(residual+branch,label+'.block');blocks.append(value[None].copy())
    last=norm(value[-1:],gamma('final.inverse_sqrt'),config['rms_norm_eps'],m['nonlinear']['final.inverse_sqrt'],'final.inverse_sqrt',records)
    operators['final.inverse_sqrt']=last.copy()
    logits=linear(last,'lm_head.weight','vocabulary')
    if logits.shape!=(1,151936) or len(operators)!=421:raise ValueError('complete full-vocabulary/operator reference required')
    return ReferenceResult(logits,tuple(blocks),tuple(attention),operators,records,positions)
