"""Independent compact dense layer-attention oracles; client/test side only."""
import json,math
import numpy as np

def compute(embeddings,model,gamma,beta,manifest,*,approximate,layer):
    x=np.asarray(embeddings,dtype=np.float64);K=len(x)
    if x.shape!=(K,768) or not 1<=K<=128:raise ValueError('compact full-width rows required')
    projections={name:x@row['weight'].T+row['bias'] for name,row in model.items() if name!='output'}
    q,k,v=[projections[name].reshape(K,12,64) for name in ('query','key','value')]
    scores=np.einsum('qhd,khd->qkh',q,k)/8
    centered=scores-scores.mean(axis=1,keepdims=True)
    domains=[]
    def observe(name,values,domain):
        lo,hi=domain;minimum=float(values.min());maximum=float(values.max())
        row=dict(name=name,minimum=minimum,maximum=maximum,domain=list(domain),passed=minimum>=lo and maximum<=hi)
        domains.append(row)
        if not row['passed']:raise ValueError('propagated domain failure '+str(row))
    if approximate:
        softmax_rule=manifest['softmax'][str(layer)]
        observe('centered_scores',centered,softmax_rule['centered_domain'])
        coeff=np.polynomial.chebyshev.cheb2poly(manifest['base_exp']['coefficients'])
        exponent=np.polynomial.polynomial.polyval(centered/(2**(softmax_rule['squarings']-1)),coeff)
        den=exponent.mean(axis=1,keepdims=True);observe('initial_mean_exp',den,[.5,2.])
        rule=manifest['initial_reciprocal'];seed=rule['seed']
        inverse=den*(-seed*seed)+2*seed
        for _ in range(1,rule['iterations']):inverse*=2-den*inverse
        prob=exponent*inverse/K
        rule=manifest['normalization_reciprocal_by_public_K'][str(K)];seed=rule['seed']
        for index in range(softmax_rule['squarings']):
            squared=prob**2;den=squared.sum(axis=1,keepdims=True)
            observe('squared_denominator_'+str(index+1),den,rule['domain'])
            inverse=den*(-seed*seed)+2*seed
            for _ in range(1,rule['iterations']):inverse*=2-den*inverse
            prob=squared*inverse
    else:
        exponent=np.exp(scores-scores.max(axis=1,keepdims=True));prob=exponent/exponent.sum(axis=1,keepdims=True)
    mixed=np.einsum('qkh,khd->qhd',prob,v).reshape(K,768)
    row=model['output'];residual=mixed@row['weight'].T+row['bias']+x
    mean=residual.mean(axis=-1,keepdims=True);variance=((residual-mean)**2).mean(axis=-1,keepdims=True)+1e-12
    if approximate:
        rule=manifest['norms'][str(layer)+'.attention'];observe('attention_variance',variance,rule['domain'])
        lo,hi=rule['domain'];t=variance*(2/(hi-lo))-(hi+lo)/(hi-lo)
        if rule['method']=='power':
            coefficients=np.polynomial.chebyshev.cheb2poly(rule['polynomial']['coefficients'])
            inverse=np.polynomial.polynomial.polyval(t,coefficients)
        elif rule['method']=='balanced_chebyshev':
            inverse=np.polynomial.chebyshev.chebval(t,rule['polynomial']['coefficients'])
        else:raise ValueError('unsupported frozen method')
    else:inverse=1/np.sqrt(variance)
    output=(residual-mean)*inverse*gamma+beta
    return dict(output=output,scores=scores,probabilities=prob,mixed_values=mixed,residual=residual,
        variance=variance,domain_observations=domains)
