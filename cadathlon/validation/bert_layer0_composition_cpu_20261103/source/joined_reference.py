"""Independent dense complete-block polynomial/native oracle; validation side only."""
import numpy as np
from scipy.special import erf
import reference as attention_reference

def compute(embeddings,attention_model,attention_gamma,attention_beta,ffn,manifest,*,approximate):
    attention=attention_reference.compute(embeddings,attention_model,attention_gamma,attention_beta,
                                          manifest,approximate=approximate)
    x=attention['output'];domains=list(attention['domain_observations'])
    def observe(name,values,domain):
        lo,hi=domain
        row=dict(name=name,minimum=float(np.min(values)),maximum=float(np.max(values)),domain=list(domain),
                 passed=bool(np.isfinite(values).all() and np.all(values>=lo) and np.all(values<=hi)))
        domains.append(row)
        if not row['passed']:raise ValueError('propagated complete-block domain failure '+str(row))
    middle=x@ffn['intermediate.weight'].T+ffn['intermediate.bias']
    rule=manifest['gelu']['0']['polynomial'];lo,hi=rule['domain']
    if approximate:
        observe('FFN_GELU_input',middle,rule['domain'])
        activated=np.polynomial.chebyshev.chebval((2/(hi-lo))*middle-(hi+lo)/(hi-lo),rule['coefficients'])
    else:activated=.5*middle*(1+erf(middle/np.sqrt(2.)))
    residual=activated@ffn['output.weight'].T+ffn['output.bias']+x
    centered=residual-residual.mean(axis=-1,keepdims=True)
    variance=(centered*centered).mean(axis=-1,keepdims=True)+1e-12
    rule=manifest['norms']['0.mlp']['polynomial'];lo,hi=rule['domain']
    if approximate:
        observe('MLP_variance',variance,rule['domain'])
        # Independent dense evaluation uses the actual rounded power-basis
        # coefficients selected by the frozen runtime, not a fitted output.
        coefficients=np.polynomial.chebyshev.cheb2poly(rule['coefficients'])
        inverse=np.polynomial.polynomial.polyval((2/(hi-lo))*variance-(hi+lo)/(hi-lo),coefficients)
    else:inverse=1/np.sqrt(variance)
    output=centered*inverse*ffn['gamma']+ffn['beta']
    return dict(output=output,attention=attention['output'],middle=middle,activated=activated,
                residual=residual,variance=variance,domain_observations=domains)
