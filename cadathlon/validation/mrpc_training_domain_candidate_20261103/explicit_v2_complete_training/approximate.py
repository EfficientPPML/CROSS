"""Float64 analogue of fixed arithmetic candidates; strict nonlinear domains."""
import hashlib,math
import numpy as np

class DomainViolation(ValueError):
    def __init__(self,label,row):
        super().__init__(f'{label} outside declared domain');self.label=label;self.row=row

class Arithmetic:
    def __init__(self,manifest):
        self.manifest=manifest;self.observations={};self.coefficients={}
        polynomials=[manifest['base_exp'],manifest['tanh']['base_polynomial']]
        polynomials += [r['polynomial'] for r in manifest['norms'].values()]
        polynomials += [r['polynomial'] for r in manifest['gelu'].values()]
        for p in polynomials:
            c=np.asarray(p['coefficients'],dtype=np.float64)
            if hashlib.sha256(c.astype('<f8').tobytes()).hexdigest()!=p['coefficient_f64le_sha256']:raise ValueError('public coefficient bytes differ')
            self.coefficients[p['coefficient_f64le_sha256']]=(c,np.polynomial.chebyshev.cheb2poly(c))

    def observe(self,label,x,domain):
        x=np.asarray(x);lo,hi=domain;finite=np.isfinite(x);values=x[finite]
        row=dict(domain=list(domain),count=int(x.size),minimum=float(values.min()) if values.size else None,
            maximum=float(values.max()) if values.size else None,below=int(np.sum(x<lo)),above=int(np.sum(x>hi)),nonfinite=int(np.sum(~finite)))
        self.observations[label]=row
        if row['below'] or row['above'] or row['nonfinite']:raise DomainViolation(label,row)

    def polynomial(self,x,rule,strategy,label):
        self.observe(label,x,rule['domain']);lo,hi=rule['domain']
        t=x*(2/(hi-lo))-(hi+lo)/(hi-lo);terms={1:t}
        cheb,power=self.coefficients[rule['coefficient_f64le_sha256']]
        coefficients=power if strategy=='power' else cheb
        def basis(n):
            if n not in terms:
                product=basis(n//2)*basis((n+1)//2)
                terms[n]=(product if strategy=='power' else 2*product-(1. if n%2==0 else t))
            return terms[n]
        indices=[n for n in range(1,len(coefficients)) if strategy!='power' or coefficients[n]!=0]
        result=basis(indices[0])*coefficients[indices[0]]
        for n in indices[1:]:result=result+basis(n)*coefficients[n]
        return result+coefficients[0]

    def inverse(self,x,rule,label):
        self.observe(label,x,rule['domain']);initial=rule['seed']
        result=x*(-initial*initial)+2*initial
        for _ in range(1,rule['iterations']):result=result*(2-x*result)
        return result

    def norm(self,x,gamma,beta,epsilon,rule,label):
        centered=x-x.mean(axis=-1,keepdims=True)
        variance=np.mean(centered*centered,axis=-1,keepdims=True)+epsilon
        strategy=rule['method']
        if strategy not in ('power','balanced_chebyshev'):raise ValueError('v2 requires an explicit polynomial norm strategy')
        inverse=self.polynomial(variance,rule['polynomial'],'power' if strategy=='power' else 'chebyshev',label+'.inverse_sqrt')
        return centered*inverse*gamma+beta

    def softmax(self,scores,rule,label):
        keys=scores.shape[-1];centered=scores-scores.mean(axis=-1,keepdims=True)
        self.observe(label+'.centered_scores',centered,rule['centered_domain'])
        base=self.polynomial(centered/(2**rule['squarings']),self.manifest['base_exp'],'power',label+'.base_exp')
        inverse=self.inverse(base.mean(axis=-1,keepdims=True),self.manifest['initial_reciprocal'],label+'.initial_mean_inverse')
        probabilities=base*inverse/keys
        reciprocal=self.manifest['normalization_reciprocal_by_public_K'][str(keys)]
        for i in range(rule['squarings']):
            square=probabilities*probabilities;denominator=square.sum(axis=-1,keepdims=True)
            inverse=self.inverse(denominator,reciprocal,label+f'.round{i}.denominator_inverse.K{keys}')
            probabilities=square*inverse
        return probabilities

    def tanh(self,x,rule,label):
        self.observe(label+'.input',x,rule['domain'])
        value=self.polynomial(x/(2**rule['doublings']),rule['base_polynomial'],'power',label+'.base')
        for i in range(rule['doublings']):
            denominator=1+value*value
            inverse=self.inverse(denominator,self.manifest['tanh_reciprocal'],label+f'.round{i}.denominator_inverse')
            value=2*value*inverse
        return value

    def forward(self,config,w,x):
        value=x;heads=config['num_attention_heads'];dimension=value.shape[-1]//heads
        for layer in range(config['num_hidden_layers']):
            prefix=f'bert.encoder.layer.{layer}.'
            def linear(x,name):return x@w[prefix+name+'.weight'].T+w[prefix+name+'.bias']
            q,k,v=[linear(value,'attention.self.'+name).reshape(len(value),heads,dimension).transpose(1,0,2) for name in ('query','key','value')]
            scores=(q@k.swapaxes(-1,-2))/math.sqrt(dimension)
            probabilities=self.softmax(scores,self.manifest['softmax'][str(layer)],f'{layer}.softmax')
            attended=(probabilities@v).transpose(1,0,2).reshape(value.shape)
            residual=value+linear(attended,'attention.output.dense');name=prefix+'attention.output.LayerNorm'
            value=self.norm(residual,w[name+'.weight'],w[name+'.bias'],config['layer_norm_eps'],self.manifest['norms'][f'{layer}.attention'],f'{layer}.attention')
            middle=linear(value,'intermediate.dense');rule=self.manifest['gelu'][str(layer)]
            middle=self.polynomial(middle,rule['polynomial'],'chebyshev',f'{layer}.gelu')
            residual=value+linear(middle,'output.dense');name=prefix+'output.LayerNorm'
            value=self.norm(residual,w[name+'.weight'],w[name+'.bias'],config['layer_norm_eps'],self.manifest['norms'][f'{layer}.mlp'],f'{layer}.mlp')
        pooled=value[0]@w['bert.pooler.dense.weight'].T+w['bert.pooler.dense.bias']
        pooled=self.tanh(pooled,self.manifest['tanh'],'pooler.tanh')
        return pooled@w['classifier.weight'].T+w['classifier.bias']
