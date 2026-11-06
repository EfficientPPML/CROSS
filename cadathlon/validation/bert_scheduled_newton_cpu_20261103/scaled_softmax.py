"""Fixed s=4 normalized Newton arithmetic; NumPy only, no CKKS-error claim."""
import numpy as np
from score_layout import FixedSoftmax

class ScaledNewtonSoftmax(FixedSoftmax):
    public_s=4.
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.intermediate_bounds={}
        self.public_round_coefficients=[]

    def observe_bound(self,name,value,lower,upper):
        value=np.asarray(value,dtype=np.float64)
        finite=np.isfinite(value);v=value[finite]
        # Diagnostic allowance for the binary64 simulator only. It is fixed
        # before model execution and is not a CKKS/noise tolerance.
        slack=64*np.finfo(np.float64).eps*max(1.,abs(lower),abs(upper))
        self.intermediate_bounds[name]={
            'ideal_interval':[float(lower),float(upper)],
            'minimum':float(v.min()) if v.size else None,
            'maximum':float(v.max()) if v.size else None,
            'nonfinite':int(np.sum(~finite)),
            'strict_below':int(np.sum(value<lower)),
            'strict_above':int(np.sum(value>upper)),
            'binary64_diagnostic_slack':float(slack),
            'beyond_binary64_slack':int(np.sum((value<lower-slack)|(value>upper+slack))),
            'count':int(value.size)}

    def bound_failures(self):
        return {k:v for k,v in self.intermediate_bounds.items()
                if v['nonfinite'] or v['beyond_binary64_slack']}

    def inverse(self,x,domain,iterations,name):
        """Same unchanged initial mean-exp reciprocal, with diagnostics."""
        self.note(name,x,domain)
        initial=2./sum(domain)
        term=x*(-initial*initial)
        out=term+2*initial
        self.observe_bound(name+'.initial_affine_term',term,-domain[1]*initial*initial,0.)
        self.observe_bound(name+'.initial_output',out,0.,1/domain[0])
        self.counts['public_affine_stages']+=1
        for n in range(1,iterations):
            product=x*out
            correction=2.-product
            self.observe_bound(f'{name}.step{n}.product',product,0.,1.)
            self.observe_bound(f'{name}.step{n}.correction',correction,1.,2.)
            out=out*correction
            self.observe_bound(f'{name}.step{n}.output',out,0.,1/domain[0])
            self.counts['ciphertext_tensor_multiplications']+=2
            self.counts['public_affine_stages']+=1
        return out,1+3*(iterations-1)

    def scaled_inverse(self,x,domain,iterations,name,valid_count):
        self.note(name,x,domain)
        self.observe_bound(name+'.input',x,*domain)
        public_c=valid_count/self.public_s
        initial=2./(public_c*sum(domain))
        # One public affine stage computes the first Newton update. It avoids
        # materializing a potentially larger public_c*x ciphertext.
        term=x*(-public_c*initial*initial)
        out=term+2*initial
        self.observe_bound(name+'.initial_affine_term',term,
                           -domain[1]*public_c*initial*initial,0.)
        self.observe_bound(name+'.initial_output',out,0.,2*self.public_s)
        self.public_round_coefficients.append({'name':name,'K':int(valid_count),
            's':self.public_s,'C':float(public_c),'z0':float(initial),
            'initial_bias':float(2*initial),
            'initial_x_coefficient':float(-public_c*initial*initial)})
        self.counts['public_affine_stages']+=1
        for n in range(1,iterations):
            # Ordering is part of the candidate: form x*z before public C.
            product=x*out
            scaled=public_c*product
            correction=2.-scaled
            self.observe_bound(f'{name}.step{n}.x_times_z',product,0.,self.public_s/valid_count)
            self.observe_bound(f'{name}.step{n}.C_times_product',scaled,0.,1.)
            self.observe_bound(f'{name}.step{n}.correction',correction,1.,2.)
            out=out*correction
            self.observe_bound(f'{name}.step{n}.z_output',out,0.,2*self.public_s)
            self.counts['ciphertext_tensor_multiplications']+=2
            self.counts['public_affine_stages']+=1
        return out,1+3*(iterations-1),public_c

    def evaluate(self,score_slots,mask_slots,valid_count):
        scores=np.asarray(score_slots).reshape(2,16,128,16)
        mask=np.asarray(mask_slots,dtype=bool).reshape(scores.shape)
        live=mask.any(axis=2,keepdims=True)
        mean=np.sum(scores*mask,axis=2,keepdims=True)/valid_count
        centered=(scores-mean)*mask
        self.note('centered_active',centered[mask],self.score_domain)
        self.note('historical_centered_active',centered[mask],(-9.,12.))
        scaled=centered*(2.**-self.squarings)
        self.note('base_exp_input',scaled,(-.5,.5))
        t=scaled*2
        powers={1:t}
        def power(n):
            if n not in powers:
                powers[n]=power(n//2)*power((n+1)//2)
                self.counts['ciphertext_tensor_multiplications']+=1
            return powers[n]
        base=np.full_like(t,self.coefficients[0])
        for n,c in enumerate(self.coefficients[1:],1):base+=c*power(n)
        base*=mask
        self.note('base_exp_output_active',base[mask],(.5,2.))
        self.counts['public_affine_stages']+=4
        denominator=np.sum(base,axis=2,keepdims=True)/valid_count
        denominator=denominator*live+(1-live)
        self.counts['public_affine_stages']+=1
        inverse,initial_depth=self.inverse(denominator,(.5,2.),6,'initial_reciprocal_input')
        probability=base*inverse/valid_count
        self.counts['ciphertext_tensor_multiplications']+=1
        self.counts['public_affine_stages']+=1
        # Baseline abstract budget plus the distinct score-mean anchor mask
        # found by independent review; still not an implemented HE schedule.
        depth=11+initial_depth
        for iteration in range(self.squarings):
            squared=probability*probability
            denominator=np.sum(squared,axis=2,keepdims=True)
            denominator=denominator*live+(1-live)
            inverse,rd,public_c=self.scaled_inverse(denominator,(.5/valid_count,2.),
                self.iterations,f'round{iteration}_reciprocal_input',valid_count)
            # The shared denominator is the sum of these exact squared values.
            # That relation, not separate loose bounds, justifies the product.
            before_scale=squared*inverse
            self.observe_bound(f'round{iteration}.squared_times_z',before_scale,0.,self.public_s/valid_count)
            probability=public_c*before_scale
            self.observe_bound(f'round{iteration}.final_probability',probability,0.,1.)
            self.counts['ciphertext_tensor_multiplications']+=2
            self.counts['public_affine_stages']+=2 # reduction + public C restoration
            depth+=4+rd
        self.depth=depth
        self.note('final_probabilities_active',probability[mask],(-1e-12,1.+1e-12))
        self.note('probability_sums_live',probability.sum(axis=2,keepdims=True)[live],(1.-1e-10,1.+1e-10))
        return probability.reshape(-1)
