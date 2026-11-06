"""Stage-specific public Newton counts; NumPy only, no encryption claim."""
import numpy as np
from scaled_softmax import ScaledNewtonSoftmax
from public_schedule import PublicSchedule

class ScheduledNewtonSoftmax(ScaledNewtonSoftmax):
    def __init__(self,*args,schedule,**kwargs):
        super().__init__(*args,**kwargs)
        if not isinstance(schedule,PublicSchedule) or tuple(schedule.certificate['coefficients'])!=tuple(self.coefficients):
            raise ValueError('schedule must certify these exact public coefficients')
        self.schedule=schedule
        self.round_contracts=[]

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
            contract=self.schedule.stage(valid_count,iteration)
            # Dummy1 would be outside early intervals. A public midpoint is
            # safe for every inactive group and never depends on activations.
            denominator=denominator*live+contract['inactive_dummy']*(1-live)
            self.round_contracts.append(contract)
            inverse,rd,public_c=self.scaled_inverse(denominator,(contract['a'],contract['b']),
                contract['iterations'],f'round{iteration}_reciprocal_input',valid_count)
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
