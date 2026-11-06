"""Score-only public slot maps and fixed arithmetic; no ciphertext execution."""
from collections import Counter
import math
import numpy as np

ROWS=16; KEYS=128; HEADS=12; PAD_HEADS=16; FEATURES=4096; SLOTS=65536; PERIOD=32768

class ScoreMap:
    """Sparse public linear map: row head anchors -> one key-offset score slice.

    apply() uses exact sparse gathers for the full plaintext experiment.
    apply_diagonals() independently executes its equivalent rotation/mask sum.
    Neither path generates keys or evaluates encrypted linear maps.
    """
    def __init__(self,key_tile,offset):
        if type(key_tile) is not int or not 0<=key_tile<8 or type(offset) is not int or not 0<=offset<16:
            raise ValueError('public tile/offset outside the128key layout')
        r,h=np.meshgrid(np.arange(ROWS),np.arange(HEADS),indexing='ij')
        j=key_tile*16+(r+offset)%16
        self.source=(r*FEATURES+h*64).ravel()
        self.target=((r*KEYS+j)*PAD_HEADS+h).ravel()
        self.diagonal=(self.source-self.target)%SLOTS
        self.indices=tuple(sorted(set(map(int,self.diagonal))))
        # Same public BSGS heuristic as the library for192 distinct diagonals.
        target=max(1,round(math.sqrt(len(self.indices)*2.)))
        self.n1=min((1<<i for i in range(17)),key=lambda n:abs(n-target))
        self.bsgs_rotations=({d%self.n1 for d in self.indices}|{d//self.n1*self.n1 for d in self.indices})-{0}
    def apply(self,value):
        value=np.asarray(value)
        if value.shape!=(SLOTS,):raise ValueError('full-slot input required')
        out=np.zeros(SLOTS);out[self.target]=value[self.source]
        return out
    def inverse(self,value):
        value=np.asarray(value)
        if value.shape!=(SLOTS,):raise ValueError('full-slot input required')
        out=np.zeros(SLOTS);out[self.source]=value[self.target]
        return out
    def apply_diagonals(self,value,*,inverse=False):
        source,target=(self.target,self.source) if inverse else (self.source,self.target)
        displacement=(source-target)%SLOTS
        out=np.zeros(SLOTS)
        for shift in sorted(set(map(int,displacement))):
            destination=target[displacement==shift]
            rotated=np.roll(value,-shift)
            out[destination]+=rotated[destination]
        return out

def duplicate_period(value):
    """One public rotate/add after all score slices fill the first half."""
    assert np.shape(value)==(SLOTS,) and not np.any(value[PERIOD:])
    return value+np.roll(value,PERIOD)

def public_mask(mask,query_tile,sequence):
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=(sequence,) or sequence>KEYS or not mask.any():raise ValueError('invalid key mask')
    out=np.zeros((ROWS,KEYS,PAD_HEADS),bool)
    # Includes public padded query rows. Only rows beyond logical sequence drop.
    count=min(ROWS,sequence-query_tile*ROWS)
    out[:count,:sequence,:HEADS]=mask[None,:,None]
    return np.tile(out.reshape(-1),2)

def exact_probabilities(scores,mask):
    """Independent exact softmax oracle only; never used by the candidate."""
    s=scores.reshape(2,ROWS,KEYS,PAD_HEADS)
    m=mask.reshape(s.shape)
    active=m.any(axis=2,keepdims=True)
    masked=np.where(m,s,-np.inf)
    maxima=np.max(masked,axis=2,keepdims=True)
    ex=np.exp(np.where(m,s-np.where(active,maxima,0),0))*m
    denominator=ex.sum(axis=2,keepdims=True)
    return (ex/np.where(active,denominator,1)).reshape(-1)

class FixedSoftmax:
    """One fixed circuit. Domain observations never branch the computation."""
    def __init__(self,coefficients,*,score_domain=(-16.,16.),squarings=5,iterations=13):
        self.coefficients=np.asarray(coefficients,dtype=np.float64)
        assert len(self.coefficients)==9
        self.score_domain=tuple(score_domain);self.squarings=squarings;self.iterations=iterations
        self.domains={};self.counts=Counter();self.depth=0;self.maximum_magnitude=0.
    def note(self,name,value,domain):
        value=np.asarray(value);finite=np.isfinite(value);x=value[finite]
        lo,hi=domain
        self.domains[name]={'domain':list(domain),'minimum':float(x.min()) if x.size else None,
          'maximum':float(x.max()) if x.size else None,'below':int(np.sum(value<lo)),'above':int(np.sum(value>hi)),
          'nonfinite':int(np.sum(~finite)),'count':int(value.size)}
        if x.size:self.maximum_magnitude=max(self.maximum_magnitude,float(np.max(np.abs(x))))
    def inverse(self,x,domain,iterations,name):
        self.note(name,x,domain)
        initial=2./sum(domain) # Public constant only.
        out=x*(-initial*initial)+2*initial
        self.counts['public_affine_stages']+=1
        for _ in range(1,iterations):
            out=out*(2.-x*out)
            self.counts['ciphertext_tensor_multiplications']+=2
            self.counts['public_affine_stages']+=1
        return out,1+3*(iterations-1)
    def evaluate(self,score_slots,mask_slots,valid_count):
        # Shape/view is a plaintext simulator operation. The planned layout
        # reduction uses rotations +/-16*2**b, one anchor mask per reduction.
        scores=np.asarray(score_slots).reshape(2,ROWS,KEYS,PAD_HEADS)
        mask=np.asarray(mask_slots,dtype=bool).reshape(scores.shape)
        live=mask.any(axis=2,keepdims=True)
        mean=np.sum(scores*mask,axis=2,keepdims=True)/valid_count
        centered=(scores-mean)*mask
        self.note('centered_active',centered[mask],self.score_domain)
        self.note('historical_centered_active',centered[mask],(-9.,12.))
        scaled=centered*(2.**-self.squarings)
        self.note('base_exp_input',scaled,(-.5,.5))
        # Coefficients are monomials in t=2*x on the[-.5,.5] exp interval.
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
        self.counts['public_affine_stages']+=4 # mean mask,center mask,map,finalweightedmask
        denominator=np.sum(base,axis=2,keepdims=True)/valid_count
        denominator=denominator*live+(1-live)
        self.counts['public_affine_stages']+=1 # denominator reduction
        inverse,initial_depth=self.inverse(denominator,(.5,2.),6,'initial_reciprocal_input')
        probability=base*inverse/valid_count
        self.counts['ciphertext_tensor_multiplications']+=1;self.counts['public_affine_stages']+=1
        # From input scores: mean mask1,center mask1,polynomial map1,powers3,
        # final coeff+validmask1,mean reduction1,reciprocal16,multiply1,scale1.
        depth=10+initial_depth
        for iteration in range(self.squarings):
            squared=probability*probability
            denominator=np.sum(squared,axis=2,keepdims=True)
            denominator=denominator*live+(1-live)
            inverse,rd=self.inverse(denominator,(.5/valid_count,2.),self.iterations,f'round{iteration}_reciprocal_input')
            probability=squared*inverse
            self.counts['ciphertext_tensor_multiplications']+=2
            self.counts['public_affine_stages']+=1 # segmented sum mask +inactive1
            depth+=3+rd
        self.depth=depth
        self.note('final_probabilities_active',probability[mask],(-1e-12,1.+1e-12))
        self.note('probability_sums_live',probability.sum(axis=2,keepdims=True)[live],(1.-1e-10,1.+1e-10))
        return probability.reshape(-1)
    def failures(self,*,historical=False):
        return {name:value for name,value in self.domains.items()
          if (historical or not name.startswith('historical_')) and any(value[k] for k in ('below','above','nonfinite'))}
