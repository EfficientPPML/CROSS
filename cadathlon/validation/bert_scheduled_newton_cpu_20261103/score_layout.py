"""Factored public score-slot routing; NumPy semantics only, no encrypted run."""
from collections import Counter
import math
import numpy as np

ROWS=16; KEYS=128; HEADS=12; PAD_HEADS=16; FEATURES=4096; SLOTS=65536; PERIOD=65536
POSITIONS=np.arange(SLOTS)
ROW_INDEX=POSITIONS//FEATURES


class ScoreMap:
    """Keep each query at pitch4096; score keys occupy its lower2048 slots.

    Public bit-routing stages use two complementary masks, one rotation and
    one addition. A stage consumes one plaintext-multiply/rescale level.
    This simulates exact real arithmetic, including filtering hostile padding.
    """
    def __init__(self,key_tile,offset):
        if type(key_tile) is not int or not 0<=key_tile<8 or type(offset) is not int or not 0<=offset<16:
            raise ValueError('public tile/offset outside the128-key layout')
        self.key_tile,self.offset=key_tile,offset
        r,h=np.meshgrid(np.arange(ROWS),np.arange(HEADS),indexing='ij')
        k=key_tile*16+(r+offset)%16
        self.source=(r*FEATURES+h*64).ravel()
        self.target=(r*FEATURES+k*PAD_HEADS+h).ravel()
        self.source_mask=np.zeros(SLOTS,dtype=bool);self.source_mask[self.source]=True
        self.target_mask=np.zeros(SLOTS,dtype=bool);self.target_mask[self.target]=True
        self.counts=Counter();self.rotations=set()

    def rotate(self,value,shift):
        shift=int(shift)%SLOTS
        if not shift:return value
        self.counts['rotations']+=1;self.rotations.add(shift)
        return np.roll(value,-shift)

    def route(self,value,selector,shift):
        self.counts['public_multiplications']+=2
        self.counts['public_multiplication_levels']+=1
        self.counts['additions']+=1
        return np.where(selector,0.,value)+self.rotate(np.where(selector,value,0.),shift)

    def apply(self,value):
        value=np.asarray(value,dtype=np.float64)
        if value.shape!=(SLOTS,):raise ValueError('full-slot input required')
        result=value*self.source_mask
        self.counts['public_multiplications']+=1
        self.counts['public_multiplication_levels']+=1
        # Move the four head-index bits down six bit positions.
        for bit in range(4):
            selector=(POSITIONS & (1<<(6+bit)))!=0
            result=self.route(result,selector,63*(1<<bit))
        # The query row never moves. Its public index determines the key row.
        for bit in range(4):
            selector=(((ROW_INDEX+self.offset)%16)&(1<<bit))!=0
            result=self.route(result,selector,-16*(1<<bit))
        return self.rotate(result,-256*self.key_tile)

    def inverse(self,value):
        value=np.asarray(value,dtype=np.float64)
        if value.shape!=(SLOTS,):raise ValueError('full-slot input required')
        result=value*self.target_mask
        self.counts['public_multiplications']+=1
        self.counts['public_multiplication_levels']+=1
        result=self.rotate(result,256*self.key_tile)
        for bit in reversed(range(4)):
            selector=(((ROW_INDEX+self.offset)%16)&(1<<bit))!=0
            result=self.route(result,selector,16*(1<<bit))
        for bit in reversed(range(4)):
            selector=(POSITIONS & (1<<bit))!=0
            result=self.route(result,selector,-63*(1<<bit))
        return result

    def oracle(self,value,*,inverse=False):
        source,target=(self.target,self.source) if inverse else (self.source,self.target)
        result=np.zeros(SLOTS);result[target]=np.asarray(value)[source]
        return result


def duplicate_period(value):
    """Compatibility spelling: this full-period layout has no duplicate alias."""
    if np.shape(value)!=(SLOTS,):raise ValueError('full-slot input required')
    return np.asarray(value)


def public_mask(mask,query_tile,sequence):
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=(sequence,) or sequence>KEYS or not mask.any():raise ValueError('invalid key mask')
    # Each query has two independent2048-slot chunks. The upper chunk is
    # inactive, with the same safe dummy denominator as padded head groups.
    out=np.zeros((ROWS,2,KEYS,PAD_HEADS),bool)
    count=min(ROWS,sequence-query_tile*ROWS)
    out[:count,0,:sequence,:HEADS]=mask[None,:,None]
    return out.reshape(-1)

# The following fixed arithmetic is unchanged. Reshaping as2x16 chunks
# instead of16x2 is just a view: reductions stay inside each2048-slot chunk.
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
