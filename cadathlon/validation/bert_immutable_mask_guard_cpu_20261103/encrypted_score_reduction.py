"""Experimental masked full-slot row-score reductions, with no refresh or codecs.

A borrowed CompiledArithmetic owner evaluates16 independent query rows at
pitch4096. Only the lower128*16 slots and first12 heads of each row are live.
Public domains are caller contracts, never inferred from encrypted values.
"""
from collections import Counter
import math
import hashlib
import numbers
import numpy as np

SLOTS,ROWS,PITCH,KEYS,HEADS,PAD_HEADS=65536,16,4096,128,12,16
SHIFTS=tuple(16*(1 << bit) for bit in range(7))
_LAYOUT_ARRAYS=('key_mask','counts','input_mask','mean_weight','anchors','dummy_anchors')

def _immutable_numeric_binding(value):
    """Bind standard read-only ndarray views ending at immutable Python bytes.

    Retained references prevent identity reuse. Every ndarray in the backing
    chain has its type, shape, dtype, strides, pointer and writeability checked.
    This is an ordinary Python/NumPy mutation guard, not protection from ctypes
    or a native extension writing through an arbitrary pointer.
    """
    references=[];state=[];seen=set();current=value
    while type(current) is np.ndarray:
        if (id(current) in seen or current.flags.writeable
                or current.dtype.hasobject or current.dtype.fields is not None
                or current.dtype.subdtype is not None or current.dtype.kind not in 'biufc'):
            return None
        seen.add(id(current));references.append(current)
        state.append((id(current),type(current),current.dtype.str,current.shape,
            current.strides,int(current.__array_interface__['data'][0]),
            bool(current.flags.writeable)))
        current=current.base
    if type(current) is not bytes:return None
    references.append(current);state.append((id(current),bytes,len(current)))
    return tuple(references),tuple(state)


def validate_public_dummy(value,domain):
    if not isinstance(domain,(tuple,list)) or len(domain)!=2:
        raise ValueError('public denominator domain must be a pair')
    if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,numbers.Real) for x in (*domain,value)):
        raise ValueError('public domain and dummy must be finite positive reals')
    a,b=map(float,domain);dummy=float(value)
    if not (math.isfinite(a) and math.isfinite(b) and 0<a<b
            and math.isfinite(dummy) and a<=dummy<=b):
        raise ValueError('public dummy must belong to the declared positive finite domain')
    return dummy


class ScoreReductionLayout:
    def __init__(self,key_mask,*,valid_rows=ROWS):
        mask=np.asarray(key_mask)
        if mask.dtype!=np.bool_ or mask.shape not in ((KEYS,),(ROWS,KEYS)):
            raise ValueError('public key mask must be bool[128] or bool[16,128]')
        if type(valid_rows) is not int or not 1<=valid_rows<=ROWS:
            raise ValueError('valid_rows must be a public integer in1..16')
        keys=np.broadcast_to(mask,(ROWS,KEYS)).copy();keys[valid_rows:]=False
        counts=keys.sum(axis=1)
        if np.any(counts[:valid_rows]==0):
            raise ValueError('each requested query needs at least one valid key')
        active=np.zeros((ROWS,PITCH),dtype=np.float64)
        active[:,:KEYS*PAD_HEADS].reshape(ROWS,KEYS,PAD_HEADS)[:,:,:HEADS]=keys[:,:,None]
        anchors=np.zeros_like(active);anchors[:valid_rows,:HEADS]=1.
        dummy_anchors=np.zeros_like(active);dummy_anchors[:,:PAD_HEADS]=1.
        dummy_anchors[:valid_rows,:HEADS]=0.
        dummy_anchors[:,KEYS*PAD_HEADS:KEYS*PAD_HEADS+PAD_HEADS]=1.
        self.valid_rows=valid_rows
        self.key_mask=keys;self.counts=counts
        self.input_mask=active.reshape(-1)
        self.mean_weight=(active/np.maximum(counts,1)[:,None]).reshape(-1)
        self.anchors=anchors.reshape(-1);self.dummy_anchors=dummy_anchors.reshape(-1)
        for name in _LAYOUT_ARRAYS:
            value=getattr(self,name)
            # A read-only flag on an owning ndarray is reversible. A bytes
            #buffer makes ordinary NumPy mutation impossible instead.
            backing=value.tobytes(order='C')
            frozen=np.ndarray(shape=value.shape,dtype=value.dtype,buffer=backing,order='C')
            setattr(self,name,frozen)

    @staticmethod
    def rotation_indices():
        return tuple(sorted({s%SLOTS for shift in SHIFTS for s in (shift,-shift)}))


class ScoreReduction:
    """Mean consumes2 levels; explicitly masked squared denominator consumes3.

Inactive denominator values are a validated public dummy, folded into the
anchor weighted_sum bias. Valid row/head denominators broadcast to all128keys;
invalid probabilities still require their own mask after subsequent nonlinear
operations. This helper rejects bound-aware or implicit-refresh owners.
"""
    def __init__(self,arithmetic,layout,*,progress=None):
        from encrypted_packed_qwen import CompiledArithmetic
        if not isinstance(arithmetic,CompiledArithmetic) or callable(getattr(arithmetic,'declare_bound',None)):
            raise TypeError('score reduction requires a non-bound-aware CompiledArithmetic owner')
        if not isinstance(layout,ScoreReductionLayout):raise TypeError('expected ScoreReductionLayout')
        if progress is not None and not callable(progress):raise TypeError('progress must be callable')
        self.arithmetic,self.layout,self.progress=arithmetic,layout,progress
        self.counts=Counter()
        self._borrowed_arithmetic=arithmetic;self._borrowed_context=arithmetic.ctx
        self._borrowed_cache=getattr(arithmetic.ctx,'_param_cache',None)
        self._borrowed_scale=arithmetic.scale
        self._borrowed_moduli=(tuple(getattr(self._borrowed_cache,'q_towers',())),tuple(getattr(self._borrowed_cache,'p_towers',())))
        self._borrowed_metadata=self._context_metadata()
        self._borrowed_layout=layout;self._bind_layout_fingerprint()
        self._layout_identity=self._layout_fingerprint()
        self._guard_context()

    def _full_layout_digest(self):
        digest=hashlib.sha256(str(self.layout.valid_rows).encode())
        for name in _LAYOUT_ARRAYS:
            value=np.asarray(getattr(self.layout,name))
            digest.update(name.encode());digest.update(value.dtype.str.encode())
            digest.update(repr(value.shape).encode());digest.update(bytes([value.flags.writeable]));digest.update(value.tobytes())
        return digest.digest()

    def _bind_layout_fingerprint(self):
        bindings=[]
        for name in _LAYOUT_ARRAYS:
            value=getattr(self.layout,name)
            binding=_immutable_numeric_binding(value)
            if binding is None:
                # Legacy/custom layouts keep the original complete byte hash
                #on every guard. Never trust an ordinary read-only flag.
                self._immutable_layout_bindings=None
                return
            bindings.append((name,value,binding))
        self._immutable_layout_bindings=tuple(bindings)
        self._immutable_layout_digest=self._full_layout_digest()

    def _layout_fingerprint(self):
        bindings=self._immutable_layout_bindings
        if bindings is None:
            return ('full-hash',self._typed(self.layout.valid_rows),self._full_layout_digest())
        for name,value,binding in bindings:
            current=getattr(self.layout,name)
            observed=_immutable_numeric_binding(current)
            if current is not value or observed is None or observed[1]!=binding[1]:
                raise ValueError('borrowed immutable score mask identity or storage changed')
        return ('immutable',self._typed(self.layout.valid_rows),self._immutable_layout_digest)

    @staticmethod
    def _typed(value):
        if isinstance(value,(tuple,list)):
            return (type(value),tuple(ScoreReduction._typed(x) for x in value))
        return (type(value),value)

    def _context_metadata(self):
        ctx=self.arithmetic.ctx;cache=getattr(ctx,'_param_cache',None)
        return tuple(self._typed(getattr(owner,name,None)) for owner,names in (
            (ctx,('degree','num_slots','composite_degree','dnum','degree_layout')),
            (cache,('batch','composite_degree','dnum','degree_layout')))
            for name in names)

    def _value_fingerprint(self,value):
        return (self._typed(self.arithmetic._level(value)),
            tuple(self._typed(getattr(value,name,None)) for name in
                ('_ckks_scale','_ckks_nsd','moduli','batch','num_elements','degree','degree_layout')),
            id(getattr(value,'polynomial',None)),id(getattr(value,'ntt_ctx',None)))

    def _guard_context(self):
        if (self.arithmetic is not self._borrowed_arithmetic or self.arithmetic.ctx is not self._borrowed_context
                or getattr(self.arithmetic.ctx,'_param_cache',None) is not self._borrowed_cache
                or self._typed(self.arithmetic.scale)!=self._typed(self._borrowed_scale)
                or self._context_metadata()!=self._borrowed_metadata
                or (tuple(getattr(self._borrowed_cache,'q_towers',())),tuple(getattr(self._borrowed_cache,'p_towers',())))!=self._borrowed_moduli
                or self.layout is not self._borrowed_layout or self._layout_fingerprint()!=self._layout_identity):
            raise ValueError('borrowed score context, arithmetic or public layout changed')
        ctx=self.arithmetic.ctx;cache=getattr(ctx,'_param_cache',None)
        if cache is None or ctx.degree!=2*SLOTS or ctx.num_slots!=SLOTS:
            raise ValueError('score reduction requires initialized full65536-slot context')
        if ctx.secret_key is not None or cache.secret_key is not None or ctx.parameters.get('secret_key') is not None:
            raise ValueError('score reduction requires a keyless context')
        if type(cache.batch)is not int or cache.batch!=1 or type(ctx.composite_degree)is not int or ctx.composite_degree!=2:
            raise ValueError('score reduction requires batchone/composite degree two')
        if getattr(self.arithmetic,'refresher',None) is not None:
            raise ValueError('score reduction does not permit implicit refresh')

    def _validate(self,value,levels):
        self._guard_context()
        from jaxite_word import Polynomial
        from support import require_secure_execution
        if not isinstance(value,Polynomial) or value.batch!=1 or value.num_elements!=2 or value.degree!=2*SLOTS:
            raise TypeError('canonical full-slot two-component ciphertext required')
        require_secure_execution(self.arithmetic.ctx,value)
        if type(getattr(value,'_ckks_nsd',None)) is not int or value._ckks_nsd!=1:
            raise ValueError('explicit integer noise-scale degree one required')
        if self.arithmetic._level(value)<levels:raise ValueError('insufficient levels for score reduction')
        if not math.isclose(value._ckks_scale,self.arithmetic.scale,rel_tol=1e-12):
            raise ValueError('canonical input scale required')

    @staticmethod
    def _forget(result):
        for name in tuple(vars(result)):
            if name.startswith('_public_'):delattr(result,name)
        return result

    def _note(self,label,value):
        if self.progress is None:return
        self._guard_context()
        before=self._value_fingerprint(value)
        self.progress(label,value,dict(self.counts))
        self._guard_context()
        if self._value_fingerprint(value)!=before:
            raise ValueError('progress callback changed score ciphertext metadata or payload')
        if type(value._ckks_nsd)is not int or value._ckks_nsd!=1:
            raise ValueError('progress callback changed integer noise-scale degree one')

    def _weighted(self,value,weight,*,bias=0.,label):
        level=self.arithmetic._level(value)
        result=self.arithmetic.weighted_sum([value],[weight],bias=bias)
        self.counts['public_multiplications']+=1;self.counts['actual_composite_rescales']+=1
        self.counts['path_levels']+=1
        if np.any(bias):self.counts['public_bias_additions']+=1
        if self.arithmetic._level(result)!=level-1 or result._ckks_nsd!=1 or not math.isclose(result._ckks_scale,self.arithmetic.scale,rel_tol=1e-12):
            raise RuntimeError('weighted reduction returned unexpected level/scale/NSD')
        result=self._forget(result);self._note(label,result);return result

    def _tree(self,value,*,reverse=False):
        for shift in SHIFTS:
            value=self.arithmetic.add(value,self.arithmetic.rotate(value,(-shift if reverse else shift)%SLOTS))
            self.counts['rotations']+=1;self.counts['ciphertext_additions']+=1
        return self._forget(value)

    def _reduce_broadcast(self,value,*,dummy=0.,label):
        value=self._tree(value)
        value=self._weighted(value,self.layout.anchors,bias=self.layout.dummy_anchors*dummy,label=label+' anchors')
        value=self._tree(value,reverse=True);self._note(label+' broadcast',value);return value

    def mean(self,value):
        self._validate(value,2)
        value=self._weighted(value,self.layout.mean_weight,label='mean masked and normalized input')
        return self._reduce_broadcast(value,label='mean')

    def squared_denominator(self,value,*,domain,inactive_dummy):
        dummy=validate_public_dummy(inactive_dummy,domain)
        self._validate(value,3)
        value=self._weighted(value,self.layout.input_mask,label='squared denominator masked input')
        level=self.arithmetic._level(value)
        value=self.arithmetic.multiply(value,value)
        self.counts['ciphertext_multiplications']+=1;self.counts['actual_composite_rescales']+=1
        self.counts['path_levels']+=1
        if self.arithmetic._level(value)!=level-1 or value._ckks_nsd!=1:
            raise RuntimeError('denominator square returned unexpected level/NSD')
        value=self._forget(value);self._note('squared denominator square',value)
        return self._reduce_broadcast(value,dummy=dummy,label='squared denominator')

    def mean_with_dummy(self,value,*,domain,inactive_dummy):
        """Two-level masked mean, with an in-domain inverse input on inactive lanes."""
        dummy=validate_public_dummy(inactive_dummy,domain)
        self._validate(value,2)
        value=self._weighted(value,self.layout.mean_weight,label='mean masked and normalized input')
        return self._reduce_broadcast(value,dummy=dummy,label='mean with inactive dummy')
