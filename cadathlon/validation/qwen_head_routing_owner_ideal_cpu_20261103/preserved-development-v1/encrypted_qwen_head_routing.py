"""Experimental full-width Qwen D128/Q16/K8 public head/routing owner.

Uses only borrowed compiled arithmetic and supplied ciphertexts. No keygen,
codec, refresh, nonlinear substitution or context ownership. It does not
implement RMSNorm, softmax, PV, a decoder block or full-model inference.
"""
from dataclasses import dataclass
from collections.abc import Mapping
import hashlib, json, math
import numpy as np
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale

SLOTS=65536
MANIFEST='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
CONFIG='660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'
def typed(x):
    if isinstance(x,(tuple,list)):return (type(x),tuple(typed(v) for v in x))
    return (type(x),x)

@dataclass(frozen=True)
class QwenHeadLayout:
    positions: tuple
    slots: int=65536
    row_width: int=4096
    query_heads: int=16
    kv_heads: int=8
    head_dim: int=128
    key_stride: int=32
    def __post_init__(self):
        if (type(self.positions)is not tuple or not 1<=len(self.positions)<=32 or
                any(type(v)is not int or not 0<=v<32768 for v in self.positions) or
                tuple(sorted(set(self.positions)))!=self.positions):
            raise ValueError('one to32 increasing original public positions required')
        expected=(65536,4096,16,8,128,32)
        if typed((self.slots,self.row_width,self.query_heads,self.kv_heads,self.head_dim,self.key_stride))!=typed(expected):
            raise ValueError('fixed full-slot Qwen head geometry required')
    @property
    def tile_count(self):return (len(self.positions)+15)//16
    def tile_rows(self,tile):
        if type(tile)is not int or not 0<=tile<self.tile_count:raise ValueError('invalid row tile')
        return min(16,len(self.positions)-16*tile)
    def tile_positions(self,tile):return self.positions[16*tile:16*tile+self.tile_rows(tile)]
    def causal_mask(self,tile):
        out=np.zeros((16,4096),dtype=np.float64)
        for row,pos in enumerate(self.tile_positions(tile)):
            for head in range(16):
                for key,kpos in enumerate(self.positions):out[row,head*32+key]=float(kpos<=pos)
        return out.reshape(-1)

@dataclass(frozen=True)
class QwenScoreTile:
    output: object
    query_positions: tuple
    key_positions: tuple
    counts: dict
    level: int
    scale: float

class QwenHeadRouting:
    def __init__(self,arithmetic,manifest_bytes,config_bytes,*,positions,progress=None):
        from encrypted_packed_qwen import CompiledArithmetic
        if not isinstance(arithmetic,CompiledArithmetic):raise TypeError('borrowed CompiledArithmetic required')
        if type(manifest_bytes)is not bytes or hashlib.sha256(manifest_bytes).hexdigest()!=MANIFEST:raise ValueError('frozen candidate3 bytes required')
        if type(config_bytes)is not bytes or hashlib.sha256(config_bytes).hexdigest()!=CONFIG:raise ValueError('pinned Qwen config bytes required')
        if progress is not None and not callable(progress):raise TypeError('progress must be callable or None')
        config=json.loads(config_bytes)
        if tuple(config[n] for n in ('hidden_size','num_attention_heads','num_key_value_heads','head_dim'))!=(1024,16,8,128):raise ValueError('Qwen architecture differs')
        self.arithmetic=arithmetic;self.ctx=arithmetic.ctx;self.layout=QwenHeadLayout(positions)
        self.theta=float(config['rope_theta']);self.progress=progress;self.last_run=None;self._active=False
        self._bound=(arithmetic,self.ctx,self.ctx._param_cache,self.layout,progress)
        self._layout_state=typed(tuple(vars(self.layout).items()))
        self._theta=typed(self.theta);self._context=self._context_state()
        self._keys=self.ctx._param_cache.raw_rotation_keys;self._maps=self.ctx._param_cache.coef_maps
        self._key_objects=self._key_state();self._guard()
    def required_rotation_indices(self):
        n=len(self.layout.positions)
        return tuple(sorted({s%SLOTS for s in [64,-64,*[-128*k for k in range(1,9)],
            *[4096*k for k in range(1,16)],*[1<<k for k in range(7)],
            *[96*h for h in range(1,16)],*[-k for k in range(1,n)]] if s%SLOTS}))
    def _context_state(self):
        c=self.ctx;k=c._param_cache;a=self.arithmetic
        return (tuple(typed(getattr(o,n,None)) for o,names in ((c,('degree','num_slots','composite_degree','dnum','degree_layout')),
            (k,('degree','batch','composite_degree','dnum','degree_layout'))) for n in names),
            typed(tuple(k.q_towers)),typed(tuple(k.p_towers)),typed(a.scale),typed(getattr(a,'strategy',None)),id(getattr(a,'progress',None)))
    def _key_state(self):
        k=self.ctx._param_cache;keys=k.raw_rotation_keys;maps=k.coef_maps
        if getattr(k,'per_level_rotation_keys',None)is not False or not isinstance(keys,Mapping) or not isinstance(maps,Mapping):raise ValueError('supplied reusable rotation inventory required')
        states=[]
        for shift in self.required_rotation_indices():
            pair=keys.get(shift)
            if not isinstance(pair,Mapping) or pair.get('a')is None or pair.get('b')is None or maps.get(shift)is None:raise ValueError('missing public rotation pair/map')
            states.append((shift,id(pair),id(pair['a']),id(pair['b']),id(maps[shift])))
        return tuple(states)
    def _guard(self):
        now=(self.arithmetic,self.ctx,self.ctx._param_cache,self.layout,self.progress)
        if any(a is not b for a,b in zip(now,self._bound)):raise ValueError('construction-bound owner changed')
        c=self.ctx;k=c._param_cache;a=self.arithmetic
        if (a.ctx is not c or c.secret_key is not None or k.secret_key is not None or c.parameters.get('secret_key')is not None or
            getattr(a,'refresher',None)is not None or callable(getattr(a,'declare_bound',None)) or a.strategy!='power' or
            typed(tuple(vars(self.layout).items()))!=self._layout_state or typed(self.theta)!=self._theta or
            self._context_state()!=self._context or k.raw_rotation_keys is not self._keys or k.coef_maps is not self._maps or self._key_state()!=self._key_objects):
            raise ValueError('borrowed context/layout/public inventory changed')
        self.layout.__post_init__()
        if any(type(getattr(o,n,None))is not int or getattr(o,n)!=v for o,n,v in
            ((c,'degree',131072),(c,'num_slots',65536),(c,'composite_degree',2),(k,'degree',131072),(k,'batch',1),(k,'composite_degree',2))):raise ValueError('full ring/CD2 batch-one context required')
        if typed(c.dnum)!=typed(k.dnum) or type(c.dnum)is not int or c.dnum<1 or typed(c.degree_layout)!=typed(k.degree_layout):raise ValueError('context decomposition/layout mismatch')
    def _fingerprint(self,v):
        return (typed(self.arithmetic._level(v)),tuple(typed(getattr(v,n,None)) for n in ('_ckks_scale','_ckks_nsd','moduli','batch','num_elements','degree','degree_layout')),id(v.polynomial),id(getattr(v,'ntt_ctx',None)))
    def _validate(self,v,depth):
        from jaxite_word import Polynomial
        from support import require_secure_execution
        if not isinstance(v,Polynomial):raise TypeError('canonical ciphertext required')
        if any(type(getattr(v,n,None))is not int or getattr(v,n)!=x for n,x in (('batch',1),('num_elements',2),('degree',131072),('_ckks_nsd',1))):raise ValueError('exact ciphertext ring/NSD metadata required')
        if typed(v.degree_layout)!=typed(self.ctx.degree_layout):raise ValueError('ciphertext layout mismatch')
        positive_scale(v._ckks_scale);require_secure_execution(self.ctx,v)
        level=self.arithmetic._level(v)
        if type(level)is not int or level<depth+2:raise ValueError('operation requires two remaining levels')
    @staticmethod
    def _forget(v):
        for n in tuple(vars(v)):
            if n.startswith('_public_'):delattr(v,n)
    def _execute(self,sources,depth,body):
        self._guard()
        if self._active:raise ValueError('serial owner cannot be reentered')
        for v in sources:self._validate(v,depth)
        originals=tuple((v,self._fingerprint(v)) for v in sources);retained={};counts={'rotations':0,'weighted_sums':0,'ciphertext_products':0,'additions':0}
        self._active=True;self.last_run=None;a=self.arithmetic
        def guard():
            self._guard()
            if any(self._fingerprint(v)!=s for v,s in (*originals,*retained.values())):raise ValueError('input or retained ciphertext changed')
        def keep(name,v):retained[name]=(v,self._fingerprint(v));return v
        def note(label,v,level,scale):
            guard();self._forget(v)
            if type(v._ckks_nsd)is not int or v._ckks_nsd!=1 or a._level(v)!=level or v._ckks_scale!=scale:raise ValueError('actual primitive metadata differs from public recurrence')
            state=self._fingerprint(v)
            if self.progress is not None:self.progress(label,v,dict(level=level,scale=scale,counts=dict(counts)))
            guard()
            if self._fingerprint(v)!=state:raise ValueError('callback changed result')
            self._forget(v)
            for prior,_ in retained.values():self._forget(prior)
            return v
        def rotate(v,shift):
            guard();shift%=SLOTS
            if not shift:return v
            counts['rotations']+=1;out=a.rotate(v,shift)
            if self._fingerprint(out)[0:2]!=self._fingerprint(v)[0:2]:raise ValueError('rotation changed metadata')
            return out
        def weighted(values,weights,label):
            guard();level=min(a._level(v) for v in values);divisor=drop_divisor(tuple(self.ctx._param_cache.q_towers),level)
            expected=weighted_sum_scale([v._ckks_scale for v in values],a.scale,divisor)['output_scale'];counts['weighted_sums']+=1
            return note(label,a.weighted_sum(values,weights),level-1,expected)
        def multiply(x,y,label):
            guard();level=min(a._level(x),a._level(y));divisor=drop_divisor(tuple(self.ctx._param_cache.q_towers),level)
            expected=multiply_scale(x._ckks_scale,y._ckks_scale,divisor);counts['ciphertext_products']+=1
            return note(label,a.multiply(x,y),level-1,expected)
        def add(x,y):guard();counts['additions']+=1;return a.add(x,y)
        try:
            output=body(rotate,weighted,multiply,add,keep,guard)
            output=note('output',output,a._level(output),output._ckks_scale)
            guard();self._forget(output);self.last_run=dict(counts=counts,output_level=a._level(output),output_scale=output._ckks_scale)
            return output
        finally:self._active=False
    def expand_kv(self,value,*,tile):
        self._guard();rows=self.layout.tile_rows(tile)
        def body(rotate,weighted,*unused):
            values=[];weights=[]
            for group in range(9):
                mask=np.zeros((16,4096))
                for h in range(16):
                    if (h+1)//2==group:mask[:rows,h*128:(h+1)*128]=1
                values.append(rotate(value,-128*group));weights.append(mask.ravel())
            return weighted(values,weights,'gqa_expand')
        return self._execute((value,),1,body)
    def rope(self,value,*,tile,heads):
        self._guard();positions=self.layout.tile_positions(tile)
        if type(heads)is not int or heads not in (8,16):raise ValueError('exact8KV or16Q heads required')
        angles=np.asarray(positions)[:,None]*self.theta**(-np.arange(0,128,2)/128)[None,:]
        angle=np.concatenate((angles,angles),axis=-1);cs=np.zeros((16,4096));plus=cs.copy();minus=cs.copy()
        for h in range(heads):
            cs[:len(positions),h*128:(h+1)*128]=np.cos(angle)
            plus[:len(positions),h*128:h*128+64]=-np.sin(angle[:,:64])
            minus[:len(positions),h*128+64:(h+1)*128]=np.sin(angle[:,64:])
        def body(rotate,weighted,*unused):return weighted([value,rotate(value,64),rotate(value,-64)],[cs.ravel(),plus.ravel(),minus.ravel()],'rope')
        return self._execute((value,),1,body)
    def scores(self,query,expanded_rope_key_tiles,*,query_tile):
        """All causal QK scores, heads16/width128, key slots0..S-1.

        Keys already have actual Q/K normalization, RoPE and GQA expansion.
        Each key is publicly broadcast to query rows, multiplied and reduced;
        only aligned head anchors are routed. Padding/future slots are zero.
        """
        self._guard();positions=self.layout.tile_positions(query_tile)
        if type(expanded_rope_key_tiles)is not tuple or len(expanded_rope_key_tiles)!=self.layout.tile_count:raise ValueError('exact ordered expanded-key tile tuple required')
        def body(rotate,weighted,multiply,add,keep,guard):
            total=None
            for key,keypos in enumerate(self.layout.positions):
                source=expanded_rope_key_tiles[key//16];values=[];weights=[]
                for row in range(len(positions)):
                    mask=np.zeros((16,4096));mask[row,:2048]=1
                    values.append(rotate(source,((key%16)-row)*4096));weights.append(mask.ravel())
                broadcast=weighted(values,weights,f'key{key}_broadcast')
                product=multiply(query,broadcast,f'key{key}_products');del broadcast
                reduced=product
                for bit in range(7):reduced=add(reduced,rotate(reduced,1<<bit))
                vals=[];masks=[]
                for head in range(16):
                    mask=np.zeros((16,4096))
                    for row,pos in enumerate(positions):
                        if keypos<=pos:mask[row,head*32]=1/math.sqrt(128)
                    vals.append(rotate(reduced,head*96));masks.append(mask.ravel())
                packed=rotate(weighted(vals,masks,f'key{key}_scores'),-key)
                total=packed if total is None else add(total,packed)
                # Keep one accumulated result protected across later callbacks.
                keep('aggregate',total);guard()
            return total
        output=self._execute((query,*expanded_rope_key_tiles),3,body)
        return QwenScoreTile(output,positions,self.layout.positions,dict(self.last_run['counts']),self.last_run['output_level'],self.last_run['output_scale'])
