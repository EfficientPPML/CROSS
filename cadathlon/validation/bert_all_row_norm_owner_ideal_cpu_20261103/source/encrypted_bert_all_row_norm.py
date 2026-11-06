"""Frozen BERT-v2 all24 row LayerNorm sites, borrowing genuine encrypted primitives.

Only compact valid-token rows with explicit original positions are supported.
Domains are public assumptions, not inspected ciphertext ranges. No codec,
refresh, native normalization, model loader or context ownership is introduced.
"""
from dataclasses import dataclass,field
from collections.abc import Mapping
import hashlib,json,math
import numpy as np
from encrypted_polynomial import PolynomialApproximation,EncryptedPolynomialEvaluator
from encrypted_row_normalization import RowLayerNorm
from encrypted_streaming_linear import RowTileLayout
from encrypted_scale_planning import positive_scale,drop_divisor,weighted_sum_scale,multiply_scale

SLOTS=65536
MANIFEST_SHA256='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
AFFINE_SHA256={'0.attention': ('75757855791f359399f2b35e7fb16db0bc93d08a3476cf7fdd8c523027029d03',
                 '2c76aba351fd94b34a299abf2cf8e2d12d2b903bf63f00e66bea6ed20773cc0d'),
 '0.mlp': ('e10771889f781626f1a16ea0ad106b3f88ee65df412cd0021b37bdf3c93c007e',
           '5ca80117e32d77fbf91306d6bce9dc1a4b93548c158312825d493346857d6a2d'),
 '1.attention': ('6316372e1a432e72debc360260c15857d0e454188bf4e7a876ea04f88bb54339',
                 'de6025ff142a079ff70501a73e0ec02ce524c4e1e28f85715bdb2875d8ff6a9f'),
 '1.mlp': ('b2bf9b467392cccee9b2c8681a0901e0a6b10562e76e580d404943c1eb7f28c5',
           '98f9a1bacb0067f1d0b8f0b9ad2d36e328c4c220c28e197db0a29c8605c1e823'),
 '2.attention': ('766e6edfa879e2913e4f9cf3c3e1274e27ed57d2861e62a278238483ea59fe29',
                 '2a1bffd76ca29eb4853976511ba8858dd8ad6b1beeb0b2b54620c25a7ae36951'),
 '2.mlp': ('21de0868829773e3a3de564347bb1606170d9a6911059f84a91478d640d0af5d',
           '657c6ef663d7d144bdaac360b2f8705a5ae40a98a2bb88632e0dca0a6a702e07'),
 '3.attention': ('c90eecee0a6db1608381fb53441abde728c70041d9e208bf70b6492828b2b490',
                 'b6c2fb26593c69dc75f7a02eb914e10ca12879e30a5bb91c829291dacbd225cf'),
 '3.mlp': ('874cdec928501b7bdce137ef169d835dd5ab3ab1abba92b11821fdd390903d5a',
           '53b8cd99fd37507ad3257f7c0e5fcfc779b6b35f6e955f127c42949878a2e336'),
 '4.attention': ('2aa9d933667e1512ba42bb76385126edef60776fab407e5498a8fea2e31665e4',
                 '2c60603f7ea87a5b12e16e150514b1045fe700892cdc4b3e9b1a3c09e870d978'),
 '4.mlp': ('8f27ad4c14ef5885d60674f9db441d8e96801feb6dc8ce9c21ffc822c7bcd431',
           'c5077ccefde4b688c95e76424635533410ec6e5b096a3ad40d7b11521144b656'),
 '5.attention': ('932fd1c0f0f294958b0599369b76667593d0edd2cf9ce3ed42d65e42ab0a89d6',
                 '75a5f349d1589356067d0bfce00e63db33afeeedee9e8955aa9536e3ec90533a'),
 '5.mlp': ('d8aaf2650fadff6c393dad1b73702cf59411cdabed2d6ce1d3c44b673fa426c6',
           '62f69fac1683ce32b31690de751c607c6a559ba62d342c6b69a142d65acb3428'),
 '6.attention': ('b5fea01335a751278baeddf1a31cb083e095f29ad7796dc8af7c4ab6118950a9',
                 '4010bdcbd7d42f338ab3835d87356cea38b8469fa01708f209c32a294e7fe131'),
 '6.mlp': ('441bc873242233cc5a9d72c0f50bff95770831f45637259470b2c759d028f9df',
           '768d8e5e5ae955b01806a36744c7123688aa45d2a561e709fc165e3febeaec4f'),
 '7.attention': ('4a2af000c606b7cf37092ced2f0ad5e201cfddcb1f772114c463b536179b1ff0',
                 '83d7f85d9ea16e31d44c901a7638703fc346d867641514fe852a8ff92b8fe71a'),
 '7.mlp': ('04dc58c809616a5ec5c96b1181e6993c36997bdf37b45e7ce40b742741a574eb',
           '0b51077b40dc93207ea596df9db71899d32757b2e788d59e3ebf657d934aee22'),
 '8.attention': ('a2091edec061f6e5a577f96f43eab3b8d31ac4833e89886cbb2e362b72467d06',
                 '7662023d46f64eded53be93004c4acc43ed31c8fd25dc0f6f5db144f35eb9eae'),
 '8.mlp': ('2c3445b33cc113a96648bb3332eda535ff72f09489274808435ad8dcd61b7893',
           'f9a9a41c863cc21ef0e7a8f9ffde427db7d08a3cb6195cb6fe6ecb47c1d83bdc'),
 '9.attention': ('c9efcd7bf3a015c9f228f8a3165d49fba019cf8043318a0a04ef0c87fdf481a8',
                 '6b44b62033253f32ca446c198535f7aa37ee26ed072bd454a5b47a037cdea880'),
 '9.mlp': ('6aa522f2876743fe43850117918163a0cf213c69dbc84d55a12fef83fa7b2c03',
           'a45de94be812eb7d01aed2d8944418d7bb17a0b45e76a066304923770b98c97f'),
 '10.attention': ('0b2aa95da210df8b4b520d20a02bc0c362cfd6ea1e6506c2171bc48eb5e6b0cc',
                  'aea7d92865aa1cc9c9ee70ff6b86b63cda2e1fd50211a78409fea8e2641294a8'),
 '10.mlp': ('fbe7c68512c3bd200f33260dff3683cde107250c60a52758f9afb8748c3cafc6',
            '05d598681c1073c38c1271044b023d04c20aa57e23901102f835b1810cc0f3d1'),
 '11.attention': ('a08386ea70d634ea2f284a174268381f3646e57e17f5faba41512534b5782905',
                  '7ced907e09f765f162f2b64b31c263668ba7275d721504eb9c830e6e37ab880c'),
 '11.mlp': ('353fa2eacc234dbc765fbe46b46199d98bba5f569397572a51ec685769784d27',
            '1f119fd4768af7863ad2f8c7b29be7c4212ccc3ca6926f66e00e875ff87b5516')}
COEFFICIENT_SHA256={'0.attention': '8a5d76da10bcf0b005c7529a2cffec56d717d3c4f03b01658d28fb767f18ad46',
 '0.mlp': '2e036ed2318ebc0a0939013421e434dcd2f831e6387ba508968fa92dc7f07c08',
 '1.attention': 'b5b454156d81149d861b55be1d5beb8b7ce0decc8c5750776381721d74873fed',
 '1.mlp': '8f5bd7c7c314ff56572e02dea325b35d0622780e3f07c1422a5eff69354cb39b',
 '2.attention': 'fe08988318d67588dcc0c10a8e8c7a7000d2739dc169ad0ddbfc54a0753502dc',
 '2.mlp': '0c42b23f122b0d977407c774c91202d619845342377793d4643f99d7cdc33ba6',
 '3.attention': '1be0b59bacb51af3cbc8730a508703cddf7600c5f2d20ff55e688181e1f2c744',
 '3.mlp': 'c930a7e011d4895899659cb396ec590ad0afd37cc1deff496c57fc8b37edc8d4',
 '4.attention': 'f4081162ff32702b19d057cfed6bfe9b00c68778391a0d3fc33e2cc978ba9f42',
 '4.mlp': 'eaff795f75b8ddf079c63aa11fef09aefc6df2bd2b36e83bb30f8874a91571f8',
 '5.attention': '344a3b137277c52e3520dcd005d4bf4826c1c7ffa46d78734b2feb43e35036db',
 '5.mlp': '94d7eb8a1559da32e8daa83190db704620bdb4af54230297e7d5ed30097e0203',
 '6.attention': '25b8a1f750dac054bd4ed0b2e752b84f0be9fd287536f724938522eac98815a6',
 '6.mlp': '9f39a0aa1d1ab01093a0474d9bf87e333c1d01c389aea69a563d87e2d765a355',
 '7.attention': 'dcc2b2f78db068a481a75742ab2677e9b7d21f10ac07399e514586820cc79244',
 '7.mlp': 'd085dd924f9d1206ae0cb10840790cb4c607e324ec97f6a0adbc86b8a9a28d3f',
 '8.attention': '62140557a76b6ea829858df776c21e04c7144fd730ce5b0c91839efb8b96c2db',
 '8.mlp': 'fb6513d66ec443de6536324482e4f88c861a7fce03f271b62ab31f9bb5946adb',
 '9.attention': '7e3661e606e85023190bbeef7204810edf0a737af008d84c1b548c5d35c451cf',
 '9.mlp': 'e0990852980b8fb323b1165cede75bf379669bd62b6b03566bbbd2afa3e4880d',
 '10.attention': '21b250d634f4e85df2f9e0b8174dbc7d3985463283ab59d740a08ebf7c3e8a18',
 '10.mlp': 'abaec5613d30300848a3f31d4e2c3021cf5da4aa0ed137d16941e23cbcc45229',
 '11.attention': 'd2976541b9e9cf2795fc2ec42340d4818617c0dc0402af5495c890f1f4de858a',
 '11.mlp': '2aef6d57cf2a352b0609c26a2a152561d92ce8662910e5f66404cab31776650a'}

def typed(value):
    if isinstance(value,(list,tuple)):return (type(value),tuple(typed(x) for x in value))
    return (type(value),value)

@dataclass(frozen=True)
class BertRowNormContract:
    manifest_bytes:bytes=field(repr=False)
    site:str='0.attention'
    epsilon:float=1e-12
    domain:tuple=field(init=False)
    coefficients:tuple=field(init=False)
    degree:int=field(init=False)
    method:str=field(init=False)
    def __post_init__(self):
        if type(self.manifest_bytes)is not bytes or hashlib.sha256(self.manifest_bytes).hexdigest()!=MANIFEST_SHA256:
            raise ValueError('exact frozen BERT-v2 manifest bytes required')
        if type(self.site)is not str or self.site not in AFFINE_SHA256:raise ValueError('only frozen12-layer attention/MLP sites supported')
        if type(self.epsilon)is not float or self.epsilon!=1e-12:raise ValueError('pinned native epsilon1e-12 required')
        rule=json.loads(self.manifest_bytes)['norms'][self.site];poly=rule['polynomial']
        if rule['method'] not in ('power','balanced_chebyshev') or type(poly['degree']) is not int or poly['degree']!=len(poly['coefficients'])-1:
            raise ValueError('fixed selected polynomial differs')
        if rule['method']=='power':
            converted=np.polynomial.chebyshev.cheb2poly(poly['coefficients'])
            if len(converted)>33 or not np.isfinite(converted).all() or np.abs(converted).sum()>1e8:
                raise ValueError('selected power polynomial exceeds the existing conditioning guard')
        coefficients=tuple(poly['coefficients']);domain=tuple(poly['domain'])
        if hashlib.sha256(np.asarray(coefficients,dtype='<f8').tobytes()).hexdigest()!=COEFFICIENT_SHA256[self.site]:raise ValueError('fixed inverse coefficients differ')
        for name,value in [('domain',domain),('coefficients',coefficients),('degree',poly['degree']),('method',rule['method'])]:object.__setattr__(self,name,value)
    @property
    def approximation(self):return PolynomialApproximation('inverse_sqrt',self.domain,self.coefficients)

FIELDS=('manifest_bytes','site','epsilon','domain','coefficients','degree','method')
def contract_state(contract):return tuple(typed(getattr(contract,n,None)) for n in FIELDS)
def verify_contract(contract):
    if not isinstance(contract,BertRowNormContract):raise TypeError('frozen row norm contract required')
    expected=BertRowNormContract(contract.manifest_bytes,contract.site,contract.epsilon)
    if contract_state(contract)!=contract_state(expected):raise ValueError('row norm contract fields changed')
    return contract_state(contract)

@dataclass(frozen=True)
class BertRowNormResult:
    output:object
    checkpoints:dict
    plan:dict
    original_positions:tuple


@dataclass(frozen=True)
class SelectedPolynomialView:
    """A fixed evaluation method over borrowed primitives, without state mutation."""
    arithmetic:object
    method:str
    def __post_init__(self):
        if type(self.method)is not str or self.method not in ('power','balanced_chebyshev'):
            raise ValueError('exact selected polynomial method required')
    @property
    def strategy(self):return 'power' if self.method=='power' else 'chebyshev'
    polynomial_depth=EncryptedPolynomialEvaluator.polynomial_depth
    evaluate=EncryptedPolynomialEvaluator.evaluate
    def _ensure_capacity(self,values,required):return self.arithmetic._ensure_capacity(values,required)
    def _level(self,value):return self.arithmetic._level(value)
    def _public_slots(self,value):return self.arithmetic._public_slots(value)
    def weighted_sum(self,*args,**kwargs):return self.arithmetic.weighted_sum(*args,**kwargs)
    def multiply(self,*args,**kwargs):return self.arithmetic.multiply(*args,**kwargs)


def row_norm_scale_plan(q_moduli,*,input_level,input_scale,canonical_scale,contract):
    verify_contract(contract);q=tuple(q_moduli)
    polynomial_depth=SelectedPolynomialView(None,contract.method).polynomial_depth(contract.approximation)
    depth=polynomial_depth+5
    if (type(input_level)is not int or input_level<depth+2 or len(q)!=2*(input_level+1)
            or len(set(q))!=len(q) or any(type(x)is not int or x<=1 for x in q)):
        raise ValueError('selected row norm needs its fixed depth plus two continuation levels')
    source=positive_scale(input_scale);delta=positive_scale(canonical_scale)
    weighted=[]
    def ws(name,scales,level):
        row=weighted_sum_scale(scales,delta,drop_divisor(q,level));row.update(name=name,input_level=level,output_level=level-1);weighted.append(row);return row['output_scale']
    mean=ws('mean',[source],input_level)
    if not math.isclose(source,mean,rel_tol=1e-12,abs_tol=0.):raise ValueError('input and computed mean exceed native HEAdd scale compatibility')
    centered=source # genuine subtraction preserves the left input scale
    squares=multiply_scale(centered,centered,drop_divisor(q,input_level-1))
    variance=ws('variance',[squares],input_level-2)
    t=ws('inverse_domain_map',[variance],input_level-3)
    powers={1:dict(power=1,level=input_level-4,scale=t)}
    products=[]
    def power(n):
        if n not in powers:
            a,b=power(n//2),power((n+1)//2);level=min(a['level'],b['level'])
            scale=multiply_scale(a['scale'],b['scale'],drop_divisor(q,level))
            products.append(dict(power=n,input_level=level,output_level=level-1,scale=scale))
            if contract.method=='power':powers[n]=dict(power=n,level=level-1,scale=scale)
            else:
                scales=[scale] if n%2==0 else [scale,t]
                powers[n]=dict(power=n,level=level-2,scale=ws(f'T{n}',scales,level-1))
        return powers[n]
    coefficients=(np.polynomial.chebyshev.cheb2poly(contract.coefficients)
                  if contract.method=='power' else contract.coefficients)
    indices=([i for i in range(1,len(coefficients)) if coefficients[i]!=0]
             if contract.method=='power' else list(range(1,len(coefficients))))
    values=[power(i) for i in indices]
    inverse=ws('inverse_combine',[p['scale'] for p in values],min(p['level'] for p in values))
    inverse_level=min(p['level'] for p in values)-1
    if inverse_level!=input_level-3-polynomial_depth:raise ValueError('selected inverse depth differs from actual recurrence')
    product=multiply_scale(centered,inverse,drop_divisor(q,inverse_level))
    output=ws('affine',[product],inverse_level-1)
    return dict(site=contract.site,input_level=input_level,input_scale=source,canonical_scale=delta,
        consumed_levels=depth,reserved_output_levels=2,polynomial_depth=polynomial_depth,method=contract.method,
        levels=dict(mean=input_level-1,centered=input_level-1,squares=input_level-2,variance=input_level-3,
                    inverse=inverse_level,normalized=inverse_level-1,output=inverse_level-2),
        scales=dict(mean=mean,centered=centered,squares=squares,variance=variance,inverse=inverse,normalized=product,output=output),
        weighted_sum_stages=weighted,power_stages=values,ciphertext_product_stages=products,logical_rotations=42,
        domain=list(contract.domain),epsilon=contract.epsilon,range_verified=False,
        conditional_scope='Only supplied valid compact rows; original positions preserved; no padded-query or CKKS domain guarantee.')


def _affine_vector(value,expected_hash):
    raw=np.asarray(value)
    if raw.shape!=(768,) or raw.dtype.kind not in 'fiu' or not np.isfinite(raw).all():raise ValueError('finite real pinned width768 vector required')
    with np.errstate(over='ignore',invalid='ignore'):lift=raw.astype(np.float64);f32=lift.astype('<f4')
    if not np.isfinite(lift).all() or not np.array_equal(lift,f32.astype(np.float64)) or hashlib.sha256(f32.tobytes()).hexdigest()!=expected_hash:
        raise ValueError('learned affine vector differs from the pinned checkpoint')
    return np.frombuffer(lift.astype('<f8').tobytes(),dtype='<f8')

def _array_state(value):return (type(value),value.dtype.str,value.shape,value.strides,value.flags.writeable,hashlib.sha256(value.tobytes()).digest())


class BertRowLayerNorm:
    """Serial row operator borrowing one initialized keyless CompiledArithmetic."""
    def __init__(self,arithmetic,manifest_bytes,gamma,beta,*,site='0.attention',epsilon=1e-12,progress=None):
        from encrypted_packed_qwen import CompiledArithmetic
        if not isinstance(arithmetic,CompiledArithmetic):raise TypeError('borrowed compiled arithmetic required')
        if progress is not None and not callable(progress):raise TypeError('progress must be callable or None')
        self.arithmetic=arithmetic;self.ctx=arithmetic.ctx;self.layout=RowTileLayout(SLOTS,4096)
        self.contract=BertRowNormContract(manifest_bytes,site,epsilon);self.approximation=self.contract.approximation
        self.inverse_evaluator=SelectedPolynomialView(arithmetic,self.contract.method)
        self.gamma=_affine_vector(gamma,AFFINE_SHA256[site][0]);self.beta=_affine_vector(beta,AFFINE_SHA256[site][1])
        self.progress=progress;self.last_run=None;self._active=False
        # Reuse the measured exact prefix/broadcast primitives without creating
        # another compiled owner or taking ownership of the shared context.
        self._core=RowLayerNorm.__new__(RowLayerNorm);self._core.ctx=self.ctx;self._core.layout=self.layout;self._core.arithmetic=arithmetic
        self._bound=(arithmetic,self.ctx,self.ctx._param_cache,self.layout,self.contract,self.approximation,self.gamma,self.beta,self._core,self.inverse_evaluator,progress)
        self._contract=verify_contract(self.contract);self._affine=(_array_state(self.gamma),_array_state(self.beta))
        self._context=self._context_state();self._approx=typed((self.approximation.name,self.approximation.domain,self.approximation.coefficients,self.approximation.input_power_basis))
        self._rotation_keys=self.ctx._param_cache.raw_rotation_keys
        self._coefficient_maps=self.ctx._param_cache.coef_maps
        self._guard()
    @staticmethod
    def required_rotation_indices():return RowLayerNorm.required_rotation_indices(RowTileLayout(SLOTS,4096))
    def _context_state(self):
        c=self.ctx;k=c._param_cache;a=self.arithmetic
        return (tuple(typed(getattr(obj,n,None)) for obj,names in ((c,('degree','num_slots','composite_degree','dnum','degree_layout')),(k,('degree','batch','composite_degree','dnum','degree_layout'))) for n in names),
                typed(tuple(k.q_towers)),typed(tuple(k.p_towers)),typed(a.scale),typed(getattr(a,'strategy',None)),id(getattr(a,'progress',None)))
    def _guard(self):
        now=(self.arithmetic,self.ctx,self.ctx._param_cache,self.layout,self.contract,self.approximation,self.gamma,self.beta,self._core,self.inverse_evaluator,self.progress)
        if any(a is not b for a,b in zip(now,self._bound)):raise ValueError('construction-bound row norm owner changed')
        c=self.ctx;k=c._param_cache
        if (self.arithmetic.ctx is not c or self._core.ctx is not c or self._core.arithmetic is not self.arithmetic or self._core.layout is not self.layout
                or c.secret_key is not None or k.secret_key is not None or c.parameters.get('secret_key') is not None
                or self._context_state()!=self._context or contract_state(self.contract)!=self._contract
                or (_array_state(self.gamma),_array_state(self.beta))!=self._affine
                or typed((self.approximation.name,self.approximation.domain,self.approximation.coefficients,self.approximation.input_power_basis))!=self._approx
                or getattr(self.arithmetic,'refresher',None) is not None or self.arithmetic.strategy!='power'
                or self.inverse_evaluator.arithmetic is not self.arithmetic
                or typed(self.inverse_evaluator.method)!=typed(self.contract.method)
                or callable(getattr(self.arithmetic,'declare_bound',None))
                or k.raw_rotation_keys is not self._rotation_keys or k.coef_maps is not self._coefficient_maps
                or typed((self.layout.slots,self.layout.features))!=typed((SLOTS,4096))
                or type(c.degree)is not int or c.degree!=2*SLOTS or type(c.num_slots)is not int or c.num_slots!=SLOTS
                or type(c.composite_degree)is not int or c.composite_degree!=2 or type(k.batch)is not int or k.batch!=1
                or type(k.degree)is not int or k.degree!=c.degree or type(k.composite_degree)is not int or k.composite_degree!=2
                or type(c.dnum)is not int or c.dnum<1 or typed(c.dnum)!=typed(k.dnum)
                or typed(c.degree_layout)!=typed(k.degree_layout)):
            raise ValueError('row norm context, layout or public contract changed')
        keys,maps=self._rotation_keys,self._coefficient_maps
        if (getattr(k,'per_level_rotation_keys',None) is not False
                or not isinstance(keys,Mapping) or not isinstance(maps,Mapping)):
            raise ValueError('supplied reusable row-normalization public rotation keys and coefficient maps required')
        for index in self.required_rotation_indices():
            pair=keys.get(index)
            if (not isinstance(pair,Mapping) or pair.get('a') is None or pair.get('b') is None or maps.get(index) is None):
                raise ValueError('all row-normalization public rotation keys and coefficient maps must be supplied')
    def _fingerprint(self,value):
        return (typed(self.arithmetic._level(value)),tuple(typed(getattr(value,n,None)) for n in ('_ckks_scale','_ckks_nsd','moduli','batch','num_elements','degree','degree_layout')),id(value.polynomial),id(getattr(value,'ntt_ctx',None)))
    def level_scale_plan(self,q_moduli,*,input_level,input_scale):
        self._guard();return row_norm_scale_plan(q_moduli,input_level=input_level,input_scale=input_scale,canonical_scale=self.arithmetic.scale,contract=self.contract)
    def normalize_tile(self,value,*,valid_rows,original_positions,capture_checkpoints=False):
        self._guard()
        if self._active:raise ValueError('row norm cannot be reentered')
        self.layout._valid_rows(valid_rows)
        if (type(original_positions)is not tuple or len(original_positions)!=valid_rows
                or any(type(x)is not int or not 0<=x<128 for x in original_positions)
                or tuple(sorted(set(original_positions)))!=original_positions):raise ValueError('explicit increasing original positions for every compact valid row required')
        if type(capture_checkpoints)is not bool:raise TypeError('capture must be explicit bool')
        if type(getattr(value,'_ckks_nsd',None))is not int or value._ckks_nsd!=1:raise ValueError('explicit integer NSD1 required')
        if (any(type(getattr(value,n,None))is not int or getattr(value,n)!=expected for n,expected in
                (('batch',1),('num_elements',2),('degree',2*SLOTS)))
                or typed(value.degree_layout)!=typed(self.ctx.degree_layout)):
            raise ValueError('exact batch-one two-component ring layout required')
        depth=self.inverse_evaluator.polynomial_depth(self.approximation)+5
        # The borrowed validator adds one reserved level to its argument.
        # Passing depth+1 therefore requires exactly depth+2 actual levels.
        self._core._validate_ciphertext(value,depth+1)
        plan=self.level_scale_plan(self.ctx._param_cache.q_moduli_at_level(self.arithmetic._level(value)),input_level=self.arithmetic._level(value),input_scale=value._ckks_scale)
        source=value;state=self._fingerprint(value);retained={};captured={};captured_states={};self._active=True;self.last_run=None
        def forget(result):
            for key in tuple(vars(result)):
                if key.startswith('_public_'):delattr(result,key)
        def guard():
            self._guard()
            if (self._fingerprint(source)!=state or any(self._fingerprint(v)!=s for v,s in retained.values())
                    or any(self._fingerprint(v)!=captured_states[name] for name,v in captured.items())):
                raise ValueError('row norm input, retained or captured ciphertext changed')
        def note(name,result):
            guard()
            if self.arithmetic._level(result)!=plan['levels'][name] or result._ckks_scale!=plan['scales'][name] or type(result._ckks_nsd)is not int or result._ckks_nsd!=1:raise ValueError('row norm metadata differs from actual public scale plan')
            forget(result)
            fingerprint=self._fingerprint(result)
            if self.progress is not None:self.progress('layer '+self.contract.site+' '+name,result,dict(level=plan['levels'][name],original_positions=original_positions))
            guard()
            if self._fingerprint(result)!=fingerprint:raise ValueError('row norm callback changed its result')
            # Callbacks may add diagnostic attributes, but no semantic interval
            # contract has been proved for any value returned by this owner.
            forget(result)
            for previous in captured.values():forget(previous)
            if capture_checkpoints and name in ('mean','variance','inverse'):
                captured[name]=result;captured_states[name]=self._fingerprint(result)
            return result
        try:
            a=self.arithmetic;anchors=np.zeros((16,4096));anchors[:valid_rows,0]=1/768
            mean=note('mean',self._core._broadcast(a.weighted_sum([self._core._prefix768(value)],[anchors.ravel()]),full=False))
            centered=note('centered',a.add(value,mean,subtract=True));del mean,value
            retained['centered']=(centered,self._fingerprint(centered))
            squares=note('squares',a.multiply(centered,centered));bias=np.zeros_like(anchors)
            bias[:valid_rows,0]=self.contract.epsilon;bias[valid_rows:,0]=sum(self.contract.domain)/2
            variance=note('variance',self._core._broadcast(a.weighted_sum([self._core._prefix768(squares)],[anchors.ravel()],bias=bias.ravel()),full=True));del squares
            inverse=note('inverse',self.inverse_evaluator.evaluate(variance,self.approximation))
            normalized=note('normalized',a.multiply(centered,inverse));retained.clear();del centered
            output=note('output',a.weighted_sum([normalized],[self.layout.feature(self.gamma,valid_rows=valid_rows)],bias=self.layout.feature(self.beta,valid_rows=valid_rows)));del normalized
            checkpoints=dict(captured)
            self.last_run=dict(plan=plan,original_positions=original_positions,valid_rows=valid_rows,site=self.contract.site)
            return BertRowNormResult(output,checkpoints,plan,original_positions)
        finally:self._active=False

