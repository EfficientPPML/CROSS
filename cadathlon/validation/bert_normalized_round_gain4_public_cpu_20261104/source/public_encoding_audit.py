"""Production host-FFT/quantization audit, stopped before RNS/NTT construction.

Only fixed public plaintext constants are examined. The exact production AST
is executed with a public Q-list cache and a capture at the RNS boundary.
No CKKSContext, backend, key, ciphertext or encryption routine is invoked.
The rounding enclosure starts from the computed binary64 FFT coefficients;
it is not an enclosure of exact-real FFT error or encrypted circuit noise.
"""
from pathlib import Path
from types import SimpleNamespace as NS
from fractions import Fraction
from functools import lru_cache
import ast,cmath,hashlib,json,math,sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'checkout/cadathlon/common'))
from encrypted_bert_normalization_gain4 import gain4_round_plan,DELTA
from encrypted_bert_normalization_round import NormalizationRoundContract,normalization_round_plan
from encrypted_score_reduction import ScoreReductionLayout
from fixtures import definitions

FUNCTIONS=('_roots','_rot_group','_inverse_fft_plan','_native_special_inverse_fft','nearest_int','slot_to_coeffs','_validated_slots','_fast_encode')

def extract(path,names,namespace):
    tree=ast.parse(path.read_bytes());selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names]
    if {n.name for n in selected}!=set(names):raise ValueError('exact public AST functions missing')
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return {n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()for n in selected}

def public_host_encoder():
    core=HERE/'checkout/jaxite_word/ckks_ctx.py';util=HERE/'checkout/jaxite_word/util.py'
    pins=json.loads((HERE/'gain4_source_origin.json').read_bytes())['predecessor_source_hashes']
    for p in (core,util):
        if hashlib.sha256(p.read_bytes()).hexdigest()!=pins[p.relative_to(HERE).as_posix()]:raise ValueError('frozen encoder source changed')
    u={'_bit_reverse_permutation_cache':{}}
    util_ast=extract(util,('bit_reverse','_bit_reverse_perm'),u)
    ns=dict(np=np,math=math,cmath=cmath,lru_cache=lru_cache,util=NS(**{n:u[n]for n in ('bit_reverse','_bit_reverse_perm')}),ScalingFactorTooSmall=ValueError)
    core_ast=extract(core,FUNCTIONS,ns);records=[]
    ns['_get_encrypt_cache']=lambda ctx,M:NS(q_int=list(ctx.q_towers))
    ns['_vectorized_ntt']=lambda captured,cache:captured
    original_fft=ns['_native_special_inverse_fft']
    def fft(values,order):
        result=original_fft(values,order);ns['_observed_coefficients']=ns['slot_to_coeffs'](result);return result
    ns['_native_special_inverse_fft']=fft
    def capture(coefficients,bound,moduli,shift,scale,noise_scale_degree,*,ring_dim):
        if noise_scale_degree!=1 or ring_dim!=131072:raise ValueError('fixed public encoder shape required')
        signed=[int(v)-bound if int(v)>(bound>>1)else int(v)for v in coefficients]
        max_integer=max(map(abs,signed))<<max(0,shift);Q=math.prod(moduli)
        coeff=ns['_observed_coefficients'];scaled=[scale*x for x in coeff]
        if not all(math.isfinite(x)for x in scaled):raise ValueError('nonfinite scaled coefficient')
        maximum=max(map(abs,scaled));quantum=1<<max(0,shift)
        # Correctly rounded binary64 multiplication, followed by the exact
        # power-of-two reduction and original nearest-integer quantizer.
        bound_fraction=(Fraction(quantum,2)+Fraction.from_float(math.ulp(maximum))/2)/Fraction.from_float(scale)
        error_bound=float(bound_fraction)
        if Fraction.from_float(error_bound)<bound_fraction:error_bound=math.nextafter(error_bound,math.inf)
        observed=max(abs((v*quantum)/scale-c)for v,c in zip(signed,coeff))
        row=dict(level=(len(moduli)//2)-1,scale=scale,coefficient_count=len(coefficients),log_approx=shift,
            significant_word_limit=61,nonzero_quantized_coefficients=sum(v!=0 for v in signed),
            maximum_scaled_coefficient=maximum,maximum_quantized_integer_bits=max_integer.bit_length(),
            signed_Q_bits=Q.bit_length()-1,coefficient_headroom_bits=Q.bit_length()-1-max_integer.bit_length(),
            exact_integer_headroom_passed=2*max_integer<Q,
            rounding_error_enclosure_from_binary64_fft=error_bound,observed_float_reconstruction_max_abs=observed,
            quantized_coefficients_sha256=hashlib.sha256(json.dumps(signed,separators=(',',':')).encode()).hexdigest(),
            scope='Production FFT/quantization only; RNS/NTT replaced by a capture, no full encoding or HE claim')
        if not row['exact_integer_headroom_passed']or not row['nonzero_quantized_coefficients']:raise ValueError('public coefficient headroom/resolution failed')
        records.append(row);return row
    ns['_native_encode_residues']=capture
    def encode(slots,q,scale):
        ctx=NS(degree=131072,num_slots=65536,q_towers=q,scaling_factor=scale,parameters={'max_bits_in_word':61,'max_bits_value':(1<<63)-(1<<9)-1})
        return ns['_fast_encode'](slots,ctx,scale=scale)
    return encode,ns['nearest_int'],dict(core=core_ast,util=util_ast)

def audit():
    profile=next(r for r in json.loads((HERE/'actual_profile_audit.json').read_bytes())['profiles']if r['name']=='depth49_p60_d8');q=tuple(profile['q']);raw=(HERE/'candidate_manifest.json').read_bytes()
    encode,nearest,ast_pins=public_host_encoder();rows=[];uniform=[]
    for fixture in definitions():
        layout=ScoreReductionLayout(fixture['key_mask'],valid_rows=fixture['valid_rows']);contract=NormalizationRoundContract(raw,fixture['key_count'])
        for incoming in (math.nextafter(DELTA,0.),DELTA,math.nextafter(DELTA,math.inf)):
            for gain in (1,4):
                plan=(normalization_round_plan if gain==1 else gain4_round_plan)(q[:52],input_level=25,canonical_scale=DELTA,contract=contract,input_scale=incoming)
                targets=(gain*DELTA,gain*gain*DELTA)
                inputs=(incoming,plan['scales']['squared_numerator'])
                for label,slots,L,target,source in (('mask',layout.input_mask,25,targets[0],inputs[0]),('anchor',layout.anchors,23,targets[1],inputs[1]),('dummy_bias',layout.dummy_anchors,23,targets[1],inputs[1])):
                    pt_scale=target*(q[2*L]*q[2*L+1])/source
                    if label=='dummy_bias':pt_scale=source*pt_scale
                    row=encode(slots,q[:2*(L+1)],pt_scale)
                    rows.append(dict(fixture=fixture['name'],gain=gain,input_scale=incoming,role=label,**row))
                inverse=plan['tracked_inverse_scale_plan'];L=22
                for label,value,scale in (('first_weight',contract.first_weight,inverse['initial_plaintext_scale']),('first_bias',contract.first_bias,plan['scales']['denominator']*inverse['initial_plaintext_scale'])):
                    coefficient=nearest(value*scale);Q=math.prod(q[:2*(L+1)])
                    if coefficient==0 or 2*abs(coefficient)>=Q:raise ValueError('uniform scalar resolution/headroom failed')
                    uniform.append(dict(fixture=fixture['name'],gain=gain,input_scale=incoming,role=label,scale=scale,coefficient_bits=abs(coefficient).bit_length(),signed_Q_bits=Q.bit_length()-1,headroom_passed=True,nonzero=True,scope='Actual uniform _encode_public nearest_int scalar rule; no plaintext allocation'))
    return dict(passed=True,scope=__doc__,nonuniform_records=rows,uniform_records=uniform,source_ast_sha256=ast_pins,
        limitations=['FFT floating error is outside the rounding enclosure.','Gain may reach the61-bit coefficient cap; no fourfold encoded or encrypted precision improvement is presumed.','No NTT/RNS/backend/keygen/ciphertext/model run.','Existing input error, keyswitch/rescale error and branch correlations remain unbounded here.'])

if __name__=='__main__':print(json.dumps(audit(),indent=2,allow_nan=False))
