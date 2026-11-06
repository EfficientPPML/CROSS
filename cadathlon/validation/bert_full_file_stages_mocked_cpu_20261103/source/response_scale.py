"""Independent public enclosure for the frozen final classifier's L2 scale.

This authenticates the fixed profile/manifest, not a server-provided expected
scale. It encloses the existing binary64 operation order for *any* finite
positive source scale for which production positive_scale accepts every
intermediate. It does not certify ciphertext payloads or numerical accuracy.
"""
from fractions import Fraction
import hashlib,json,math,sys
PROFILE_SHA256='fe2408637f9d6de19ac6aef9f6f749aaef9dd56ccbb37b5334293cd223ede035'
MANIFEST_SHA256='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'


def scale_certificate(*,profile_bytes,manifest_bytes):
    for raw,pin in ((profile_bytes,PROFILE_SHA256),(manifest_bytes,MANIFEST_SHA256)):
        if type(raw)is not bytes or hashlib.sha256(raw).hexdigest()!=pin:
            raise ValueError('exact frozen profile and candidate bytes required')
    if sys.float_info.radix!=2 or sys.float_info.mant_dig!=53 or sys.float_info.rounds!=1:
        raise ValueError('IEEE binary64 round-to-nearest required')
    profile=json.loads(profile_bytes)['profile'];manifest=json.loads(manifest_bytes)
    if (profile['degree']!=131072 or profile['depth']!=49 or profile['dnum']!=8
            or len(profile['q'])!=100 or manifest['tanh']['doublings']!=6
            or manifest['tanh_reciprocal']['iterations']!=5):
        raise ValueError('fixed classifier source level3 contract differs')
    delta=profile['canonical_scale'];divisor=profile['q'][6]*profile['q'][7]
    # Exactly weighted_sum -> public PTCT -> CompiledRescale, in Python's
    #float/int conversion order. The first summed term owns the metadata.
    target=delta*divisor
    u=Fraction(1,2**53);T=Fraction.from_float(target)
    lower=T*(1-u)**2;upper=T*(1+u)**2
    minimum=Fraction.from_float(sys.float_info.min);maximum=Fraction.from_float(sys.float_info.max)
    # T/s cannot be subnormal for any finite positive s. Overflow is rejected
    #by production positive_scale. For an accepted quotient, the next exact
    #product is close to T, so it too is normal and cannot overflow.
    if not (T/maximum>minimum and T*(1-u)>minimum and T*(1+u)<maximum
            and lower/Fraction.from_float(float(divisor))>minimum
            and upper/Fraction.from_float(float(divisor))<maximum):
        raise ValueError('normal finite intermediate proof failed')
    start=float(lower)
    while Fraction.from_float(start)<lower:start=math.nextafter(start,math.inf)
    products=[];current=start
    while Fraction.from_float(current)<=upper:
        products.append(current)
        if len(products)>16:raise ValueError('unexpectedly broad binary64 product enclosure')
        current=math.nextafter(current,math.inf)
    outputs=tuple(sorted(set(product/divisor for product in products)))
    if not outputs or any(not math.isfinite(v) or v<=0 for v in outputs):raise ValueError('invalid classifier scale enclosure')
    return dict(input_level=3,output_level=2,divisor=divisor,float_divisor=float(divisor),canonical_scale=delta,
                target_before_rescale=target,product_candidates=tuple(products),accepted_scales=outputs,
                normal_intermediates_proved=True,first_term_metadata=True,
                scope='Enclosing binary64 metadata set; no reachability, payload correctness, CKKS error or refresh-accuracy claim.')


def accepts_scale(scale,*,profile_bytes,manifest_bytes):
    """False for invalid scalar types/values or unbound public artifacts."""
    if type(scale)is not float or not math.isfinite(scale) or scale<=0:return False
    try:return scale in scale_certificate(profile_bytes=profile_bytes,manifest_bytes=manifest_bytes)['accepted_scales']
    except (ValueError,TypeError,OverflowError,KeyError):return False
