"""Narrow public applicability bridge, never a calibration or execution grant.

The original certificate and parameter manifests retain their historical bytes
and candidate identity. Only the identical protected rules use that certificate;
the 113 changed normalization contracts require newly derived certificates.
"""
import hashlib,json

OLD_MANIFEST='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
MANIFEST='344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
CONFIG='660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'
CHECKPOINT='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
PROTECTED='13ace05723150ba282514220c94432335a291f7421d2e0991295a94082a9b58b'
CERTIFICATE='ad386cb499682a0abb54f81a26d67136b2df0da26383b4b140184344de904a37'
PARAMETERS={
    'qkv':'c707068e1a247f4ef14fda3493a3e2edf9dea2f11ec19e94531a64d80dbcf48b',
    'decoder':'60c556376e42358f227a2978edb6b2addf59dc9e694b65e17265b15e813d65c3',
}
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(raw):return hashlib.sha256(raw).hexdigest()
def pinned(raw,pin):
    if type(raw)is not bytes or digest(raw)!=pin:raise ValueError('exact public bridge bytes required')
    return json.loads(raw)
def protected(manifest):
    return dict(nonlinear={k:v for k,v in manifest['nonlinear'].items() if not k.endswith('inverse_sqrt')},
        **{k:manifest[k] for k in ('softmax','exp','initial_reciprocal','reciprocal_by_keys')})
def candidate(raw):
    value=pinned(raw,MANIFEST)
    if digest(canonical(protected(value)))!=PROTECTED:raise ValueError('protected rule identity differs')
    return value
def verify_protected_certificate(manifest_bytes,certificate_bytes):
    candidate(manifest_bytes);proof=pinned(certificate_bytes,CERTIFICATE)
    if proof['passed']is not True or proof['manifest_sha256']!=OLD_MANIFEST:raise ValueError('historical protected certificate identity differs')
    return proof
def verify_parameters(manifest_bytes,parameter_bytes,role):
    candidate(manifest_bytes)
    if type(role)is not str or role not in PARAMETERS:raise ValueError('unknown parameter role')
    value=pinned(parameter_bytes,PARAMETERS[role])
    if value['candidate_sha256']!=OLD_MANIFEST or value['config_sha256']!=CONFIG or value['checkpoint_sha256']!=CHECKPOINT:raise ValueError('historical parameter identity differs')
    return value
def derive_bridge(old_bytes,new_bytes,config_bytes,certificate_bytes,qkv_bytes,decoder_bytes):
    old=pinned(old_bytes,OLD_MANIFEST);new=candidate(new_bytes);pinned(config_bytes,CONFIG)
    if canonical(protected(old))!=canonical(protected(new)):raise ValueError('protected rules changed')
    verify_protected_certificate(new_bytes,certificate_bytes)
    for role,raw in (('qkv',qkv_bytes),('decoder',decoder_bytes)):verify_parameters(new_bytes,raw,role)
    return dict(schema='qwen-expanded-candidate-public-bridge-v1',old_manifest_sha256=OLD_MANIFEST,
        candidate_manifest_sha256=MANIFEST,config_sha256=CONFIG,checkpoint_sha256=CHECKPOINT,
        protected_rules_sha256=PROTECTED,protected_counts=dict(silu=28,softmax=28,reciprocal_by_keys=32,exp=1,initial_reciprocal=1),
        historical_certificate_sha256=CERTIFICATE,historical_parameter_manifest_sha256=dict(PARAMETERS),
        protected_certificate_applicable=True,normalization_certificate_transferred=False,
        historical_bytes_rewritten=False,calibration_pass_claimed=False,execution_authorized=False,
        scope='Exact public rule and parameter identity only; no new model/calibration/HE result. Historical certificate excludes FP/CKKS/refresh perturbations.')
