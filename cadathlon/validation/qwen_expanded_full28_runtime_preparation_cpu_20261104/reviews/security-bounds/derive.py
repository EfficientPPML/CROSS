"""Public metadata/scalar derivation only. No runtime imports or model IO."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
import hashlib,json,math

HERE=Path(__file__).resolve().parent
OLD=Path('/tmp/qwen-full-model-runtime-v3-20261103')
NEW=Path('/tmp/qwen-expanded-candidate-checker-ready-v2-20261104/candidate_proof/candidate-manifest.json')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
read=lambda p:p.read_bytes()
oldraw=read(OLD/'inputs/candidate_manifest.json');newraw=read(NEW)
assert sha(oldraw)=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
assert sha(newraw)=='344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
old=json.loads(oldraw);new=json.loads(newraw)
sites=json.loads(read(OLD/'inputs/site_metadata.json'))
joined=json.loads(read(OLD/'inputs/join_plan.json'));rows={row['site']:row for row in joined['normalizations']}
assert len(sites)==len(rows)==113
def ceil_sqrt(x):
    y=math.sqrt(float(x))
    while F.from_float(y)**2<x:y=math.nextafter(y,math.inf)
    return F.from_float(y)
def power2_guard(x):
    target=F(x)*F(9,8);result=1
    while result<target:result*=2
    return result
def inverse_bound(rule):
    lo,hi=map(F.from_float,rule['domain'])
    if rule['method']=='newton':
        w,b=map(F.from_float,(rule['rounded_first_weight'],rule['rounded_first_bias']))
        value=max(w*lo+b,w*hi+b,ceil_sqrt(1/lo))
    else:value=F(1000001,1000000)*ceil_sqrt(1/lo)
    return power2_guard(value)
raw_changes=[];inverse_changes=[];rule_changes=[];schedule_changes=[]
for site,metadata in sites.items():
    left=old['nonlinear'][site];right=new['nonlinear'][site]
    if left!=right:rule_changes.append(site)
    if any(left.get(k)!=right.get(k) for k in ('method','degree','iterations')):schedule_changes.append(site)
    width=metadata['group_width']
    b0=power2_guard(ceil_sqrt(width*F.from_float(left['domain'][1])))
    b1=power2_guard(ceil_sqrt(width*F.from_float(right['domain'][1])))
    assert b0==rows[site]['raw_input_bound']
    if b0!=b1:raw_changes.append(dict(site=site,old=b0,new=b1))
    i0,i1=inverse_bound(left),inverse_bound(right)
    assert all(event['public_bound']==i0 for event in metadata['refresh_events'])
    if i0!=i1:inverse_changes.append(dict(site=site,old=i0,new=i1))
assert len(rule_changes)==90 and schedule_changes==['final.inverse_sqrt']
assert raw_changes==[dict(site='final.inverse_sqrt',old=2048,new=8192)]
assert inverse_changes==[]
assert sum(len(m['refresh_events']) for m in sites.values())==260
assert all(new[k]==old[k] for k in ('exp','initial_reciprocal','reciprocal_by_keys','softmax'))
assert all(new['nonlinear'][s]==old['nonlinear'][s] for s in old['nonlinear'] if s.endswith('.silu'))
rule=new['nonlinear']['final.inverse_sqrt'];assert (rule['degree'],rule['depth'])==(256,18)
lo,hi=map(F.from_float,rule['domain']);epsilon=F.from_float(1e-6)
square=1024*(hi-epsilon);conservative=ceil_sqrt(1024*hi)
assert F(4096)**2<square<F(8192)**2
witness=F(4096**2,1024)+epsilon;assert lo<=witness<=hi and 4096>2048
gamma=F.from_float(rows['final.inverse_sqrt']['gamma_max_abs']);assert gamma==F(245,16)
normalized=gamma*32*F(1000001,1000000)
assert power2_guard(normalized)==1024
pins={str(NEW):sha(newraw)}
relative=('full_contract.py','checkout/cadathlon/common/encrypted_qwen_full_model.py',
    'checkout/cadathlon/common/encrypted_qwen_rmsnorm.py','checkout/cadathlon/common/encrypted_qwen_head_routing.py',
    'checkout/cadathlon/common/encrypted_qwen_attention_join.py','checkout/cadathlon/common/encrypted_qwen_decoder_tail.py',
    'checkout/cadathlon/common/qwen_silu_contract.py','checkout/cadathlon/common/encrypted_qwen_causal_softmax.py',
    'inputs/candidate_manifest.json','inputs/site_metadata.json','inputs/certificates.json','inputs/join_plan.json',
    'inputs/decoder_parameters.json','inputs/actual_key_plan.json')
for name in relative:pins[str(OLD/name)]=sha(read(OLD/name))
for path in ('/tmp/qwen-candidate3-full-he-plan-20261103/plan.py','/tmp/qwen-candidate3-joined-continuation-20261103/plan.py'):
    pins[path]=sha(read(Path(path)))
result=dict(schema='qwen-expanded-full-contract-public-derivation-v1',passed=True,utc=datetime.now(timezone.utc).isoformat(),
    scope='Independent source/public-metadata scalar derivation; no runtime imports, backend, model/parameter/input/outcome/private bodies, execution grant or numerical calibration result.',
    source_hashes=pins,changed_norm_rules=rule_changes,changed_schedule_sites=schedule_changes,
    raw_bound_changes=raw_changes,inverse_bound_changes=inverse_changes,internal_norm_refresh_events=260,
    final=dict(variance_domain=rule['domain'],degree=256,abstract_polynomial_depth=18,
        actual_schedule=dict(source=25,square=24,variance=23,map=22,T256=6,inverse=5,normalized=4,gamma_output=3,vocabulary_output=2),
        internal_inverse_refresh_required=False,
        same_input_amplitude_squared_exact=[str(square.numerator),str(square.denominator)],
        same_input_amplitude_approx=math.sqrt(float(square)),
        inherited_conservative_unrounded=math.sqrt(float(1024*hi)),
        inherited_guarded_upper=float(conservative*F(9,8)),public_refresh_bound=8192,
        old_bound_counterexample=dict(nonzero_coordinate=4096,other1023_coordinates=0,variance_approx=float(witness),inside_new_variance_domain=True,exceeds_old2048=True),
        gamma_max_abs=float(gamma),conditional_normalized_abs_upper=float(normalized),conditional_normalized_refresh_bound=1024,
        basis_operation_delta=dict(ciphertext_products=192,weighted_sums=192,rescales=384,rotations=0),
        count_scope='Balanced evaluation retains T1..Tdegree; same final width1024. Abstract DAG operation delta, not measured performance.'),
    required_successor_bindings=[
        'Fresh113-site metadata identity with final outputlevel3 and all domain/first-affine/certificate relationships checked against exact344954; no copied old-manifest success flag.',
        'Fresh continuation/join/full plan with final raw8192 and final3/head2 levels, actual incoming-scale/drop-prime replay, resource and operation counts; retain all old raw/error/domain conditions.',
        'Full contract VocabularyBlock7/6 and runtime metadata/drop_divisor(q,7)/output6 constants must become reviewed candidate-specific3/2 values; generic RMS accepts256 but old metadata/pins reject it.',
        'Preserve old protected certificate ad386 unchanged; new applicability bridge must authenticate old/new manifest digests and exact protected rule equality, bind old certificate identity and explicit conditional scope.',
        'Preserve immutable QKV/decoder/gamma/keyplan artifacts; explicit checkpoint/parameter provenance bridge must reconcile their old candidate labels with new candidate identity without changing array hashes, tensor shapes, tied aliases or key membership.',
        'Bind the new manifest/site/join/bridge sources through all child/full owners, source closures and tests. Old87 proofs stay old87 and do not establish344954 runtime readiness.',
    ],
    scientific_limits=['Variance relation is exact-real same-input only; refresh/CKKS/binary64 perturbation and domain-transfer obligations remain separate.',
        'B8192 is only a public amplitude envelope. Root reports the current secureP60 refresh component covers B16/B128 and is still running; that does not establish precision at B8192 or oldB2048. No encrypted refresh-error guarantee or actual last-headL2 support is inferred. Future actual full28/component evidence remains required; no new launch is authorized.',
        'Uniform normalization certificates and unchanged protected rules are not a3200-context calibration PASS, row0 acceptance, full-model trajectory pass or HE correctness proof.',
        'No new coefficient selection or changes to the selected polynomial/Newton counts are proposed.'])
(HERE/'review.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(passed=True,review_sha256=sha(read(HERE/'review.json')),changed_norm_rules=90,final_raw_bound=8192,final_output_level=3,vocabulary_output_level=2),sort_keys=True))
