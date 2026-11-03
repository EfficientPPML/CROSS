"""Exact public GELU64 envelope and optional refresh relocation; no HE execution."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,io,json,math,os,struct,tarfile,time
HERE=Path(__file__).resolve().parent
REPO=Path(os.environ.get('CROSS_ROOT','/home/jianming_gatech/CROSS'))
MANIFEST=REPO/'cadathlon/validation/mrpc_training_domain_candidate_20261103/explicit_v2_prelaunch/candidate-v2/candidate_manifest.json'
PRIOR=Path('/tmp/bert-first-block-v2-integration-plan-20261103/report.json')
EVALUATOR=REPO/'cadathlon/common/encrypted_polynomial.py'

def sha(data):return hashlib.sha256(data).hexdigest()
def outward(value):
    result=float(value)
    if F.from_float(result)<value:result=math.nextafter(result,math.inf)
    return result

def fraction(value):return dict(numerator=str(value.numerator),denominator=str(value.denominator),upper_binary64=outward(value))
def value_at(coefficients,t):
    # Independent exact Chebyshev recurrence, no float evaluation or native GELU oracle.
    before,current=F(1),t
    result=coefficients[0]+coefficients[1]*t
    for coefficient in coefficients[2:]:
        before,current=current,2*t*current-before
        result+=coefficient*current
    return result

def main():
    start=time.monotonic();inputs=[Path(__file__),MANIFEST,PRIOR,EVALUATOR]
    before={str(p):sha(p.read_bytes()) for p in inputs}
    assert before[str(MANIFEST)]=='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
    manifest=json.loads(MANIFEST.read_text());entry=manifest['gelu']['0'];poly=entry['polynomial']
    floats=poly['coefficients'];coefficients=[F.from_float(c) for c in floats]
    assert entry['degree']==64 and entry['domain']==[-21,14]
    assert sha(struct.pack('<65d',*floats))==poly['coefficient_f64le_sha256']
    l1=sum(map(abs,coefficients),F());derivative=sum((abs(c)*n*n for n,c in enumerate(coefficients)),F())
    a,b=F.from_float(2/35),F.from_float(7/35)
    endpoints=[-21*a+b,14*a+b]
    assert -1<=endpoints[0]<endpoints[1]<=1
    count=4096
    grid=[value_at(coefficients,F(-1)+F(2*j,count)) for j in range(count+1)]
    largest=max(map(abs,grid));argmax=max(range(len(grid)),key=lambda i:abs(grid[i]))
    # Nearest point is at most1/count away on a grid of spacing2/count.
    envelope=largest+derivative/count
    padded=envelope*F(9,8)
    assert l1>16 and padded<16
    profiles=[]
    for old in json.loads(PRIOR.read_text())['profiles']:
        R=old['refresh_output'];tail=[('attention_norm_refresh',R),('first_linear',R-1),('GELU64',R-15),
             ('GELU_output_refresh',R),('output_linear',R-1),('residual_add',R-1),('MLP_norm',R-12)]
        assert R-15>=2 and R-12>=2
        previous_tail_refreshes=1+int(R<29)
        modes={}
        for name,mode in old['modes'].items():
            k128=mode['K128']
            modes[name]=dict(inherited_all_K_level_feasible=mode['all_K_level_feasible'],
                inherited_K128_total_refreshes=k128.get('total_refreshes_per_query_tile'),
                optional_relocated_K128_total_refreshes=(k128['total_refreshes_per_query_tile']-previous_tail_refreshes+2) if k128.get('feasible') else None,
                optional_relocated_output_level=R-12 if k128.get('feasible') else None)
        profiles.append(dict(name=old['name'],depth=old['depth'],scale_bits=old['scale_bits'],dnum=old['dnum'],
            Q_count=old['Q_count'],P_count=old['P_count'],log2_QP=old['log2_QP'],refresh_output=R,
            graph_plus_refresh_keys_unchanged=old['direct_graph_plus_refresh_keys'],raw_direct_rotation_store_GiB=old['raw_direct_rotation_store_GiB'],
            old_tail_refreshes=previous_tail_refreshes,new_tail_refreshes=2,tail=tail,modes=modes,
            note='At output29 the old no-intermediate-refresh tail already works; forcing this optional refresh adds one and is unnecessary.'))
    report=dict(scope=__doc__,passed=True,manifest_sha256=before[str(MANIFEST)],coefficient_sha256=poly['coefficient_f64le_sha256'],
        exact_binary64_coefficient_l1=fraction(l1),l1_with_12_5_percent_margin=fraction(l1*F(9,8)),
        simple_l1_certifies_bound16=False,actual_evaluation='Balanced Chebyshev identities; no cheb2poly/monomial conversion.',
        rounded_normalization=dict(weight=fraction(a),bias=fraction(b),mapped_endpoints=[fraction(x) for x in endpoints],
            image_subset_minus1_plus1=True,statement='Binary64 public affine coefficients interpreted exactly; encryption/encoding/arithmetic errors excluded.'),
        tighter_certificate=dict(method='Exact rational grid plus global derivative envelope; |T_n prime|<=n^2 on[-1,1].',
            grid_intervals=count,grid_points=count+1,max_grid_abs=fraction(largest),argmax_index=argmax,
            derivative_upper=fraction(derivative),nearest_grid_distance=fraction(F(1,count)),
            polynomial_abs_upper=fraction(envelope),upper_with_12_5_percent_margin=fraction(padded),
            bound16_margin_after_guard=fraction(F(16)-padded),bound16_exact_real_polynomial_supported=True),
        scale_basis_distinctions=['Coefficient L1 alone exceeds16; tighter interval proof is required.',
            'All65 supplied binary64 coefficients are kept. Public constants2,-1 in balanced identities are exact real coefficients; output weight1 leaves them unchanged.',
            'Actual rounded domain map is covered because its image stays inside[-1,1]. This does not bound CKKS perturbations of that map or subsequent products.',
            'Monomial cheb2poly conversion has different rounded coefficients and conditioning and is not the selected64-degree evaluator.',
            'A12.5% public guard is a stated envelope margin, not a proven CKKS error estimate.'],
        levels=dict(domain_map=1,balanced_degree64_multiply_depth=6,balanced_identity_public_affine_depth=6,final_weighted_sum=1,total_GELU=14,
            output25_tail=[25,24,10,25,24,24,13]),profiles=profiles,
        obligations=['Every actual first-linear activation must satisfy the frozen[-21,14] GELU domain, including approximation and encryption perturbations; finite train/dev coverage does not certify arbitrary inputs.',
            'The encrypted balanced circuit and refresh must preserve the tighter envelope plus an explicit CKKS error allowance; this is not established by exact-real coefficients.',
            'Full65536-slot refresh with bound16 and the common ring/keys remains its own numerical obligation; the completed sparse-period128 result cannot be substituted.',
            'Attention-normalized residual values still require their own justified refresh bound (prior native estimate suggests128), and are retained/lowered without changing scale for the final residual add.',
            'Output linear must mask sourcewidth3072 and inactive rows; GELU padding isP(0), not zero.',
            'MLP variance+epsilon must remain in its frozen domain before inverse square root; removing a residual refresh does not certify that domain.',
            'The output25 whole-block level path depends on the prospective tracked-residual inverse; its all130-rule component is currently running and does not establish common-ring or composed-softmax precision.',
            'No new rotation keys arise from moving a full-slot refresh, but the same direct graph/bootstrap union, temporary controls and tables are still required. Raw stores are not peak-memory estimates.'],
        source_hashes_before=before,elapsed_s=time.monotonic()-start)
    with tarfile.open(HERE/'sources.tar.gz','w:gz') as t:
        for p in inputs:
            data=p.read_bytes();info=tarfile.TarInfo(str(p).lstrip('/'));info.size=len(data);t.addfile(info,io.BytesIO(data))
    after={str(p):sha(p.read_bytes()) for p in inputs};assert before==after
    with tarfile.open(HERE/'sources.tar.gz') as t:assert { '/'+m.name:sha(t.extractfile(m).read()) for m in t.getmembers()}==before
    report.update(source_hashes_after=after,source_files_unchanged=True,source_archive_verified=True,source_archive_sha256=sha((HERE/'sources.tar.gz').read_bytes()))
    (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','exact_binary64_coefficient_l1','tighter_certificate','elapsed_s')}))
if __name__=='__main__':main()
