from pathlib import Path
import sys,json,math,hashlib
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P/'checkout/cadathlon/common'))
from qwen_silu_contract import QwenSiLUContract,silu_scale_plan,DELTA

def derive():
    manifest=(P/'inputs/candidate_manifest.json').read_bytes();certificate=(P/'inputs/certificates.json').read_bytes()
    profile=next(p for p in json.loads((P/'inputs/profile_audit.json').read_bytes())['profiles'] if p['name']=='depth49_p60_d8')
    contracts=[QwenSiLUContract(manifest,certificate,l) for l in range(28)];plans=[]
    for c in contracts:
        scales=tuple(math.nextafter(DELTA,math.inf if j%2 else -math.inf) for j in range(len(c.events)))
        plan=silu_scale_plan(profile['q'],c,input_scale=math.nextafter(DELTA,math.inf),refresh_scales=scales)
        plan.update(domain=list(c.domain),strategy=c.strategy,declared_degree=c.degree,actual_degree=len(c.coefficients)-1,events=list(c.events))
        plans.append(plan)
    return dict(passed=True,candidate_sha256=hashlib.sha256(manifest).hexdigest(),sites=plans,
        polynomial_sites=sum(c.method=='polynomial' for c in contracts),rational_sites=sum(c.doublings>0 for c in contracts),
        total_refreshes=sum(c.doublings for c in contracts),minimum_output_level=min(p['output_level'] for p in plans),
        scope='Public fixed circuits/actual-scale replay, no keys/backend/model/HE; same-input perturbation obligations unproved.')
if __name__=='__main__':
    report=derive();(P/'public_plan.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print({k:v for k,v in report.items() if k!='sites'})
