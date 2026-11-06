"""Derive fresh public certificates and metadata from the exact selected rules.

No selection, extrema fitting, tensor loading, trajectory or backend execution.
The copied verifier code checks exact stored polynomial/Newton certificates.
"""
from pathlib import Path
from fractions import Fraction as F
import ast,hashlib,json,math,sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'checkout/cadathlon/common'))
import qwen_candidate_bridge as bridge
import certificate,newton_certificate
def canonical(value):return bridge.canonical(value)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def serialize(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def norm_planner():
    raw=(P/'reference_sources/public_norm_plan.py').read_bytes()
    if sha(raw)!='225d97400dd26df7fb1c877bae5830ddee71f79896aa9121445ea59163e0e1dc':raise ValueError('original public level/bound planner changed')
    names={'ceil_sqrt','power2_guard','upper_float','original_update_level','norm_plan'}
    tree=ast.parse(raw);body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    if len(body)!=len(names):raise ValueError('public planner definition inventory differs')
    ns={'F':F,'math':math};exec(compile(ast.Module(body=body,type_ignores=[]),'<frozen public norm planner>','exec'),ns)
    return ns['norm_plan']
def derive():
    if not __debug__:raise RuntimeError('certificate assertions required')
    oldroot=P/'before87';pins=json.loads((P/'predecessor87.json').read_bytes())['sources']
    for name,pin in pins.items():
        if sha((oldroot/name).read_bytes())!=pin:raise ValueError('preserved predecessor changed: '+name)
    for name,pin in (('certificate.py','1283be55dad48f5569d362116c1ada3b12b6b5e8a2fef2986880e23f863db8f1'),('newton_certificate.py','5632832d1f7e728c6b27987dbf2258976f430ee0aa78c54accc2a7bbaf88c4ea')):
        if sha((P/name).read_bytes())!=pin:raise ValueError('frozen certificate verifier differs')
    oldraw=(oldroot/'inputs/candidate_manifest.json').read_bytes();raw=(P/'inputs/candidate_manifest.json').read_bytes()
    manifest=bridge.candidate(raw);old=json.loads(oldraw)
    relation=bridge.derive_bridge(oldraw,raw,(P/'inputs/config.json').read_bytes(),(P/'inputs/certificates.json').read_bytes(),
        (P/'qkv-export-v1/parameter_manifest.json').read_bytes(),(P/'inputs/decoder_parameters.json').read_bytes())
    metadata=json.loads((oldroot/'inputs/site_metadata.json').read_bytes())
    previous=json.loads((oldroot/'inputs/join_plan.json').read_bytes());oldrows={r['site']:r for r in previous['normalizations']}
    names=sorted(n for n in manifest['nonlinear'] if n.endswith('inverse_sqrt'))
    if len(names)!=113 or set(metadata)!=set(names):raise ValueError('exact113 normalization inventory required')
    planner=norm_planner();certificates={};plans=[];rows=[]
    for name in names:
        rule=manifest['nonlinear'][name]
        if rule['method']=='polynomial':
            proof=certificate.certify(rule)
            if proof['certified']is not True:raise ValueError('selected polynomial certificate fails: '+name)
        else:
            proof=newton_certificate.check_certificate(rule)
            if proof['verified']is not True:raise ValueError('selected Newton certificate fails: '+name)
        certificates[name]=dict(rule_sha256=sha(canonical(rule)),method=rule['method'],certificate=proof)
        plan=planner(name,rule);plans.append(plan)
        # Only immutable gamma descriptor fields come from the earlier public
        # checkpoint export. Every event and level comes from the new rule.
        meta=metadata[name]
        if meta['group_width']!=plan['group_width']:raise ValueError('gamma width differs')
        meta['output_level']=plan['output_level'];meta['refresh_events']=plan['refresh_events']
        row=dict(oldrows[name]);row.update(method=rule['method'],degree=rule.get('degree',0),iterations=rule.get('iterations',0),
            output_level=plan['output_level'],internal_inverse_refreshes=len(plan['refresh_events']),raw_input_bound=plan['conditional_input_refresh_bound'])
        rows.append(row)
    # Same learned gamma vectors, widths and <=1e-6 relative target establish
    # the identical normalized-output/RoPE bounds. They remain conditional on
    # same-input variance and refresh perturbation premises, never CKKS proof.
    continuation=dict(schema='qwen-expanded-joined-continuation-v1',passed=True,
        scope='Conditional exact-real public continuation; no calibration/trajectory/HE execution.',candidate_manifest_sha256=bridge.MANIFEST,
        historical_continuation_sha256=sha((oldroot/'inputs/join_plan.json').read_bytes()),
        protected_bridge=relation,normalizations=rows,
        rope_table=previous['rope_table'],rope_bounds=previous['rope_bounds'],score_domain_outer_bounds=previous['score_domain_outer_bounds'],
        all_centered_domains_proved=previous['all_centered_domains_proved'],
        preserved_bound_scope='Gamma/width/relative-target and actual stored RoPE table unchanged. No new normalization coefficients or raw-input bounds inherit an old certificate.',
        limitations=previous['limitations'],scientific_calibration_verified=False,execution_authorized=False)
    report=dict(schema='qwen-expanded-normalization-public-v1',candidate_manifest_sha256=bridge.MANIFEST,
        certificate_scope='Recomputed exact stored-map/coefficient or directed Newton certificates only; no binary64/CKKS/refresh error proof.',
        certificates=certificates,normalizations=plans,site_count=len(names),
        changed_rule_count=sum(manifest['nonlinear'][n]!=old['nonlinear'][n] for n in names),
        inverse_refresh_count=sum(len(r['refresh_events']) for r in plans),
        scientific_calibration_verified=False,execution_authorized=False)
    generated={name:serialize(value) for name,value in (('inputs/candidate_bridge.json',relation),('inputs/normalization_certificates.json',report),('inputs/site_metadata.json',metadata),('inputs/join_plan.json',continuation))}
    bindings={key:sha(generated[name]) for key,name in (('bridge_sha256','inputs/candidate_bridge.json'),('normalization_certificates_sha256','inputs/normalization_certificates.json'),('site_metadata_sha256','inputs/site_metadata.json'),('join_plan_sha256','inputs/join_plan.json'))}
    generated['derived_bindings.json']=serialize(bindings);return generated
def verify_generated():
    generated=derive()
    for name,value in generated.items():
        if (P/name).read_bytes()!=value:raise ValueError('fresh public derivation differs from frozen input: '+name)
    return {name:sha(value) for name,value in generated.items()}
if __name__=='__main__':print(json.dumps(verify_generated(),sort_keys=True))
