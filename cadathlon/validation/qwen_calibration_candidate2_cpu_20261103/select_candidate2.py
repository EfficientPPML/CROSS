"""Select SiLU rules from fixed public bounds; no model/validation evaluation."""
from pathlib import Path
import copy,hashlib,io,json,math,os,tarfile,time
import numpy as np
import numeric
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
SOFT_FIELDS=('softmax','exp','initial_reciprocal','reciprocal_by_keys')


def fixed_parts(manifest):
    return ({k:v for k,v in manifest['nonlinear'].items() if not k.endswith('.silu')},
            {k:manifest[k] for k in SOFT_FIELDS})


def require_fixed(manifest,protocol):
    norms,soft=fixed_parts(manifest)
    if len(norms)!=113 or canonical(norms)!=protocol['unchanged_norms_sha256'] or canonical(soft)!=protocol['unchanged_softmax_sha256']:
        raise ValueError('Frozen norm/softmax rules changed')


def verify_inputs(root,protocol):
    actual={name:sha(root/name) for name in protocol['source_input_sha256']}
    if actual!=protocol['source_input_sha256']:raise ValueError('Frozen selection source/input mismatch')
    return actual


def grids(domain):
    lo,hi=domain
    return {'full':np.linspace(lo,hi,16385),
            'central':np.linspace(max(lo,-16),min(hi,16),16385),
            'endpoints_zero':np.asarray([lo,0.,hi])}


def rule(kind,domain,degree,strategy):
    coeff=numeric.coefficients(kind,domain,degree)
    result={'kind':kind,'domain':list(domain),'method':'polynomial','degree':degree,'strategy':strategy,
            'chebyshev_coefficients':coeff.tolist(),'evaluation_coefficients':numeric.basis_coefficients(coeff,strategy).tolist()}
    result['depth']=numeric.polynomial_depth(result)
    return result


def rational_checked(x,candidate):
    result=numeric.polynomial(x,candidate['base'],2.**-(candidate['doublings']+1))
    details=[];valid=bool(np.isfinite(result).all())
    for index in range(candidate['doublings']):
        denominator=1+result*result;finite=bool(np.isfinite(denominator).all())
        inside=finite and bool(np.all((denominator>=1)&(denominator<=2.1)))
        details.append({'step':index,'finite':finite,'inside_domain':inside,
                        'minimum':float(denominator.min()) if finite else None,'maximum':float(denominator.max()) if finite else None})
        valid &= inside
        if not valid:return None,details
        result=2*(result*numeric.reciprocal(denominator,candidate['reciprocal']))
        valid &= bool(np.isfinite(result).all())
        if not valid:return None,details
    return x*(.5*result+.5),details


def assess(candidate):
    metrics={};passed=True
    for name,x in grids(candidate['domain']).items():
        truth=x*np.exp(-np.logaddexp(0,-x))
        if candidate['method']=='polynomial':actual=numeric.polynomial(x,candidate);details=[]
        else:actual,details=rational_checked(x,candidate)
        finite=actual is not None and np.isfinite(actual).all()
        error=float(np.max(np.abs(actual-truth))) if finite else None
        ok=bool(finite and error<=1e-4);passed &= ok
        metrics[name]={'points':len(x),'finite':bool(finite),'max_abs_error':error,'passed':ok,'denominator_checks':details}
    return {'passed':bool(passed),'grids':metrics}


def select_site(domain,protocol,reciprocal):
    attempts=[]
    for degree in protocol['degrees_in_order']:
        candidate=rule('silu',domain,degree,'chebyshev');result=assess(candidate)
        attempts.append({'method':'polynomial','degree':degree,'strategy':'chebyshev','depth':candidate['depth'],**result})
        if result['passed']:return candidate,attempts
    doublings=0
    while 2**(doublings+1)<domain[1]:doublings+=1
    for degree in protocol['fallback_base_degrees']:
        base=rule('tanh',[-1.,1.],degree,'power');l1=float(np.abs(base['evaluation_coefficients']).sum())
        if not math.isfinite(l1) or l1>1e8:
            attempts.append({'method':'rational_tanh','base_degree':degree,'passed':False,'unsupported':'power conditioning'});continue
        candidate={'kind':'silu','method':'rational_tanh','domain':list(domain),'base':base,'doublings':doublings,'reciprocal':copy.deepcopy(reciprocal)}
        candidate['depth']=base['depth']+doublings*(reciprocal['depth']+4)+2
        result=assess(candidate);attempts.append({'method':'rational_tanh','base_degree':degree,'doublings':doublings,'depth':candidate['depth'],**result})
        if result['passed']:return candidate,attempts
    return None,attempts


def depths(manifest,S):
    depth=0;layers=[];expD=manifest['exp']['depth'];initialD=manifest['initial_reciprocal']['depth']
    for i in range(28):
        start=depth;n=manifest['nonlinear'];prefix=str(i)
        base=depth+n[prefix+'.input.inverse_sqrt']['depth']+4
        q=base+1+n[prefix+'.q.inverse_sqrt']['depth']+4+1
        k=base+1+n[prefix+'.k.inverse_sqrt']['depth']+4+1+1
        v=base+2;score=max(q,k)+2;attended=v
        for K in range(2,S+1):
            prob=score+1+expD+initialD+3+manifest['softmax'][prefix]['squarings']*(manifest['reciprocal_by_keys'][str(K)]['depth']+3)
            attended=max(attended,max(prob,v)+2)
        attention=max(depth,attended+1);post=attention+n[prefix+'.post.inverse_sqrt']['depth']+4
        depth=max(attention,max(post+1+n[prefix+'.silu']['depth'],post+1)+2)
        layers.append({'layer':i,'depth_increment':depth-start,'depth_after':depth})
    return {'sequence_length':S,'layers':layers,'final_depth':depth+manifest['nonlinear']['final.inverse_sqrt']['depth']+5}


def main():
    started=time.perf_counter();out=P/'candidate-v2';out.mkdir(exist_ok=False)
    before={}
    report={'passed':False,'complete':False,'errors':[],'unsupported':[],'source_hashes_before':before,'model_evaluated':False,'validation_outputs_read':False,'scope':__doc__}
    assessments={};primary=None
    try:
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        protocol=json.loads((P/'protocol.json').read_text());before=verify_inputs(P,protocol);before['protocol.json']=sha(P/'protocol.json');report['source_hashes_before']=before
        with tarfile.open(out/'sources.tar.gz','w:gz') as t:
            for n in before:b=(P/n).read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
        with tarfile.open(out/'sources.tar.gz') as t:
            ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(before) and all(m.isfile() for m in ms)
            assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==before
        report['source_archive_sha256']=sha(out/'sources.tar.gz')
        base=json.loads((P/'inputs/base_manifest.json').read_text());require_fixed(base,protocol)
        certificates=json.loads((P/'norm_multipliers.json').read_text());assert len(certificates)==28
        manifest=copy.deepcopy(base)
        for layer in range(28):
            label=f'{layer}.silu';domain=certificates[str(layer)]['public_silu_domain']
            chosen,attempts=select_site(domain,protocol,base['nonlinear']['27.silu']['reciprocal'])
            assessments[label]={'public_domain':domain,'bound_certificate':certificates[str(layer)],'attempts':attempts,'supported':chosen is not None}
            if chosen is None:report['unsupported'].append(label);continue
            manifest['nonlinear'][label]=chosen
            print(json.dumps({'site':label,'domain':domain,'method':chosen['method'],'degree':chosen.get('degree'),'base_degree':chosen.get('base',{}).get('degree'),'doublings':chosen.get('doublings'),'depth':chosen['depth']}),flush=True)
        require_fixed(manifest,protocol)
        manifest.update(schema='qwen-calibration-candidate-v2',candidate_index=2,source_input_sha256=before,
            selection='Public-weight bounds and certified norm multipliers only for28 SiLU rules; fixed analytic grids/fallbacks. All113 norm and allsoftmax rules unchanged; no model/validation-output fitting.',
            parent_manifest_sha256=sha(P/'inputs/base_manifest.json'),unchanged_norms_sha256=protocol['unchanged_norms_sha256'],unchanged_softmax_sha256=protocol['unchanged_softmax_sha256'])
        manifest['depth_by_sequence']={str(S):depths(manifest,S) for S in (8,16,19,32)} if not report['unsupported'] else {}
        # An unsupported selection is explicitly partial; it is never an
        # approved complete candidate merely because an old entry survives.
        manifest['all_silu_sites_supported']=not report['unsupported']
        for name,value in [('manifest.json',manifest),('assessments.json',assessments)]:
            (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        report.update(passed=not report['unsupported'],complete=True,all28_supported=not report['unsupported'],
            unchanged_norms_verified=True,unchanged_softmax_verified=True,manifest_sha256=sha(out/'manifest.json'),assessments_sha256=sha(out/'assessments.json'),
            depth_by_sequence={S:v['final_depth'] for S,v in manifest['depth_by_sequence'].items()})
    except BaseException as exc:
        primary=exc;report['passed']=False;report['errors'].append({'type':type(exc).__name__,'message':str(exc)})
    finally:
        try:
            after={n:sha(P/n) for n in before};report.update(source_hashes_after=after,source_files_unchanged=bool(before) and before==after)
            report['passed']=report['passed'] and before==after
        except BaseException as exc:
            report['passed']=False
            if primary is None:primary=exc
            else:primary.add_note('Final provenance failed: '+repr(exc))
        report['elapsed_s']=time.perf_counter()-started
        try:(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        except BaseException as exc:
            if primary is None:raise
            primary.add_note('Final publication failed: '+repr(exc))
    if primary is not None:raise primary
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
