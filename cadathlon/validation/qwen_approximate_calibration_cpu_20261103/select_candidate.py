"""Candidate1: calibration input ranges plus frozen analytic targets only."""
from pathlib import Path
from collections import Counter
import hashlib,io,json,math,sys,tarfile,time
import numpy as np
import numeric
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rule(kind,domain,degree,strategy):
    c=numeric.coefficients(kind,domain,degree)
    r={'kind':kind,'domain':list(domain),'method':'polynomial','degree':degree,'strategy':strategy,'chebyshev_coefficients':c.tolist(),'evaluation_coefficients':numeric.basis_coefficients(c,strategy).tolist()}
    r['depth']=numeric.polynomial_depth(r);return r

def depths(manifest,S):
    depth=0;layers=[];expD=manifest['exp']['depth'];initialD=manifest['initial_reciprocal']['depth']
    for i in range(28):
        start=depth;n=manifest['nonlinear'];prefix=str(i)
        base=depth+n[prefix+'.input.inverse_sqrt']['depth']+4
        q=base+1+n[prefix+'.q.inverse_sqrt']['depth']+4+1
        k=base+1+n[prefix+'.k.inverse_sqrt']['depth']+4+1+1
        v=base+1+1;score=max(q,k)+2;attended=v
        for K in range(2,S+1):
            prob=score+1+expD+initialD+3
            prob+=manifest['softmax'][prefix]['squarings']*(manifest['reciprocal_by_keys'][str(K)]['depth']+3)
            attended=max(attended,max(prob,v)+2)
        attention=max(depth,attended+1)
        post=attention+n[prefix+'.post.inverse_sqrt']['depth']+4
        gate=post+1;depth=max(attention,max(gate+n[prefix+'.silu']['depth'],post+1)+2)
        layers.append({'layer':i,'depth_increment':depth-start,'depth_after':depth})
    return {'sequence_length':S,'layers':layers,'final_depth':depth+manifest['nonlinear']['final.inverse_sqrt']['depth']+5}

def main():
    out=P/'candidate-v1';out.mkdir(exist_ok=False);began=time.perf_counter()
    sources=[P/n for n in ('select_candidate.py','numeric.py','protocol.json','selection_inputs.json','input_origin.json','reciprocal_schedule.json','model_pins.json','corpus.json','tokenized_calibration_inputs.json')]
    before={f.name:sha(f) for f in sources};report={'passed':False,'unsupported':[],'errors':[],'source_hashes_before':before,'model_evaluated':False,'validation_data_used':False}
    with tarfile.open(out/'sources.tar.gz','w:gz') as a:
        for f in sources:
            b=f.read_bytes();item=tarfile.TarInfo(f.name);item.size=len(b);a.addfile(item,io.BytesIO(b))
    report['source_archive_sha256']=sha(out/'sources.tar.gz')
    try:
        protocol=json.loads((P/'protocol.json').read_text());inputs=json.loads((P/'selection_inputs.json').read_text());a=protocol['analytic_selection'];gridN=a['grid_size'];assessments={};nonlinear={}
        for label,row in inputs.items():
            if label.endswith('softmax.centered'):continue
            low,high=row['minimum'],row['maximum'];kind='silu' if label.endswith('silu') else 'inverse_sqrt'
            if not math.isfinite(low+high) or low>=high or (kind=='inverse_sqrt' and low<=0):raise ValueError('Unsupported calibration domain: '+label)
            domain=[.8*low,1.2*high] if kind=='inverse_sqrt' else [math.floor(low-.1*(high-low)),math.ceil(high+.1*(high-low))]
            grid=np.linspace(*domain,gridN);truth=1/np.sqrt(grid) if kind=='inverse_sqrt' else grid*np.exp(-np.logaddexp(0,-grid));sweep=[];chosen=None
            for degree in a['degrees_in_order']:
                c=numeric.coefficients(kind,domain,degree);l1=float(np.abs(np.polynomial.chebyshev.cheb2poly(c)).sum())
                strategy='power' if kind=='inverse_sqrt' and degree<=a['power_max_degree'] and l1<=a['power_coefficient_L1_limit'] else 'chebyshev'
                candidate=rule(kind,domain,degree,strategy);value=numeric.polynomial(grid,candidate)
                error=float(np.max(np.abs(value/truth-1))) if kind=='inverse_sqrt' else float(np.max(np.abs(value-truth)))
                target=a['inverse_sqrt_relative_target'] if kind=='inverse_sqrt' else a['silu_absolute_target']
                clenshaw=np.polynomial.chebyshev.chebval((2*grid-domain[0]-domain[1])/(domain[1]-domain[0]),c)
                sweep.append({'degree':degree,'strategy':strategy,'power_coefficient_l1':l1,'explicit_circuit_grid_error':error,'clenshaw_vs_circuit_max_abs':float(np.max(np.abs(clenshaw-value))),'target':target,'passed':bool(np.isfinite(value).all() and error<=target)})
                if sweep[-1]['passed']:chosen=candidate;break
            alternatives=[]
            if chosen is None and kind=='inverse_sqrt':
                chosen=numeric.rsqrt_contract(domain,a['inverse_sqrt_relative_target']);chosen.update(kind=kind,method='newton')
                error=float(np.max(np.abs(numeric.rsqrt(grid,chosen)/truth-1)));chosen['explicit_grid_relative_error']=error
                if error>a['inverse_sqrt_relative_target']:report['unsupported'].append({'site':label,'reason':'Newton actual grid target failed','error':error})
            elif chosen is None:
                doubles=max(0,math.ceil(math.log2(max(map(abs,domain))/2)));rec=numeric.reciprocal_contract([1.,2.1])
                for degree in a['silu_fallback']['base_degrees_in_order']:
                    base=rule('tanh',[-1.,1.],degree,'power');candidate={'kind':'silu','method':'rational_tanh','domain':domain,'base':base,'doublings':doubles,'reciprocal':rec}
                    candidate['depth']=base['depth']+doubles*(rec['depth']+4)+2
                    value=numeric.rational_silu(grid,candidate);error=float(np.max(np.abs(value-truth)))
                    alternatives.append({'base_degree':degree,'doublings':doubles,'depth':candidate['depth'],'max_abs_grid_error':error,'passed':error<=a['silu_absolute_target']})
                    if alternatives[-1]['passed']:chosen=candidate;break
                if chosen is None:report['unsupported'].append({'site':label,'reason':'No direct/rational SiLU candidate meets target'});continue
            nonlinear[label]=chosen;assessments[label]={'observed_range':[low,high],'domain':domain,'polynomial_sweep':sweep,'fallbacks':alternatives,'selected_method':chosen['method'],'selected_depth':chosen['depth']}
            print(json.dumps({'site':label,'method':chosen['method'],'strategy':chosen.get('strategy'),'degree':chosen.get('degree'),'iterations':chosen.get('iterations'),'depth':chosen['depth']}),flush=True)
        saved=json.loads((P/'reciprocal_schedule.json').read_text());byK={}
        for rr in saved['round_contracts']:
            c=numeric.reciprocal_contract(rr['domain']);assert c['iterations']==rr['iterations'] and c['rounded_first_weight']==rr['rounded_first_weight'] and c['rounded_first_bias']==rr['rounded_first_bias'];c['inactive_midpoint']=rr['inactive_midpoint'];byK[str(rr['valid_keys'])]=c
        softmax={}
        for i in range(28):
            rr=inputs[f'{i}.softmax.centered'];lo,hi=rr['minimum'],rr['maximum'];domain=[math.floor(lo-.1*(hi-lo)),math.ceil(hi+.1*(hi-lo))]
            softmax[str(i)]={'centered_domain':domain,'squarings':max(0,math.ceil(math.log2(max(1,*map(abs,domain))/.5)))}
        manifest={'schema':'qwen-calibration-candidate-v1','candidate_index':1,'source_input_sha256':before,'nonlinear':nonlinear,'softmax':softmax,'exp':rule('exp',[-.5,.5],8,'power'),'initial_reciprocal':numeric.reciprocal_contract([.5,2.]),'reciprocal_by_keys':byK,
            'selection':'Frozen calibration-input union/slack and analytic functions only; no native/approximate logits, labels, validation extrema or model evaluation used to choose coefficients.',
            'depth_scope':'Explicit abstract rescale-consuming arithmetic schedule. Learned linear1, affine1, CTCT1, masked reduction1, layout move1, add/view0; no ring/refresh feasibility or accelerator claim.',
            'error_scope':'Polynomial errors are sampled analytic grid errors, not uniform certificates. Newton certificates include rounded first affine then exact-real updates only; later binary64/CKKS/refresh error excluded.'}
        manifest['depth_by_sequence']={str(S):depths(manifest,S) for S in (8,16,19,32)} if not report['unsupported'] else {}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n');(out/'assessments.json').write_text(json.dumps(assessments,indent=2,allow_nan=False)+'\n')
        report.update(passed=not report['unsupported'],manifest_sha256=sha(out/'manifest.json'),assessments_sha256=sha(out/'assessments.json'),method_counts=dict(Counter((r['method']+':'+r.get('strategy','')) for r in nonlinear.values())),
            maximum_nonlinear_depth=max(r['depth'] for r in nonlinear.values()),depth_by_sequence={S:v['final_depth'] for S,v in manifest['depth_by_sequence'].items()},site_count=len(nonlinear),software={'numpy':np.__version__,'python':sys.version})
    except BaseException as exc:
        report['errors'].append({'type':type(exc).__name__,'message':str(exc)});raise
    finally:
        after={f.name:sha(f) for f in sources};report.update(source_hashes_after=after,source_files_unchanged=before==after,elapsed_s=time.perf_counter()-began)
        if before!=after:report['passed']=False
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
