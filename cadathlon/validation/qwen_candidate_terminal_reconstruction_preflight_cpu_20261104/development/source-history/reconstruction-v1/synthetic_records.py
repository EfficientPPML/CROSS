"""Synthetic journal fixtures; no saved input/model/outcome files are read.

The copied producer's pure dispatcher and Aggregate AST construct expected
reports independently of verify_records. No arithmetic/model code is executed.
"""
import ast
import copy
import hashlib
import importlib.util
import pathlib
import sys
from dataclasses import asdict
from contract import entries,operator_counts
from verify_statistics import canonical,context_hash,domain_plan,INPUT_SCOPE,DOMAIN_SCOPE

HERE=pathlib.Path(__file__).parent
def load_pure_dispatcher():
    prior=sys.modules.get('workload')
    spec=importlib.util.spec_from_file_location('workload',HERE/'sources/workload.py');w=importlib.util.module_from_spec(spec);sys.modules['workload']=w;spec.loader.exec_module(w)
    spec=importlib.util.spec_from_file_location('_synthetic_producer_dispatch',HERE/'sources/candidate_dispatch.py');d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
    if prior is not None:sys.modules['workload']=prior
    else:sys.modules.pop('workload',None)
    return d

def producer_aggregate():
    tree=ast.parse((HERE/'sources/model_runtime.py').read_text())
    chosen=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in('merge_statistics','Aggregate')]
    assert len(chosen)==2
    g={'copy':copy};exec(compile(ast.Module(body=chosen,type_ignores=[]),'<exact-public-Aggregate-AST>','exec'),g)
    return g['Aggregate']

def manifest():
    nonlinear={f'{i}.{p}.inverse_sqrt':dict(method='newton',iterations=2,domain=[.1,20.])for i in range(28)for p in('input','post','q','k')}
    nonlinear['final.inverse_sqrt']=dict(method='polynomial',domain=[.1,20.])
    nonlinear.update({f'{i}.silu':dict(method='polynomial',domain=[-20.,20.])for i in range(28)})
    nonlinear['0.silu']=dict(method='rational_tanh',domain=[-20.,20.],base=dict(domain=[-.5,.5]),reciprocal=dict(domain=[1.,2.1],iterations=5),doublings=2)
    return dict(nonlinear=nonlinear,softmax={str(i):dict(squarings=1,centered_domain=[-20.,20.])for i in range(28)},exp=dict(domain=[-.5,.5]),
        initial_reciprocal=dict(domain=[.2,2.],iterations=6),reciprocal_by_keys={str(k):dict(domain=[.5/k,1.5],iterations=3)for k in range(2,33)})

def calibration():
    rows=[]
    for p in range(96):
        for s in range(1,33):rows.append(dict(prompt_ordinal=p,id='synthetic-'+str(p),sequence_length=s,token_ids=[p+1]*s,original_prefill_member=s in(8,16,32)))
    return dict(prefills=rows,decode_seeds=[dict(seed_ordinal=i,prompt_ordinal=i*6,id='synthetic-'+str(i*6),token_ids=[i*6+1]*16)for i in range(16)])

def metric(kind,n,passed=True):
    m=dict(passed=passed,finite=True,shape_matches=True,real_numeric=True,count=n,max_abs=0. if passed else .2,nrmse=0. if passed else .2,limit=dict(operator=.05,block=.05,logit=.1,semantic=1e-5)[kind],floor=.01)
    if kind=='semantic':m.update(top1_match=True,max_abs_limit=.001)
    return m

def stat(n,value):return dict(count=n,minimum=float(value),maximum=float(value),nonfinite=0,nonpositive=0 if value>0 else n)

def inputs(s,origin):
    ranges={f'{i}.{p}.inverse_sqrt':stat(s*w,1.)for i in range(28)for p,w in [('input',1),('post',1),('q',16),('k',8)]}
    ranges['final.inverse_sqrt']=stat(1,1.);ranges.update({f'{i}.silu':stat(3072*s,0.)for i in range(28)})
    scores={}
    for i in range(28):
        scores[str(i)]={str(k):dict(query=k-1,key_count=k,shape=[1,16,k],k1_bypass=k==1,reciprocal_observations=0,raw_valid=stat(16*k,0.),centered=stat(16*k,0.))for k in range(1,s+1)}
        for phase in('raw_valid','centered'):ranges[f'{i}.softmax.{phase}']=stat(16*s*(s+1)//2,0.)
    return dict(sequence_length=s,ranges=ranges,score_rows=scores,ignored_exact_output_callbacks=0 if origin=='native'else 169,complete=True,invalid=False,norm_input_sites=113,silu_gate_sites=28,scope=INPUT_SCOPE)

def domain_observations(plan,stopped=False):
    chosen=plan[:1]if stopped else plan;domains={}
    for label,interval,n in chosen:
        v=(interval[0]+interval[1])/2
        domains[label]=dict(domain=interval.copy(),count=n,minimum=v,maximum=v,below=0,above=0,nonfinite=0)
    if stopped:
        label=chosen[-1][0];r=domains[label];r.update(minimum=r['domain'][1]+1.,maximum=r['domain'][1]+1.,above=r['count'])
    return dict(expected_observations=len(plan),observations=len(chosen),domains=domains,
        normalization_variance={k:copy.deepcopy(v)for k,v in domains.items()if k.endswith('.inverse_sqrt')},scope=DOMAIN_SCOPE)

class Fixture:
    def __init__(self,*,numeric_failed=(),stopped=(),different_choice=()):
        self.manifest=manifest();self.raw=canonical(self.manifest);self.pin=hashlib.sha256(self.raw).hexdigest();self.calibration=calibration()
        self.numeric_failed=set(numeric_failed);self.stopped=set(stopped);self.different_choice=set(different_choice)
        self.plans={s:domain_plan(self.manifest,s)for s in range(1,33)};self.templates={}
        for s in range(1,33):
            self.templates[(s,'native')]=inputs(s,'native');self.templates[(s,'exact')]=inputs(s,'exact');self.templates[(s,'candidate')]=domain_observations(self.plans[s]);self.templates[(s,'stopped')]=domain_observations(self.plans[s],True)
        d=load_pure_dispatcher()
        def candidate(prefix,e):
            stop=e.index in self.stopped
            return d.Candidate(None if stop else 8 if e.index in self.different_choice else 7,not stop,not stop and e.index not in self.numeric_failed,not stop,0 if stop else 421,0 if stop else 28,0 if stop else 151936)
        result=d.run_candidate(self.calibration,manifest_sha256=self.pin,native=lambda p,e:d.Native(7),exact=lambda p,e,n:d.Exact(7,True),candidate=candidate)
        self.records=[asdict(r)for r in result.records];self.records=json_roundtrip(self.records)
        A=producer_aggregate();na=A();ex=A();candidate_aggregate={};measured=complete=0
        for r in self.records:
            if r['status']=='unmeasured_suffix':continue
            measured+=1;s=r['entry']['length'];na.add(self.templates[(s,'native')]);ex.add(self.templates[(s,'exact')]);complete+=r['status']=='complete'
            obs=self.templates[(s,'stopped'if r['status']=='candidate_invalid'else'candidate')]
            for label,value in obs['domains'].items():
                if label not in candidate_aggregate:candidate_aggregate[label]=copy.deepcopy(value)
                else:
                    old=candidate_aggregate[label]
                    for name in('count','below','above','nonfinite'):old[name]+=value[name]
                    old['minimum']=min(old['minimum'],value['minimum']);old['maximum']=max(old['maximum'],value['maximum'])
        summary=result.public_report();summary.update(phase_counts=dict(native=measured,exact=measured,candidate=measured),semantic_passes=measured,candidate_complete=complete,
            native_invariance_by_length={str(s):True for s in range(1,33)},exact_invariance_by_length={str(s):True for s in range(1,33)},all_length_invariance=True,
            input_ranges=dict(native=na.report(),exact=ex.report(),candidate_propagated=candidate_aggregate),limits=dict(operator_block_nrmse=.05,full_logit_nrmse=.10,semantic_nrmse=1e-5,semantic_max_abs=.001,nrmse_floor=.01),
            coefficient_selection_performed=False,encrypted_execution=False,validation_inputs_read=False,complete_measured_context_coverage=complete==3200,revision_comparison_coverage=True,complete=True)
        self.ledger=dict(manifest_sha256=self.pin,records=self.records,summary=summary);self.worker=copy.deepcopy(summary);self.worker.update(native_forward_started=measured+32,native_forward_completed=measured+32,native_forward_accounting=dict(expected=measured+32,started=measured+32,completed=measured+32,passed=True))
        self.inventory={'ledger.json'}|{f'{r["entry"]["index"]:04d}-{p}.json'for r in self.records if r['status']!='unmeasured_suffix'for p in('native','exact','candidate')}
        self.reads=[]
    def read(self,name):
        self.reads.append(name);i,phase=name[:-5].split('-');i=int(i);r=self.records[i];s=r['entry']['length'];prefix=r['prefix']
        h=dict(entry=r['entry'],manifest_sha256=self.pin,token_ids=prefix,context_sha256=context_hash(prefix))
        observed=self.templates[(s,'stopped'if phase=='candidate'and r['status']=='candidate_invalid'else phase)]
        h.update(observations=observed,observations_sha256=hashlib.sha256(canonical(observed)).hexdigest())
        if phase=='native':h.update(status='complete',native_top1=7,operator_count=421,block_count=28,all_logits=151936)
        elif phase=='exact':h.update(status='complete',exact_top1=7,semantic=metric('semantic',151936))
        else:
            stop=r['status']=='candidate_invalid';failure=None
            if stop:
                label=self.plans[s][0][0];failure=dict(type='InvalidCircuit',label=label,record=observed['domains'][label])
            op={}if stop else{k:metric('operator',n,not(i in self.numeric_failed and k=='0.q_linear'))for k,n in operator_counts(s).items()}
            h.update(status=r['status'],candidate_top1=r['candidate']['token'],numeric_pass=r['candidate']['numeric_pass'],domain_pass=not stop,failure=failure,
                operator_metrics=op,block_metrics=[]if stop else[metric('block',1024*s)for _ in range(28)],full_logit_metric=None if stop else metric('logit',151936),all_logits=0 if stop else 151936)
        return h

def json_roundtrip(x):
    import json
    return json.loads(canonical(x))
