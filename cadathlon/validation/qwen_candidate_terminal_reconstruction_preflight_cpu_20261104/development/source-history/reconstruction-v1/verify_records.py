"""Independent3200 recorded-journal replay; no filesystem/model imports.

read_phase supplies already-authenticated parsed private journals. This pure
function confers no permission to open them and never returns token choices.
Only comparison-valid complete scientific attempts can produce an audit result.
Comparison failures fail closed, separately from a legitimate candidate stop.
"""
import copy
import hashlib
import json
from contract import entries,operator_counts,metric,expected_files
from verify_statistics import (canonical,same,fields,integer,digest,token,tokens,
    context_hash,bound_hash,verify_inputs,InputAggregate,domain_plan,verify_domains,merge)

class IneligibleComparison(ValueError):pass

def parse(raw):
    if type(raw)is not bytes:raise ValueError('Immutable JSON bytes required')
    def pairs(rows):
        out={}
        for k,v in rows:
            if k in out:raise ValueError('Duplicate JSON field')
            out[k]=v
        return out
    def bad(x):raise ValueError('Nonfinite JSON rejected')
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
    canonical(value) # Also rejects numeric overflow such as1e999.
    return value

def calibration_prefixes(data):
    fields(data,('prefills','decode_seeds'));rows=data['prefills'];seeds=data['decode_seeds']
    if type(rows)is not list or len(rows)!=3072 or type(seeds)is not list or len(seeds)!=16:raise ValueError('Fixed calibration workload required')
    prefixes=[];ids=[]
    for p in range(96):
        name=None;prior=()
        for s in range(1,33):
            r=rows[32*p+s-1];fields(r,('prompt_ordinal','id','sequence_length','token_ids','original_prefill_member'))
            same(r['prompt_ordinal'],p);same(r['sequence_length'],s);same(r['original_prefill_member'],s in(8,16,32))
            if type(r['id'])is not str or not r['id']:raise ValueError('Private prompt identity required')
            if name is None:name=r['id']
            if name!=r['id']:raise ValueError('Prompt family identity changed')
            current=tokens(r['token_ids'],s)
            if current[:-1]!=prior:raise ValueError('Calibration family is not append only')
            prior=current;prefixes.append(current)
        ids.append(name)
    if len(set(ids))!=96:raise ValueError('Unique original prompt IDs required')
    selected=[];used=set()
    for i,r in enumerate(seeds):
        fields(r,('seed_ordinal','prompt_ordinal','id','token_ids'));same(r['seed_ordinal'],i)
        p=integer(r['prompt_ordinal'],0,95)
        if p in used or r['id']!=ids[p]:raise ValueError('Original seed identity differs')
        current=tokens(r['token_ids'],16)
        if current!=prefixes[32*p+15]:raise ValueError('Original S16 seed prefix differs')
        used.add(p);selected.append(current)
    return tuple(prefixes),tuple(selected)

HEADER=('entry','manifest_sha256','token_ids','context_sha256')
def journal(row,entry,prefix,pin,extra):
    fields(row,(*HEADER,*extra));same(row['entry'],entry);same(row['manifest_sha256'],pin)
    if tokens(row['token_ids'],entry['length'])!=prefix:raise ValueError('Journal actual prefix differs')
    same(row['context_sha256'],context_hash(prefix))

def native_journal(row,e,prefix,pin,aggregate):
    journal(row,e,prefix,pin,('status','native_top1','observations','observations_sha256','operator_count','block_count','all_logits'))
    same(row['status'],'complete');choice=token(row['native_top1'])
    for k,v in dict(operator_count=421,block_count=28,all_logits=151936).items():same(row[k],v)
    bound_hash(row['observations'],row['observations_sha256']);verify_inputs(row['observations'],e['length'],'native');aggregate.add(row['observations'])
    return choice

def exact_journal(row,e,prefix,pin,native,aggregate):
    journal(row,e,prefix,pin,('status','exact_top1','semantic','observations','observations_sha256'))
    same(row['status'],'complete');choice=token(row['exact_top1'])
    if not metric(row['semantic'],'semantic',151936,top1_match=choice==native):raise IneligibleComparison('Native/exact semantic comparison failed')
    bound_hash(row['observations'],row['observations_sha256']);verify_inputs(row['observations'],e['length'],'exact');aggregate.add(row['observations'])
    return choice

def candidate_journal(row,e,prefix,pin,plan,aggregate):
    journal(row,e,prefix,pin,('status','candidate_top1','numeric_pass','domain_pass','failure','operator_metrics','block_metrics','full_logit_metric','observations','observations_sha256','all_logits'))
    if row['status']not in('complete','candidate_invalid'):raise ValueError('Candidate status differs')
    complete=row['status']=='complete';same(row['domain_pass'],complete)
    bound_hash(row['observations'],row['observations_sha256'])
    domains=verify_domains(row['observations'],plan,complete,row['failure'])
    for label,value in domains.items():
        if label in aggregate:same(aggregate[label]['domain'],value['domain'])
        aggregate[label]=merge(aggregate.get(label),value,('count','below','above','nonfinite'))
    counts=operator_counts(e['length']);op=row['operator_metrics'];blocks=row['block_metrics']
    if type(op)is not dict or type(blocks)is not list:raise ValueError('Metric collections differ')
    integer(len(op),0,421);integer(len(blocks),0,28)
    expected=tuple(counts)[:len(op)]
    if set(op)!=set(expected):raise ValueError('Candidate operator prefix set differs')
    # Persisted mappings are lexically sorted JSON. Source guards attest temporal
    # order; here derive and check the expected prefix independently.
    op_pass=[metric(op[label],'operator',counts[label])for label in expected]
    block_pass=[metric(m,'block',1024*e['length'])for m in blocks]
    if complete:
        if len(op)!=421 or len(blocks)!=28:raise ValueError('Complete metric inventory missing')
        choice=token(row['candidate_top1']);same(row['all_logits'],151936)
        logit_pass=metric(row['full_logit_metric'],'logit',151936)
        numeric=all(op_pass)and all(block_pass)and logit_pass
    else:
        if row['candidate_top1']is not None or row['full_logit_metric']is not None:raise ValueError('Stopped candidate fabricated final logits')
        same(row['all_logits'],0);choice=None;numeric=False
    same(row['numeric_pass'],numeric)
    return dict(token=choice,complete=complete,numeric_pass=numeric,domain_pass=complete,
                operators=len(op),blocks=len(blocks),logits=151936 if complete else 0)

def reconstruct(calibration,manifest_bytes,ledger,read_phase,worker_summary,inventory):
    """Replay an externally authenticated complete scientific attempt.

    Rejects infrastructure/comparison-invalid evidence. Returns public aggregate
    statistics only. No actual reads occur except the explicit caller-supplied
    parsed-journal seam, which is synthetic in this preparation.
    """
    manifest=parse(manifest_bytes);pin=hashlib.sha256(manifest_bytes).hexdigest()
    prefills,seeds=calibration_prefixes(calibration);fields(ledger,('manifest_sha256','records','summary'));same(ledger['manifest_sha256'],pin)
    rows=ledger['records'];schedule=entries()
    if type(rows)is not list or len(rows)!=3200:raise ValueError('All3200 ledger entries required')
    statuses=[]
    for row,e in zip(rows,schedule):
        fields(row,('entry','prefix','status','native','exact','candidate','failed_stage','failure'));same(row['entry'],e)
        if row['status']=='comparison_invalid':raise IneligibleComparison('Comparison-invalid attempt cannot freeze or authorize revision')
        if row['status']not in('complete','candidate_invalid','unmeasured_suffix'):raise ValueError('Unknown ledger status')
        statuses.append((row['status'],row['failed_stage']))
    wanted_files=expected_files(tuple(statuses))
    if type(inventory)is not set or inventory!=wanted_files:raise ValueError('Exact private phase inventory differs')
    plans={s:domain_plan(manifest,s)for s in range(1,33)}
    native=InputAggregate();exact=InputAggregate();propagated={};chains={};stopped=set();seen=set()
    kind_counts={k:dict(expected=n,records=0,complete=0,candidate_invalid=0,comparison_invalid=0,unmeasured=0,top1_matches=0,top1_denominator=n)for k,n in [('prefill',3072),('teacher',64),('free',64)]}
    teacher={};free={};all_numeric=True;measured=0;complete_count=0
    def read(e,phase):
        name=f'{e["index"]:04d}-{phase}.json'
        if name in seen:raise ValueError('Duplicate phase read')
        seen.add(name);return read_phase(name)
    for e,row in zip(schedule,rows):
        kind=e['kind'];key=(kind,e['seed_ordinal']);totals=kind_counts[kind];totals['records']+=1
        if key in stopped:
            same(row['status'],'unmeasured_suffix')
            if any(row[k]is not None for k in('prefix','native','exact','candidate','failed_stage','failure')):raise ValueError('Stopped suffix fabricated evidence')
            totals['unmeasured']+=1;continue
        if row['status']=='unmeasured_suffix':raise ValueError('Unmeasured suffix lacks causal candidate stop')
        prefix=tokens(row['prefix'],e['length'])
        expected_prefix=prefills[e['index']]if kind=='prefill'else chains.get(key,seeds[e['seed_ordinal']])
        if prefix!=expected_prefix:raise ValueError('Actual own/teacher continuation differs')
        n=native_journal(read(e,'native'),e,prefix,pin,native)
        x=exact_journal(read(e,'exact'),e,prefix,pin,n,exact)
        c=candidate_journal(read(e,'candidate'),e,prefix,pin,plans[e['length']],propagated)
        same(row['native'],dict(token=n));same(row['exact'],dict(token=x,semantic_pass=True));same(row['candidate'],c)
        measured+=1
        if kind=='teacher':teacher.setdefault(e['seed_ordinal'],[]).append(n)
        if row['status']=='candidate_invalid':
            if c['complete']or row['failed_stage']!='candidate'or type(row['failure'])is not str or not row['failure']:raise ValueError('Candidate failure ledger differs')
            totals['candidate_invalid']+=1;all_numeric=False
            if kind=='free':stopped.add(key)
            elif kind=='teacher':chains[key]=prefix+(n,)
        else:
            if not c['complete']or row['failed_stage']is not None or row['failure']is not None:raise ValueError('Complete ledger evidence differs')
            totals['complete']+=1;complete_count+=1;totals['top1_matches']+=int(n==c['token']);all_numeric &= c['numeric_pass']
            if kind=='free':free.setdefault(e['seed_ordinal'],[]).append(c['token'])
            if kind!='prefill':chains[key]=prefix+((n if kind=='teacher'else c['token']),)
    if seen|{'ledger.json'}!=inventory:raise ValueError('Not every admitted phase journal consumed')
    full_sequences=sum(len(free.get(i,()))==4 and free[i]==teacher[i]for i in range(16))
    trajectories=complete_count==3200
    gates=dict(all_numeric_semantic_domains=trajectories and all_numeric,prefill_top1=kind_counts['prefill']['top1_matches']>=2765,
               teacher_top1=kind_counts['teacher']['top1_matches']>=58,complete_free_sequences=full_sequences>=12)
    phases=dict(native=measured,exact=measured,candidate=measured)
    summary=ledger['summary']
    for name in('native_invariance_by_length','exact_invariance_by_length'):same(summary[name],{str(s):True for s in range(1,33)})
    passed=all(gates.values());ranges=dict(native=native.report(),exact=exact.report(),candidate_propagated=propagated)
    rebuilt=dict(schema='expanded-candidate-ledger-v1',manifest_sha256=pin,expected_entries=3200,retained_entries=3200,
        ledger_complete=True,trajectories_complete=trajectories,by_kind=kind_counts,complete_free_sequences=full_sequences,complete_free_denominator=16,
        gates=gates,passed=passed,revision_eligible_ledger=not passed,original416_members=416,contains_tokens_logits_or_failure_text=False,
        phase_counts=phases,semantic_passes=measured,candidate_complete=complete_count,
        native_invariance_by_length={str(s):True for s in range(1,33)},exact_invariance_by_length={str(s):True for s in range(1,33)},all_length_invariance=True,
        input_ranges=ranges,limits=dict(operator_block_nrmse=.05,full_logit_nrmse=.10,semantic_nrmse=1e-5,semantic_max_abs=.001,nrmse_floor=.01),
        coefficient_selection_performed=False,encrypted_execution=False,validation_inputs_read=False,
        complete_measured_context_coverage=trajectories,revision_comparison_coverage=True,complete=True)
    fields(summary,rebuilt)
    for name,value in rebuilt.items():same(summary[name],value);same(worker_summary[name],value)
    calls=measured+32
    same(worker_summary['native_forward_accounting'],dict(expected=calls,started=calls,completed=calls,passed=True))
    same(worker_summary['native_forward_started'],calls);same(worker_summary['native_forward_completed'],calls)
    coverage=dict(ledger_entries=3200,measured_contexts=measured,candidate_complete=complete_count,candidate_invalid=measured-complete_count,
        unmeasured_free_suffixes=3200-measured,comparison_invalid=0,semantic_passes=measured,native_forward_started=calls,native_forward_completed=calls,
        all_length_invariance=True,phase_counts=phases)
    public_gates=dict(all_numeric_semantic_domains=gates['all_numeric_semantic_domains'],trajectories_complete=trajectories,
        prefill_matches=kind_counts['prefill']['top1_matches'],teacher_matches=kind_counts['teacher']['top1_matches'],whole_free_sequences=full_sequences)
    return dict(verification_passed=True,scientific_pass=passed,complete_scientific_failure_eligible=not passed,coverage=coverage,gates=public_gates,
        aggregate_sha256=hashlib.sha256(canonical(rebuilt)).hexdigest(),aggregate=rebuilt,private_files_checked=len(inventory),
        recorded_statistic_reconstruction=True,fresh_array_recomputation=False,argmax_recomputed_from_logits=False,invariance_recomputed_from_arrays=False,
        selection_authorized=False,validation_authorized=False,row0_authorized=False)
