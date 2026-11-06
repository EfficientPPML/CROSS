"""Independent terminal-only reconstruction of the fixed61 initial observer.

No imports from producer/selector code; no model construction or validation read.
The command requires a separately recorded authoritative outer exit0/reap receipt.
It verifies producer success BEFORE opening calibration or private journals.
"""
from pathlib import Path
import argparse, copy, hashlib, io, json, math, os, stat, tarfile

BASE = '7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
MANIFEST = '8131c5ac045f1cf274a493bbc14278e330c0a8e057079edc4f10f9a23dd9bf91'
CALIBRATION = '80eeaefad423167ee0710797d7eb76d8cf3a422e6f3b2e6069932d682835963b'
GRANT = 'e0068125a3a271880a69231d5108fe9591ae10489e05f06d043646d8be192704'
LAB = Path('/tmp/qwen-expanded-prefix-observer-runtime-v5-20261104')
ACTUAL = Path('/tmp/qwen-expanded-prefix-observer-actual-v2-20261104')
PRIVATE = Path('/tmp/qwen-expanded-prefix-observer-private-v2-20261104')
REVIEW = Path('/tmp/qwen-expanded-prefix-observer-terminal-root-review-20261104')
NORMS = {f'{i}.{part}.inverse_sqrt' for i in range(28) for part in ('input','post','q','k')} | {'final.inverse_sqrt'}
LABELS = NORMS | {f'{i}.silu' for i in range(28)} | {f'{i}.softmax.{phase}' for i in range(28) for phase in ('raw_valid','centered')}

def digest(raw): return hashlib.sha256(raw).hexdigest()
def canonical(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def parse(raw):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError('duplicate JSON field')
            out[k]=v
        return out
    def invalid(value): raise ValueError('nonfinite JSON constant')
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)
def same(a,b):
    if canonical(a)!=canonical(b): raise ValueError('exact structured binding differs')
def integer(value,wanted):
    if type(value)is not int or value!=wanted: raise ValueError('exact integer differs')
def flags(row,yes=(),no=()):
    if any(row.get(k)is not True for k in yes) or any(row.get(k)is not False for k in no):
        raise ValueError('typed completion flags differ')
def identity(s): return [s.st_dev,s.st_ino]
def stamp(s): return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode,s.st_uid)
BODY='_initial_root_review_body_origin'
CLEANUP='_initial_root_review_cleanup_origin'
def cleanup_origin(error):
    return error is not None and getattr(error,BODY,False)is not True and getattr(error,CLEANUP,False)is True
def protect_body(error):
    if error is not None and not cleanup_origin(error): setattr(error,BODY,True)
    return error
def cleanup_choice(primary,error,body_primary):
    if error is primary:return primary
    if body_primary is not None:protect_body(body_primary)
    if getattr(error,BODY,False)is not True:setattr(error,CLEANUP,True)
    if primary is None:return error
    if body_primary is None and cleanup_origin(primary) and isinstance(primary,Exception) and not isinstance(error,Exception):
        error.add_note('Earlier cleanup failure: '+type(primary).__name__);return error
    primary.add_note('review cleanup close failure: '+type(error).__name__);return primary
def close(fd,primary):
    body=None if primary is None or cleanup_origin(primary) else protect_body(primary)
    try: os.close(fd)
    except BaseException as error:
        chosen=cleanup_choice(primary,error,body)
        if chosen is not primary:raise chosen
def read(path, expected=None, *, limit=32<<20, private=False):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)
    primary=None
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=limit: raise ValueError('bounded regular file required')
        if private and (stat.S_IMODE(before.st_mode)!=0o600 or before.st_uid!=os.getuid()): raise ValueError('private0600 file required')
        chunks=[];remaining=limit+1
        while remaining:
            data=os.read(fd,min(1<<20,remaining))
            if not data: break
            chunks.append(data);remaining-=len(data)
        raw=b''.join(chunks)
        if stamp(before)!=stamp(os.fstat(fd)) or stamp(before)!=stamp(os.stat(path,follow_symlinks=False)) or len(raw)!=before.st_size:
            raise ValueError('input changed while reading')
        if expected is not None and digest(raw)!=expected: raise ValueError('input digest differs')
        return raw,before
    except BaseException as error:
        primary=error;raise
    finally: close(fd,primary)
def directory(path, expected=None):
    s=os.stat(path,follow_symlinks=False)
    if not stat.S_ISDIR(s.st_mode) or stat.S_IMODE(s.st_mode)!=0o700 or s.st_uid!=os.getuid(): raise ValueError('private0700 directory required')
    if expected is not None and identity(s)!=expected: raise ValueError('directory identity differs')
    return s
def file_map(root):
    result={}
    for p in root.rglob('*'):
        if p.is_symlink(): raise ValueError('source symlink')
        if p.is_file(): result[p.relative_to(root).as_posix()]=digest(read(p)[0])
        elif not p.is_dir(): raise ValueError('source special file')
    return result
def absent(pid):
    if type(pid)is not int or pid<=1 or Path('/proc',str(pid)).exists(): raise ValueError('process not independently absent')
def absent_group(pgid):
    try: os.killpg(pgid,0)
    except ProcessLookupError: return
    raise ValueError('process group still exists')

def entries():
    out=[]
    for p in range(96):
        for s in range(1,33): out.append(dict(index=len(out),kind='prefill',prompt_ordinal=p,seed_ordinal=None,step=None,length=s,original416_member=s in (8,16,32)))
    for kind in ('teacher','free'):
        for seed in range(16):
            for step in range(4): out.append(dict(index=len(out),kind=kind,prompt_ordinal=None,seed_ordinal=seed,step=step,length=16+step,original416_member=True))
    return out
def tokens(value,length):
    if type(value)is not list or len(value)!=length or any(type(x)is not int or not 0<=x<151936 for x in value): raise ValueError('token prefix differs')
    return value
def counts(s):
    out={n:1 if n=='final.inverse_sqrt' else s*(16 if '.q.'in n else 8 if '.k.'in n else 1) for n in NORMS}
    for i in range(28):
        out[f'{i}.silu']=s*3072
        for phase in ('raw_valid','centered'): out[f'{i}.softmax.{phase}']=16*s*(s+1)//2
    return out
def statistic(row,count,positive=False):
    if type(row)is not dict or set(row)!={'count','minimum','maximum','nonfinite','nonpositive'}: raise ValueError('statistic fields differ')
    integer(row['count'],count);integer(row['nonfinite'],0)
    if type(row['nonpositive'])is not int or not 0<=row['nonpositive']<=count: raise ValueError('invalid nonpositive count')
    lo,hi=row['minimum'],row['maximum']
    if any(type(v)is not float or not math.isfinite(v)for v in (lo,hi)) or lo>hi: raise ValueError('invalid finite bounds')
    if (hi<=0 and row['nonpositive']!=count) or (lo>0 and row['nonpositive']!=0): raise ValueError('bounds/count contradiction')
    if positive and (lo<=0 or row['nonpositive']): raise ValueError('norm input is not positive')
def merge(target,source):
    for label,row in source.items():
        if label not in target: target[label]=copy.deepcopy(row);continue
        prior=target[label]
        if set(prior)!=set(row): raise ValueError('statistic schema drift')
        for k,v in row.items():
            if k in ('count','nonfinite','nonpositive','below','above'): prior[k]+=v
            elif k=='minimum': prior[k]=min(prior[k],v)
            elif k=='maximum': prior[k]=max(prior[k],v)
            elif prior[k]!=v: raise ValueError('statistic invariant drift')
def observation(value,s):
    flags(value,yes=('complete',),no=('invalid',))
    integer(value['sequence_length'],s);integer(value['norm_input_sites'],113);integer(value['silu_gate_sites'],28)
    if set(value['ranges'])!=LABELS or set(value['score_rows'])!=set(map(str,range(28))): raise ValueError('nonlinear sites differ')
    for name,count in counts(s).items(): statistic(value['ranges'][name],count,name in NORMS)
    for layer,rows in value['score_rows'].items():
        if set(rows)!=set(map(str,range(1,s+1))): raise ValueError('per-query causal rows differ')
        merged={}
        for key,row in rows.items():
            k=int(key)
            if set(row)!={'query','key_count','shape','k1_bypass','reciprocal_observations','raw_valid','centered'}: raise ValueError('score fields differ')
            integer(row['query'],k-1);integer(row['key_count'],k);integer(row['reciprocal_observations'],0)
            same(row['shape'],[1,16,k])
            if row['k1_bypass']is not(k==1): raise ValueError('K1 bypass differs')
            for phase in ('raw_valid','centered'): statistic(row[phase],16*k)
            if k==1 and (row['centered']['minimum']!=0. or row['centered']['maximum']!=0.): raise ValueError('K1 centered scores differ')
            merge(merged,{phase:row[phase] for phase in ('raw_valid','centered')})
        for phase in ('raw_valid','centered'): same(merged[phase],value['ranges'][f'{layer}.softmax.{phase}'])
def add_observation(aggregate,value):
    merge(aggregate['ranges'],value['ranges'])
    for layer,rows in value['score_rows'].items():
        target=aggregate['score_rows'].setdefault(layer,{})
        for key,row in rows.items():
            if key not in target:
                target[key]=copy.deepcopy(row);target[key]['contexts']=1
            else:
                old=target[key]
                for field in ('query','key_count','shape','k1_bypass','reciprocal_observations'): same(old[field],row[field])
                for field in ('raw_valid','centered'):
                    temporary={field:old[field]};merge(temporary,{field:row[field]})
                old['contexts']+=1
    aggregate['contexts']+=1
def empty_aggregate():
    return dict(contexts=0,ranges={},score_rows={},reciprocal_inputs_observed=False,scope='Native/exact input observations only; K1 is an explicit bypass.')
def validate_output(path,protected):
    if path!=REVIEW or not path.is_absolute() or path.resolve()!=path or path.exists() or path.is_symlink(): raise ValueError('exact fresh canonical root-review output required')
    for other in protected:
        other=Path(other)
        if path==other or path in other.parents or other in path.parents: raise ValueError('root-review output overlaps protected inputs')

class Publication:
    def __init__(self,path):
        self.path=path;self.fd=None
        path.mkdir(mode=0o700)
        self.fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            self.owner=identity(os.fstat(self.fd));self.check()
        except BaseException as error:
            fd,self.fd=self.fd,None;close(fd,error);raise
    def check(self):
        directory(self.path,self.owner)
        if identity(os.fstat(self.fd))!=self.owner: raise ValueError('publication directory handle differs')
    def publish(self,name,value):
        if name not in ('reconstructed-snapshot.json','review.json'): raise ValueError('fixed publication name required')
        self.check();raw=canonical(value)
        fd=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=self.fd)
        primary=None
        try:
            opened=os.fstat(fd);view=memoryview(raw)
            while view:
                n=os.write(fd,view)
                if n<=0: raise OSError('publication stalled')
                view=view[n:]
            os.fsync(fd)
            if identity(os.fstat(fd))!=identity(opened): raise ValueError('publication file handle differs')
        except BaseException as error:
            primary=error;raise
        finally: close(fd,primary)
        self.check();path=self.path/name;raw,s=read(path,digest(raw),private=True)
        if identity(s)!=identity(opened): raise ValueError('publication file replaced')
        return dict(path=str(path),sha256=digest(raw),bytes=len(raw),identity=identity(s),parent_identity=self.owner)
    def close(self,primary=None):
        if self.fd is not None:
            fd,self.fd=self.fd,None;close(fd,primary)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--outer-receipt',type=Path,required=True);ap.add_argument('--outer-receipt-sha256',required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    validate_output(args.output,(LAB,ACTUAL,PRIVATE,Path('/tmp/qwen-expanded-prefix-token-inputs-20261104-v1')))
    outer=parse(read(args.outer_receipt,args.outer_receipt_sha256)[0])
    same({k:outer[k]for k in ('session','exit_code','reaped')},dict(session=2619,exit_code=0,reaped=True))
    raw,_=read(ACTUAL/'report.json',outer['report_sha256']);parent=parse(raw);parent_sha=digest(raw)
    raw,_=read(ACTUAL/'worker/report.json',parent['worker_report_sha256']);worker=parse(raw);worker_sha=digest(raw)
    same(worker,parent['worker'])
    manifest_raw,_=read(Path('/tmp/qwen-expanded-prefix-observer-v5-sources-v1-20261104.json'),MANIFEST)
    sources=parse(manifest_raw);integer(len(sources),61)
    raw,_=read(ACTUAL/'sources.tar.gz',parent['source_archive']['sha256']);archive_sha=digest(raw)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as archive:
        members=archive.getmembers()
        if len(members)!=61 or len({m.name for m in members})!=61 or any(not m.isfile() or m.name not in sources or not 0<m.size<=8<<20 for m in members): raise ValueError('archive members differ')
        blobs={m.name:archive.extractfile(m).read()for m in members}
    same({n:digest(b)for n,b in blobs.items()},sources)
    for root in (LAB,ACTUAL/'execution_source'): same(file_map(root),sources)
    read(ACTUAL/'source_manifest.json',MANIFEST)
    common=('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash')
    for r in (parent,worker):
        flags(r,yes=common,no=('coefficient_selection',))
        same(r['source_hashes_start'],sources);same(r['source_hashes_end'],sources)
        if r['source_manifest_sha256']!=MANIFEST or r['audit_denials']!=[] or 'error_type'in r: raise ValueError('terminal producer rejected')
    flags(parent,yes=('actual_observer_stage','child_reaped','child_group_absent'),no=('model_forward',))
    integer(parent['child_returncode'],0)
    integer(parent['child_pid'],worker['pid']);integer(parent['child_pgid'],worker['pid'])
    for pid in (parent['pid'],worker['pid']): absent(pid);absent_group(pid)
    flags(worker,yes=('model_loaded','model_forward','model_loading_exact','model_execution_requested','initial_selection_admissible','calibration_reauthenticated','private_directory_handle_closed','private_outputs_retained','external_inputs_never_removed','exact_order'),no=('tokenizer_execution','validation_inputs_read','coefficient_selection_performed','encrypted_execution','baseline_ranges_in_initial_selection','contains_tokens_or_logits'))
    for k,n in dict(expected_entries=3200,retained_entries=3200,semantic_passes=3200,baseline_complete=64,original416_members=416,new_protocol_candidates_used=0,new_candidates_evaluated=0,native_forward_started=3232,native_forward_completed=3232).items(): integer(worker[k],n)
    if worker['schema']!='qwen-expanded-initial-observer-ledger-v1' or worker['baseline_manifest_sha256']!=BASE or worker['grant_sha256']!=GRANT or worker['calibration_sha256']!=CALIBRATION: raise ValueError('frozen run identity differs')
    for k in ('hook_invariance_by_length','exact_invariance_by_length'): same(worker[k],{str(n):True for n in range(1,33)})
    same(worker['prefills_by_length'],{str(n):96 for n in range(1,33)})
    same(worker['by_kind'],{kind:dict(expected=n,records=n,observed=n,complete=n,invalid=0,unmeasured=0,semantic_failures=0)for kind,n in (('prefill',3072),('teacher',64),('free',64))})
    backend={n:p['sha256']for n,p in parse(blobs['backend_sources.json']).items()}
    same(worker['backend_sources_before'],backend);same(worker['backend_sources_after'],backend)
    same(worker['installed_dependencies'],parse(blobs['dependencies.json']))
    same(worker['architecture'],dict(num_hidden_layers=28,hidden_size=1024,intermediate_size=3072,num_attention_heads=16,num_key_value_heads=8,head_dim=128,vocab_size=151936))
    same(worker['backend_environment'],dict(USE_TF='0',USE_FLAX='0',USE_TORCH='1',USE_TORCH_XLA='0',FORCE_TF_AVAILABLE='0'))
    if worker['native_dtype']!='float32' or worker['exact_dtype']!='float64': raise ValueError('native/exact precision differs')
    same(worker['model_asset_sha256'],{n:h for n,h in parse(blobs['public/model_pins.json'])['asset_sha256'].items()if n in ('config.json','generation_config.json','model.safetensors')})
    ext=parse(blobs['external_inputs.json']);calpin=ext['calibration']
    # Terminal public/source proof accepted above. Private evidence opens below.
    directory(Path(calpin['path']).parent,calpin['parent_identity'])
    calraw,cals=read(Path(calpin['path']),CALIBRATION,limit=calpin['bytes'],private=True)
    if identity(cals)!=calpin['identity'] or len(calraw)!=calpin['bytes']: raise ValueError('calibration identity differs')
    calibration=parse(calraw)
    if set(calibration)!={'prefills','decode_seeds'} or len(calibration['prefills'])!=3072 or len(calibration['decode_seeds'])!=16: raise ValueError('calibration schema differs')
    prompt_ids=[]
    for p in range(96):
        family=calibration['prefills'][p*32:(p+1)*32]
        full=tokens(family[-1]['token_ids'],32);name=family[-1]['id']
        if type(name)is not str or not name: raise ValueError('prompt ID required')
        for s,row in enumerate(family,1):
            if set(row)!={'prompt_ordinal','id','sequence_length','token_ids','original_prefill_member'} or row['id']!=name: raise ValueError('prefill family schema differs')
            integer(row['prompt_ordinal'],p);integer(row['sequence_length'],s)
            if row['original_prefill_member']is not(s in (8,16,32)): raise ValueError('prefill membership differs')
            same(tokens(row['token_ids'],s),full[:s])
        prompt_ids.append(name)
    if len(set(prompt_ids))!=96: raise ValueError('prompt IDs are not unique')
    seed_prompts=set()
    for index,seed in enumerate(calibration['decode_seeds']):
        if set(seed)!={'seed_ordinal','prompt_ordinal','id','token_ids'}: raise ValueError('seed schema differs')
        integer(seed['seed_ordinal'],index);p=seed['prompt_ordinal']
        if type(p)is not int or not 0<=p<96 or p in seed_prompts or seed['id']!=prompt_ids[p]: raise ValueError('seed prompt mapping differs')
        same(tokens(seed['token_ids'],16),calibration['prefills'][p*32+15]['token_ids']);seed_prompts.add(p)
    private_stat=directory(PRIVATE,worker['private_output']['identity'])
    if worker['private_output']['path']!=str(PRIVATE): raise ValueError('private output path differs')
    expected_names={f'{i:04d}-{phase}.json'for i in range(3200)for phase in ('started','native','exact')}|{f'{i:04d}-baseline.json'for i in range(3136,3200)}|{'initial-ledger.json','input-ranges.json'}
    if set(os.listdir(PRIVATE))!=expected_names: raise ValueError('complete private inventory differs')
    inventory={};stamps={}
    def journal(name):
        raw,s=read(PRIVATE/name,limit=16<<20,private=True)
        if name in inventory: raise ValueError('journal read twice')
        inventory[name]=dict(sha256=digest(raw),bytes=len(raw),identity=identity(s));stamps[name]=stamp(s)
        return parse(raw)
    ledger=journal('initial-ledger.json')
    if set(ledger)!={'records','baseline_manifest_sha256'} or ledger['baseline_manifest_sha256']!=BASE or len(ledger['records'])!=3200: raise ValueError('private initial ledger differs')
    aggregates={kind:empty_aggregate()for kind in ('native','exact')};baseline_ranges={};chains={}
    for entry,record in zip(entries(),ledger['records']):
        i,s,kind=entry['index'],entry['length'],entry['kind'];same(record['entry'],entry)
        if set(record)!={'entry','prefix','status','native','exact','baseline','failed_stage','failure'} or record['status']!='complete' or record['failed_stage']is not None or record['failure']is not None: raise ValueError('incomplete private context')
        if kind=='prefill':
            source=calibration['prefills'][i]
            integer(source['prompt_ordinal'],entry['prompt_ordinal']);integer(source['sequence_length'],s)
            if source['original_prefill_member']is not entry['original416_member']: raise ValueError('prefill membership differs')
            prefix=tokens(source['token_ids'],s)
        else:
            key=(kind,entry['seed_ordinal']);seed=calibration['decode_seeds'][entry['seed_ordinal']]
            integer(seed['seed_ordinal'],entry['seed_ordinal'])
            prefix=chains.get(key,tokens(seed['token_ids'],16))
        tokens(prefix,s);same(record['prefix'],prefix)
        start=journal(f'{i:04d}-started.json')
        same(start,dict(entry=entry,prefix=prefix,context_complete=False,status='attempt_started'))
        phases={}
        for phase in ('native','exact'):
            row=journal(f'{i:04d}-{phase}.json');same(row['entry'],entry)
            if row['status']!='observed' or type(row['token'])is not int or not 0<=row['token']<151936: raise ValueError('invalid native/exact phase')
            if not isinstance(row['logits_sha256'],str) or len(row['logits_sha256'])!=64: raise ValueError('missing full-logit hash')
            if digest(canonical(row['observations']))!=row['ranges_sha256']: raise ValueError('observation digest differs')
            observation(row['observations'],s);add_observation(aggregates[phase],row['observations']);phases[phase]=row
        integer(phases['native']['all_final_logits'],151936)
        metric=phases['exact']['semantic'];flags(metric,yes=('passed','finite','shape_matches','top1_agreement'))
        integer(metric['all_final_logits_checked'],151936)
        for field,bound in (('max_abs',.001),('nrmse',1e-5)):
            if type(metric[field])is not float or not math.isfinite(metric[field]) or not 0<=metric[field]<=bound: raise ValueError('semantic metric rejected')
        same(record['native'],dict(token=phases['native']['token']))
        same(record['exact'],dict(token=phases['exact']['token'],semantic_pass=True))
        integer(phases['native']['token'],phases['exact']['token'])
        if kind=='free':
            row=journal(f'{i:04d}-baseline.json');same(row['entry'],entry)
            if row['status']!='complete' or row['failure']is not None or row['ranges_role']!='separate_initial_generator_diagnostic_only': raise ValueError('baseline trace incomplete')
            tokens([row['token']],1);same(record['baseline'],dict(token=row['token'],domain_pass=True))
            for value in row['domain_ranges'].values():
                for field in ('below','above','nonfinite'): integer(value[field],0)
            merge(baseline_ranges,row['domain_ranges']);chains[key]=prefix+[row['token']]
        elif kind=='teacher':
            if record['baseline']is not None: raise ValueError('teacher baseline contamination')
            chains[key]=prefix+[phases['native']['token']]
        elif record['baseline']is not None: raise ValueError('prefill baseline contamination')
        if (i+1)%400==0: print(json.dumps(dict(reconstructed_contexts=i+1)),flush=True)
    union=copy.deepcopy(aggregates['native']['ranges']);merge(union,aggregates['exact']['ranges'])
    reconstructed=dict(**aggregates,native_exact_union=union,baseline_generator_separate_diagnostic=baseline_ranges,initial_bounds_include_baseline=False)
    private_ranges=journal('input-ranges.json');same(private_ranges,reconstructed);same(worker['input_ranges'],reconstructed)
    descriptor=worker['private_ranges_descriptor']
    for k in ('sha256','bytes','identity'): same(descriptor[k],inventory['input-ranges.json'][k])
    if descriptor['mode']!='0600': raise ValueError('private range mode differs')
    integer(len(inventory),9666)
    if len({tuple(row['identity'])for row in inventory.values()})!=9666: raise ValueError('private journal inode alias')
    if set(os.listdir(PRIVATE))!=expected_names or identity(directory(PRIVATE))!=identity(private_stat): raise ValueError('private directory changed')
    for name,recorded in stamps.items():
        if stamp(os.stat(PRIVATE/name,follow_symlinks=False))!=recorded: raise ValueError('private journal changed after reconstruction')
    read(Path(calpin['path']),CALIBRATION,limit=calpin['bytes'],private=True)
    snapshot=dict(schema='qwen-initial3200-reconstructed-snapshot-v1',parent_report_sha256=parent_sha,worker_report_sha256=worker_sha,source_archive_sha256=archive_sha,source_manifest_sha256=MANIFEST,calibration_sha256=CALIBRATION,input_ranges=reconstructed,ledger_sha256=inventory['initial-ledger.json']['sha256'],inventory=inventory)
    publication=Publication(args.output);primary=None
    try:
        snapshot_descriptor=publication.publish('reconstructed-snapshot.json',snapshot)
        review=dict(schema='qwen-initial3200-root-review-contract-v2',passed=True,parent_report_sha256=parent_sha,worker_report_sha256=worker_sha,source_archive_sha256=archive_sha,source_manifest_sha256=MANIFEST,outer_exit_code=0,outer_reaped=True,outer_group_absent=True,worker_exit_code=0,worker_reaped=True,worker_group_absent=True,exact_ordered_contexts=3200,per_context_norm_input_sites=113,per_context_silu_sites=28,per_context_causal_score_sites=28,baseline_free_contexts=64,baseline_excluded=True,per_context_coverage_independently_verified=True,aggregate_ranges_independently_recomputed=True,source_inventory_independently_verified=True,native_model_loaded=True,native_forward_started=3232,native_forward_completed=3232,reconstruction_script_sha256=digest(read(Path(__file__))[0]),snapshot_descriptor=snapshot_descriptor,private_inventory_sha256=digest(canonical(inventory)),private_files=9666,calibration_sha256=CALIBRATION,private_journals_immutable_at_review=True)
        review_descriptor=publication.publish('review.json',review)
        publication.check()
    except BaseException as error:
        primary=error;raise
    finally: publication.close(primary)
    print(json.dumps(dict(passed=True,pid=os.getpid(),pgid=os.getpgrp(),review=review_descriptor,snapshot=snapshot_descriptor,scope='Initial3200 native/exact range reconstruction; no new coefficient selection or heldout reads')))

if __name__=='__main__': main()
