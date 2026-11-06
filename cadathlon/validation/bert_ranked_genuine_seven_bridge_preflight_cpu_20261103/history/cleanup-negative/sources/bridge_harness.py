"""Unchanged harness dispatcher and original timer provenance, gated genuine path.

The online timer includes wrappers, phase-proof readbacks and log_size overhead.
It is never a primitive-only timer. This module performs no substitute reference
inference: the literal upstream cleartext subprocess remains in its usual place.
"""
from pathlib import Path
import argparse,json,os,subprocess,sys,time
import bridge_common as common,bridge_integration as integration,bridge_protocol as protocol
import bridge_terminal,ranked_worker as worker,ranked_timing as timing
import check_transformer_suite as process

def verify(root):
    root=Path(root);config,state=integration.invocation(root);integration.verify_receipts(config,state)
    if state['next_phase']!=7:raise ValueError('all seven complete executables required')
    envelope=integration.envelope_for(config,dict(next_phase=6,proofs=state['proofs'][:6]),common.NAMES[6])
    _,case,paths,keys,model,previous,bindings=worker.authenticate(envelope['request'],config['source_hashes'])
    protocol.authorize(envelope,config['source_hashes'],case)
    # Reauthenticate final source/phase/count proof, not merely its stdout.
    import ranked_stage_gate as gate
    final=state['proofs'][6]
    result=gate.authenticate(final['evidence'],final['report_sha256'],stage=5,case=case,expected_sources=config['source_hashes'],
        context_id=keys['context_id'],model_id=model['model_id'],key_report_sha256=envelope['request']['key_report_sha256'],model_report_sha256=envelope['request']['model_report_sha256'],previous=previous)
    summary=result['readback'];N=case.count
    # Original timed preprocessing must produce exact ranked selection bytes;
    #the retained private preparation is an expectation, never a replacement.
    generated={}
    for name,h in case.input_files:
        if name=='private_selection.json':continue # preprocessing helper-only diagnostic, not original harness output
        raw=worker.private_bytes(root/name)
        if common.digest(raw)!=h:raise ValueError('fresh original dataset/input differs from authenticated selection')
        generated[name]=h
    def labels(path):
        try:values=[int(v) for v in worker.private_bytes(path).decode('utf8').splitlines()]
        except Exception:raise ValueError('complete private prediction/label inventory invalid') from None
        if len(values)!=N or any(v not in (0,1) for v in values):raise ValueError('complete binary ranked labels required')
        return values
    folder=__import__('ranked_execution').NAMES[case.size]
    truth=labels(root/'datasets'/folder/'intermediate/test_labels.txt');pred=labels(paths['predictions']);native=labels(root/'io'/folder/'harness_model_predictions.txt')
    correct=sum(a==b for a,b in zip(pred,truth));official=sum(a==b for a,b in zip(native,truth))
    if correct!=summary['metrics']['he_correct'] or official!=case.official_correct:raise ValueError('original Quality counts disagree with authenticated private-reference acceptance')
    files=root/'measurements'/folder
    if {p.name for p in files.iterdir()}!={'results-1.json'}:raise ValueError('original default one-run results required')
    raw=__import__('key_gate').read(files/'results-1.json',4<<20);value=__import__('key_gate').parse(raw)
    stages=timing.stages_from_harness_json(raw)
    for label,want in (('Encrypted model quality',correct),('Harness model quality',official)):
        q=value['Quality'][label]
        if type(q.get('correct_predictions'))is not int or q['correct_predictions']!=want or type(q.get('total_samples'))is not int or q['total_samples']!=N:raise ValueError('original per-run Quality integer counts differ')
    marker=common.read(root/'.ranked_attempt.json')
    policy=timing.TimingPolicy('upstream_harness',config['source_inventory_sha256'],config['hardware_policy_sha256'],config['cache_policy_sha256'],True)
    run=timing.RunTiming(case.size,case.seed,marker['attempt_id'],keys['context_id'],case.input_sha256,policy,stages,correct,official)
    return dict(sample_count=N,case_size=case.size,seed=case.seed,num_runs=1,phase_count=7,full_ranked_acceptance=True,
        timing=timing.repeat_summary([run],num_runs=1),original_result_sha256=common.digest(raw),generated_input_hashes=generated,
        original_timer_fields=[dict(number=s.number,name=s.name,seconds_text=s.seconds_text) for s in stages],
        timer_scope='unchanged upstream steps5–9 including wrapper/proof work and original log_size placement',
        primitive_only_timer=False,three_seed_median_complete=False,final_stage_report_sha256=final['report_sha256'])

def run(root):
    root=Path(root);config,state=integration.invocation(root)
    if state['next_phase']!=0:raise ValueError('fresh original invocation required; no implicit restart')
    envelope=integration.envelope_for(config,state,common.NAMES[0]);_,case=worker.public_case(envelope['request'])
    protocol.authorize(envelope,config['source_hashes'],case,check_storage=True)
    start=time.monotonic();report=dict(passed=False,actual_harness_started=False,actual_harness_completed=False,pid=os.getpid(),source_hashes_start=config['source_hashes'],genuine=True)
    try:
        import bridge_assets
        report['default_model_assets_before']=bridge_assets.authenticate()
        child_env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_CACHE=report['default_model_assets_before']['hub_cache'],HUGGINGFACE_HUB_CACHE=report['default_model_assets_before']['hub_cache'])
        os.umask(0o077);argv=[sys.executable,str(root/'harness/run_submission.py'),str(case.size),'--seed',str(case.seed)] # omit --num_runs: originaldefault1
        report['argv']=argv
        with (root/'harness.stdout.log').open('w') as out,(root/'harness.stderr.log').open('w') as err:
            report['actual_harness_started']=True
            subprocess.run(argv,cwd=root,stdout=out,stderr=err,check=True,env=child_env)
        report.update(actual_harness_completed=True,original_child_reaped=True)
        report['readback']=verify(root)
        report['default_model_assets_after']=bridge_assets.authenticate()
        if report['default_model_assets_after']!=report['default_model_assets_before']:raise ValueError('original default model cache changed during harness')
        current=common.hashes(common.sources());report.update(source_hashes_end=current,source_files_unchanged=current==config['source_hashes'])
        report.update(passed=report['source_files_unchanged'],cleanup_completed=True,terminal_scope_completed=True)
    except BaseException as error:
        report.update(passed=False,cleanup_completed=False,terminal_scope_completed=False,error={'type':type(error).__name__,'message':'genuine harness failed; private error withheld'})
        try:common.write(root/'bridge_harness_report.json',report)
        except BaseException as secondary:error.add_note('harness failure publication: '+type(secondary).__name__)
        raise
    finally:report['elapsed_s']=time.monotonic()-start
    return report

def run_prepared(root,evidence,*,python=Path('/tmp/cross-pretrained-validation-env/bin/python')):
    """Owned outer process proof; all failures retain artifacts and fail closed."""
    import ranked_execution as execution
    root=Path(root);config,state=integration.invocation(root)
    if state['next_phase']!=0:raise ValueError('fresh invocation required')
    envelope=integration.envelope_for(config,state,common.NAMES[0]);_,case=worker.public_case(envelope['request'])
    target=execution.outside(evidence,execution.protected(case)+(root,Path(config['evidence_root']),envelope['request']['fixture_evidence'],envelope['request']['reference_evidence'],envelope['authorization']['size0_evidence'],envelope['authorization']['storage_path']))
    protocol.authorize(envelope,config['source_hashes'],case,check_storage=True)
    target.mkdir(mode=0o700,parents=True,exist_ok=False)
    def collect():
        primary=None;cleanup_error=None;record=None;active=False
        report=dict(passed=False,genuine=True,actual_harness_attempted=False,all_processes_reaped=False,cleanup_completed=False,source_hashes_start=config['source_hashes'])
        try:
            common.write(target/'result.json',report)
            old=process.child_environment_overrides
            process.child_environment_overrides=lambda:dict(JAX_PLATFORMS='cpu',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PATH=str(python.parent)+os.pathsep+os.environ['PATH'],OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4')
            directory=target/'harness-process';directory.mkdir();active=True;report['actual_harness_attempted']=True
            try:record=process.run_case([str(python),str(root/'_source/bridge_harness.py'),'--root',str(root)],directory,timeout=None,cleanup_grace_s=120.)
            finally:process.child_environment_overrides=old
            active=False;report['process']=record
            proof=common.read(root/'bridge_harness_report.json');report['proof']=proof
            passed=(type(record.get('returncode'))is int and record['returncode']==0 and not record.get('error')
                and all(record.get(n)is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
                and proof.get('passed')is True and proof.get('termination_scope_completed')is True
                and proof.get('source_hashes_start')==config['source_hashes'] and proof.get('source_hashes_end')==config['source_hashes']
                and record.get('result')==dict(passed=True,genuine=True,report_sha256=common.digest((root/'bridge_harness_report.json').read_bytes())))
            if not passed:raise ValueError('strict genuine harness process/proof incomplete')
            report['passed']=True
        except BaseException as error:
            primary=error;report.update(passed=False,error={'type':type(error).__name__,'message':'outer genuine harness failed; private error withheld'})
        finally:
            def attempt(label,fn):
                nonlocal cleanup_error
                try:return fn()
                except BaseException as error:
                    report.setdefault('cleanup_errors',[]).append({'operation':label,'type':type(error).__name__})
                    if primary is not None:primary.add_note(label+': '+type(error).__name__)
                    elif cleanup_error is None or (not isinstance(error,Exception) and isinstance(cleanup_error,Exception)):cleanup_error=error
            from terminal import terminal_scope
            from cleanup import cleanup_signals
            with terminal_scope(report,target/'result.json',common.write,cleanup_signals,lambda:(primary,cleanup_error)):
                if active:record=attempt('active harness process evidence',lambda:common.read(target/'harness-process/process.json'))
                report['process']=record
                report['all_processes_reaped']=bool(record and all(record.get(n)is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed')))
                after=attempt('source readback',lambda:common.hashes(common.sources()))
                report.update(source_hashes_end=after,source_files_unchanged=after==config['source_hashes'],cleanup_completed=primary is None and cleanup_error is None and report['all_processes_reaped'])
                report['passed']=bool(report['passed'] and report['cleanup_completed'] and report['source_files_unchanged'])
                attempt('outer report',lambda:common.write(target/'result.json',report))
                if primary is not None:raise primary
                if cleanup_error is not None:raise cleanup_error
        return report
    return bridge_terminal.finish(target/'result.json',collect,common.write)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    try:r=bridge_terminal.finish(args.root/'bridge_harness_report.json',lambda:run(args.root),common.write)
    except BaseException as error:
        print(json.dumps(dict(passed=False,genuine=True,error_type=type(error).__name__)));raise SystemExit(130 if not isinstance(error,Exception) else 1)
    print(json.dumps(dict(passed=r['passed'],genuine=True,report_sha256=common.digest((args.root/'bridge_harness_report.json').read_bytes()))))
    raise SystemExit(0 if r['passed'] else 1)
