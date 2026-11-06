"""Source-bound original-harness synthetic integration proof; no genuine mode."""
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
from decimal import Decimal
import argparse,json,os,sys,time,xml.etree.ElementTree as ET
import integration_common as c
import harness_integration as h
import check_transformer_suite as process
from ranked_terminal import finish
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope


def test_worker(output):
    import pytest
    output=Path(output)
    with (output/'tests.log').open('w') as log,redirect_stdout(log),redirect_stderr(log),termination_signals():
        code=pytest.main(['-q','-x',str(c.HERE/'test_harness_integration.py'),str(c.HERE/'test_harness_timing.py'),
            '--junitxml='+str(output/'tests.xml'),'--basetemp='+str(output/'owned-fixtures')])
    return dict(passed=code==0,synthetic=True,genuine_eligible=False,pytest_exit_code=int(code))


def collect(output):
    output=Path(output);output.mkdir(exist_ok=False);started=time.monotonic();primary=None;cleanup_error=None;before={};active=False;record=None
    report=dict(schema='cross-ranked-original-seven-harness-preflight-v1',passed=False,synthetic=True,genuine_eligible=False,
        actual_ranked_launch=False,real_key_generation=False,real_model_forward=False,real_ciphertext_work=False,pid=os.getpid(),all_processes_reaped=False)
    def attempt(label,call):
        nonlocal cleanup_error
        try:return call()
        except BaseException as error:
            report.setdefault('cleanup_errors',[]).append({'operation':label,'type':type(error).__name__})
            if primary is not None:primary.add_note(label+': '+type(error).__name__)
            elif cleanup_error is None or (not isinstance(error,Exception) and isinstance(cleanup_error,Exception)):cleanup_error=error
    try:
        c.write(output/'report.json',report)
        raw=c.sources();before=c.hashes(raw);h.archive(output/'sources.tar.gz',raw);c.write(output/'source_manifest.json',before)
        if not h.verify_archive(output/'sources.tar.gz',before):raise ValueError('initial source archive differs')
        report.update(source_hashes_start=before,source_count=len(before),parent_source_count=272,original_harness_source_count=11,
            source_archive_sha256=c.digest((output/'sources.tar.gz').read_bytes()))
        c.write(output/'report.json',report);directory=output/'test-process';directory.mkdir();active=True
        previous=process.child_environment_overrides
        process.child_environment_overrides=lambda:dict(JAX_PLATFORMS='cpu',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        try:record=process.run_case([str(h.PYTHON),str(c.HERE/'preflight.py'),'--test-worker',str(output)],directory,timeout=None,cleanup_grace_s=120.)
        finally:process.child_environment_overrides=previous
        active=False;report['test_process']=record
        if record.get('returncode')!=0 or record.get('error') or record.get('result')!=dict(passed=True,synthetic=True,genuine_eligible=False,pytest_exit_code=0):raise ValueError('complete synthetic test worker required')
        if not all(record.get(n)is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed')):raise ValueError('test worker cleanup incomplete')
        xml=ET.parse(output/'tests.xml');suites=list(xml.getroot().iter('testsuite'));counts={n:sum(int(s.get(n,0)) for s in suites) for n in ('tests','failures','errors','skipped')}
        if counts['tests']!=45 or any(counts[n] for n in ('failures','errors','skipped')):raise ValueError('exact complete45 checks required')
        report['tests']=counts
        parents=list((output/'owned-fixtures').glob('seven-owned*'));parents=[p for p in parents if p.is_dir() and not p.is_symlink()]
        if len(parents)!=1:raise ValueError('one synthetic fixture parent required')
        parent=parents[0];cases=[];contexts=set()
        for size in (1,2,3):
            for seed in (42,43,44):
                root=parent/f'{size}-{seed}';result=c.read(parent/f'{size}-{seed}-evidence/result.json');proof=c.read(root/'harness_report.json')
                if result['passed']is not True or proof['passed']is not True or proof['termination_scope_completed']is not True:raise ValueError('all nine full original invocations required')
                if c.read(root/'invocation.json')['source_hashes']!=before:raise ValueError('case source identity differs')
                r=proof['readback'];contexts.add(r['context_id']);cases.append(dict(size=size,seed=seed,count=r['count'],num_runs=r['num_runs'],
                    synthetic_correct=r['synthetic_correct'],synthetic_native_correct=r['synthetic_native_correct'],context_id=r['context_id'],
                    stage_count=r['stage_count'],fresh_unique_request_count=r['fresh_unique_request_count'],timing=r['timing'],original_timing=r['original_timing'],
                    source_inventory_sha256=r['source_inventory_sha256'],report_sha256=c.digest((root/'harness_report.json').read_bytes()),
                    actual_worker_pids=r['distinct_stage_pids'],harness_worker_pid=proof['pid']))
        if len(contexts)!=9:raise ValueError('nine fresh synthetic setup contexts required')
        repeat=c.read(parent/'repeat/harness_report.json')['readback']
        if repeat['stage_count']!=12 or repeat['fresh_unique_request_count']!=200:raise ValueError('explicit repeat setup/request semantics differ')
        medians=[]
        for size in (1,2,3):
            values=[Decimal(r['timing']['median_online_seconds']) for r in cases if r['size']==size]
            medians.append(dict(size=size,seeds=[42,43,44],synthetic_diagnostic_median_seconds=str(sorted(values)[1]),official_ranked_eligible=False))
        report.update(cases=cases,explicit_repeat=repeat,synthetic_per_size_medians=medians,ordered_ranked_records=2124,synthetic_online_handler_records=2324,
            original_seven_process_invocations=9,original_num_runs_default_omitted=True,all_processes_reaped=True,passed=True)
    except BaseException as error:
        primary=error;report.update(passed=False,error={'type':type(error).__name__,'message':'synthetic preflight failed; preserved public logs'})
    finally:
        with terminal_scope(report,output/'report.json',c.write,cleanup_signals,lambda:(primary,cleanup_error)):
            if active:
                record=attempt('active process evidence',lambda:c.read(output/'test-process/process.json'))
                report['test_process']=record
            report['all_processes_reaped']=bool(record and all(record.get(n)is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed')))
            after=attempt('source readback',lambda:c.hashes(c.sources())) if before else {}
            report.update(source_hashes_end=after or {},source_files_unchanged=bool(before) and before==after,
                source_archive_verified=bool(before) and bool(attempt('archive readback',lambda:h.verify_archive(output/'sources.tar.gz',before))),
                cleanup_completed=cleanup_error is None and report['all_processes_reaped'],elapsed_s=time.monotonic()-started)
            report['passed']=bool(report['passed'] and report['source_files_unchanged'] and report['source_archive_verified'] and report['cleanup_completed'])
            attempt('report publication',lambda:c.write(output/'report.json',report))
            if primary is not None:raise primary
            if cleanup_error is not None:raise cleanup_error
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--test-worker',type=Path);group.add_argument('--output',type=Path);args=parser.parse_args()
    if args.test_worker:
        try:r=test_worker(args.test_worker)
        except BaseException as error:
            print(json.dumps(dict(passed=False,error_type=type(error).__name__)));raise SystemExit(130 if not isinstance(error,Exception) else 1)
        print(json.dumps(r));raise SystemExit(0 if r['passed'] else 1)
    try:r=finish(args.output/'report.json',lambda:collect(args.output),c.write)
    except BaseException as error:
        print(json.dumps(dict(passed=False,error_type=type(error).__name__)));raise SystemExit(130 if not isinstance(error,Exception) else 1)
    print(json.dumps(dict(passed=r['passed'],source_count=r['source_count'],tests=r['tests'],source_archive_sha256=r['source_archive_sha256'])))
