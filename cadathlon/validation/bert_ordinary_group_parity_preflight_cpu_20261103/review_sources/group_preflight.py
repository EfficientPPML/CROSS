"""Source-bound mock/import/public-gate preflight, never genuine parity."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import group_collect as collector
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope
protocol=collector.group_protocol
process=collector.process
write=collector.write

def main(output):
    protocol.environment();output=Path(output).absolute()
    # A public preparation attempt may write only its explicitly owned lab tree.
    if output.parent!=HERE or output==HERE:raise ValueError('preflight output must be a new direct lab child')
    output.mkdir(exist_ok=False);start=time.monotonic();primary=None;active=None;before={};records=[]
    report=dict(passed=False,pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),actual_execution_attempted=False,
        scope='Mocked lifecycle/representation checks, CPU installed entrypoint denials and public prerequisites only; no context, key, ciphertext, private reference or model payload reads; no HE execution.')
    try:
        write(output/'report.json',report)
        raw=protocol.sources();before=protocol.hashes(raw);collector.archive(output/'sources.tar.gz',raw)
        if not collector.check_archive(output/'sources.tar.gz',before):raise ValueError('initial source archive differs')
        report.update(source_hashes_before=before,source_count=len(before),reviewed_sources=protocol.check_sources(allow_pending=True))
        write(output/'report.json',report)
        for kind,artifact in (('tests',output/'tests.xml'),('installed',output/'installed'),('public',output/'public-gates')):
            active=output/('process-'+kind);active.mkdir();write(output/'report.json',report)
            command=[sys.executable,str(HERE/'group_preflight_child.py'),kind,str(artifact)]
            previous=process.child_environment_overrides
            process.child_environment_overrides=lambda:{'JAX_PLATFORMS':'cpu','PYTHONUNBUFFERED':'1'}
            try:record=process.run_case(command,active,timeout=None,cleanup_grace_s=30.)
            finally:process.child_environment_overrides=previous
            records.append(dict(kind=kind,record=record));active=None
            report['processes']=records;write(output/'report.json',report)
            if (record.get('returncode')!=0 or record.get('error') or record.get('result',{}).get('passed')is not True
                    or any(record.get(n)is not True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))):
                raise ValueError('preflight child failed: '+kind)
        report.update(operation_complete=True,tests_passed=records[0]['record']['result']['tests_passed'],
            installed_denials=28,public_prerequisite_gates_passed=True)
    except BaseException as error:
        primary=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        with terminal_scope(report,output/'report.json',write,cleanup_signals,lambda:(primary,None)):
            errors=[]
            def attempt(label,fn):
                try:return fn()
                except BaseException as error:
                    errors.append(error);report.setdefault('cleanup_errors',[]).append(dict(operation=label,type=type(error).__name__,message=str(error)))
                    if primary is not None:primary.add_note(label+': '+repr(error))
            if active is not None:
                row=attempt('active child evidence',lambda:json.loads((active/'process.json').read_bytes()))
                if type(row)is dict:records.append(dict(kind='interrupted',record=row))
                else:records.append(dict(kind='unknown',record={}))
            after=attempt('source readback',lambda:protocol.hashes(protocol.sources()))
            verified=attempt('archive readback',lambda:collector.check_archive(output/'sources.tar.gz',before)) if before else False
            report.update(processes=records,source_hashes_after=after,source_files_unchanged=bool(before) and before==after,
                source_archive_verified=verified is True,elapsed_s=time.monotonic()-start,
                all_processes_reaped=all(all(r['record'].get(n)is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed')) for r in records))
            if before:report['source_archive_sha256']=attempt('archive digest',lambda:process.sha256(output/'sources.tar.gz'))
            report['cleanup_completed']=not errors and report['all_processes_reaped']
            report['passed']=bool(report.get('operation_complete') and primary is None and not errors and report['source_files_unchanged'] and verified and report['cleanup_completed'])
            try:write(output/'report.json',report)
            except BaseException as error:
                if primary is not None:primary.add_note('preflight final publication: '+repr(error))
                else:raise
            if primary is not None:raise primary
            if errors:raise next((e for e in errors if not isinstance(e,Exception)),errors[0])
    print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(dict,list))},allow_nan=False))
    return 0 if report['passed'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    with termination_signals():code=main(**vars(p.parse_args()))
    raise SystemExit(code)
