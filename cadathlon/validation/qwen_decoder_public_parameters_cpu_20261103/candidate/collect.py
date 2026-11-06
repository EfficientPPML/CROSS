"""Frozen-source child execution and public preparation evidence."""
from pathlib import Path
import argparse,hashlib,io,json,os,shutil,sys,tarfile,time,xml.etree.ElementTree as ET
import export_decoder as e
from component_processes import OwnedChild
from cleanup import termination_signals,cleanup_signals
from terminal import terminal_scope
HERE=Path(__file__).resolve().parent
NAMES=('README.md','export_decoder.py','collect.py','test_export_decoder.py','cleanup.py',
       'component_processes.py','terminal.py','finish.py','inputs/config.json','inputs/candidate_manifest.json')

def sources():return {n:(HERE/n).read_bytes() for n in NAMES}
def hashes(raw):return {n:e.digest(b) for n,b in raw.items()}
def archive(path,raw):
    with tarfile.open(path,'w:gz') as t:
        for name,b in sorted(raw.items()):
            m=tarfile.TarInfo(name);m.size=len(b);t.addfile(m,io.BytesIO(b))
def verify_archive(path,expected):
    with tarfile.open(path) as t:
        ms=t.getmembers()
        if len(ms)!=len({m.name for m in ms}) or len(ms)!=len(expected):raise ValueError('source archive member count differs')
        if len(ms)!=len(expected) or not all(m.isfile() for m in ms):raise ValueError('exact regular source archive required')
        if {m.name:e.digest(t.extractfile(m).read()) for m in ms}!=expected:raise ValueError('source archive bytes differ')
    return True
def run(output,execute):
    output=Path(output).absolute();output.mkdir(mode=0o700,exist_ok=False);start=time.monotonic()
    report=dict(passed=False,execute=execute,actual_execution_attempted=False,wrapper_pid=os.getpid(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),termination_scope_completed=False,
        scope='Public decoder parameter export' if execute else 'Synthetic public-format and cleanup tests',
        model_forward=False,he=False,process=None)
    child=None;expected={};primary=None;cleanup_error=None
    try:
        e.write_json(output/'report.json',report);raw=sources();expected=hashes(raw)
        archive(output/'sources.tar.gz',raw);verify_archive(output/'sources.tar.gz',expected)
        checkout=output/'execution';checkout.mkdir()
        for name,b in raw.items():
            p=checkout/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        report.update(source_hashes_before=expected,source_archive_sha256=e.file_sha(output/'sources.tar.gz'))
        if execute:
            # Exactly113 physical matrices; neither old exports nor other work is deleted.
            required=sum(rows*cols*8+4096 for n,(rows,cols) in e.inventory().items() if n!='model.embed_tokens.weight')+(1<<30)
            if shutil.disk_usage(output).free<required:raise OSError('insufficient storage for all public matrices plus reserve')
            command=[sys.executable,str(checkout/'export_decoder.py'),'--output',str(output/'worker'),'--body',str(output/'public_matrices')]
            report['actual_execution_attempted']=True;report['required_storage_with_reserve']=required
        else:command=[sys.executable,'-m','pytest','-q','test_export_decoder.py','--junitxml='+str(output/'tests.xml')]
        e.write_json(output/'report.json',report)
        with e.owned_stream(output/'process.log',write=True) as log:
            child=OwnedChild(command,cwd=checkout,stdout=log)
            report['process']=child.record;e.write_json(output/'report.json',report)
            code=child.wait()
        if code!=0:raise RuntimeError('parameter worker/test child failed')
        if execute:
            result=json.loads((output/'worker/report.json').read_bytes())
            if any(result.get(k)is not True for k in ('passed','operation_complete','termination_scope_completed')):
                raise ValueError('complete public export and termination proof required')
            report.update(worker_report_sha256=e.file_sha(output/'worker/report.json'),
                parameter_manifest_sha256=result['parameter_manifest_sha256'],named_matrix_count=result['named_matrix_count'],physical_matrix_count=result['physical_matrix_count'])
        else:
            suites=ET.parse(output/'tests.xml').getroot().iter('testsuite')
            rows=list(suites);counts={k:sum(int(s.attrib.get(k,0)) for s in rows) for k in ('tests','errors','failures','skipped')}
            if counts['tests']<1 or any(counts[n] for n in ('errors','failures','skipped')):raise ValueError('complete synthetic suite must pass')
            report['tests']=counts
        report['passed']=True
    except BaseException as error:
        primary=error;report.update(passed=False,error=dict(type=type(error).__name__,message=str(error)))
    finally:
        with terminal_scope(report,output/'report.json',e.write_json,cleanup_signals,lambda:(primary,cleanup_error)):
            errors=[]
            def attempt(label,fn):
                nonlocal cleanup_error
                try:return fn()
                except BaseException as error:
                    errors.append((label,error))
                    if cleanup_error is None or not isinstance(error,Exception) and isinstance(cleanup_error,Exception):cleanup_error=error
                    return None
            if child is not None:attempt('owned child cleanup',lambda:child.close(primary=primary))
            after=attempt('current source readback',lambda:hashes(sources()))
            copied=attempt('execution source readback',lambda:{n:e.file_sha(output/'execution'/n) for n in expected}) if expected else {}
            report.update(source_hashes_after=after,execution_source_hashes_after=copied,
                source_archive_verified=bool(expected) and attempt('source archive',lambda:verify_archive(output/'sources.tar.gz',expected))is True,
                sources_unchanged=bool(expected) and expected==after==copied,elapsed_s=time.monotonic()-start)
            complete=child is not None and all(child.record.get(n)is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
            report['all_processes_reaped_and_owned_group_empty']=complete
            report['passed']=bool(report['passed'] and complete and report['sources_unchanged'] and report['source_archive_verified'] and not errors)
            report['cleanup_errors']=[dict(operation=label,type=type(error).__name__) for label,error in errors]
            attempt('final report',lambda:e.write_json(output/'report.json',report))
            if primary is not None:
                for label,error in errors:primary.add_note(label+': '+type(error).__name__)
                raise primary
            if cleanup_error is not None:raise cleanup_error
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
    from finish import finish
    result=finish(a.output/'report.json',lambda:run(a.output,a.execute),e.write_json,e.public_report)
    print(json.dumps({k:v for k,v in result.items() if k in ('passed','tests','named_matrix_count','physical_matrix_count','elapsed_s')}))
    raise SystemExit(0 if result['passed']else 1)
