"""Exact-source ideal attention integration collection; never HE execution."""
from pathlib import Path
import argparse,io,json,math,os,sys,tarfile,time,xml.etree.ElementTree as ET
import export_decoder as public_io
from component_processes import OwnedChild
from cleanup import cleanup_signals
from terminal import terminal_scope
from finish import finish
HERE=Path(__file__).resolve().parent
NAMES=('README.md','collect_join.py','source_origins.json','original_attention_tile.py',
 'encrypted_bert_bsgs_attention.py','encrypted_bsgs_score_routing.py',
 'encrypted_factored_score_routing.py','encrypted_shallow_score_routing.py',
 'encrypted_scale_planning.py','p60_profile_and_refresh.json','test_bsgs.py',
 'test_attention_join.py','export_decoder.py','component_processes.py','cleanup.py',
 'terminal.py','finish.py')

def sources():return {n:(HERE/n).read_bytes() for n in NAMES}
def hashes(raw):return {n:public_io.digest(b) for n,b in raw.items()}
def archive(path,raw):
    with tarfile.open(path,'w:gz') as t:
        for name,b in sorted(raw.items()):
            member=tarfile.TarInfo(name);member.size=len(b);t.addfile(member,io.BytesIO(b))
def verify_archive(path,expected):
    with tarfile.open(path) as t:
        members=t.getmembers()
        if len(members)!=len(expected) or len({m.name for m in members})!=len(expected) or not all(m.isfile() for m in members):
            raise ValueError('exact regular source archive required')
        if {m.name:public_io.digest(t.extractfile(m).read()) for m in members}!=expected:
            raise ValueError('archived source bytes differ')
    return True
def verify_metrics(rows):
    if len(rows)!=6:raise ValueError('all six independent numerical records required')
    expected={(op,L,s) for op in ('scores','values') for L,s in ((49,1.1467305003699405e18),(25,math.nextafter(1.1467305003699405e18,math.inf)),(25,1.1467305003699405e18*1.21))}
    if {(r['operation'],r['input_level'],r['input_scale']) for r in rows}!=expected:
        raise ValueError('complete canonical/noncanonical score/PV inventory required')
    for r in rows:
        if (r['physical_slots']!=65536 or r['gate']!=1e-8 or r['output_level']!=r['input_level']-3
                or not math.isfinite(r['max_abs_error']) or not 0<=r['max_abs_error']<=1e-8
                or not math.isfinite(r['output_scale']) or r['output_scale']<=0):
            raise ValueError('fixed finite full-slot numerical gates required')
        counts=r['counts']
        if any(counts.get(k)!=v for k,v in dict(ciphertext_products=128,head_rotations=768,row_alignment_rotations=120,
             accumulator_additions=127,route_rotations=1520,route_public_multiplications=3584,
             route_actual_composite_rescales=896).items()):raise ValueError('complete attention operation count differs')
    return dict(fixtures=6,physical_slots=393216,absolute_gate=1e-8,maximum_absolute_error=max(r['max_abs_error'] for r in rows))

def run(output):
    output=Path(output).absolute();output.mkdir(mode=0o700,exist_ok=False);start=time.monotonic()
    report=dict(passed=False,encrypted_execution=False,model_forward=False,wrapper_pid=os.getpid(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),termination_scope_completed=False,
        scope='Isolated full128-key attention loops with explicit NumPy payload/security doubles; no full model, HE or performance claim.')
    child=None;expected={};primary=None;cleanup_error=None
    try:
        public_io.write_json(output/'report.json',report);raw=sources();expected=hashes(raw)
        archive(output/'sources.tar.gz',raw);verify_archive(output/'sources.tar.gz',expected)
        checkout=output/'execution';checkout.mkdir()
        for name,b in raw.items():(checkout/name).write_bytes(b)
        report.update(source_hashes_before=expected,source_archive_sha256=public_io.file_sha(output/'sources.tar.gz'))
        command=[sys.executable,'-m','pytest','-q','--tb=short','test_attention_join.py','--junitxml='+str(output/'tests.xml')]
        env=dict(os.environ,JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        env.pop('PYTHONPATH',None)
        public_io.write_json(output/'report.json',report)
        with public_io.owned_stream(output/'tests.log',write=True) as log:
            child=OwnedChild(command,cwd=checkout,stdout=log,env=env)
            report['process']=child.record;public_io.write_json(output/'report.json',report)
            code=child.wait()
        if code!=0:raise RuntimeError('ideal attention suite failed')
        suites=list(ET.parse(output/'tests.xml').getroot().iter('testsuite'))
        counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
        if counts!={'tests':22,'errors':0,'failures':0,'skipped':0}:raise ValueError('complete22-test integration suite required')
        numeric=(checkout/'ideal_results.jsonl').read_bytes()
        rows=[json.loads(line) for line in numeric.splitlines()];summary=verify_metrics(rows)
        (output/'ideal_results.jsonl').write_bytes(numeric)
        report.update(tests=counts,numeric_summary=summary,numeric_sha256=public_io.digest(numeric),passed=True)
    except BaseException as error:
        primary=error;report.update(passed=False,error=dict(type=type(error).__name__,message=str(error)))
    finally:
        with terminal_scope(report,output/'report.json',public_io.write_json,cleanup_signals,lambda:(primary,cleanup_error)):
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
            copied=attempt('execution source readback',lambda:{n:public_io.file_sha(output/'execution'/n) for n in expected}) if expected else {}
            report.update(source_hashes_after=after,execution_source_hashes_after=copied,
                source_archive_verified=bool(expected) and attempt('source archive',lambda:verify_archive(output/'sources.tar.gz',expected))is True,
                sources_unchanged=bool(expected) and expected==after==copied,elapsed_s=time.monotonic()-start)
            complete=child is not None and all(child.record.get(n)is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
            report['all_processes_reaped_and_owned_group_empty']=complete
            report['passed']=bool(report['passed'] and complete and report['sources_unchanged'] and report['source_archive_verified'] and not errors)
            report['cleanup_errors']=[dict(operation=label,type=type(error).__name__) for label,error in errors]
            attempt('final report',lambda:public_io.write_json(output/'report.json',report))
            if primary is not None:
                for label,error in errors:primary.add_note(label+': '+type(error).__name__)
                raise primary
            if cleanup_error is not None:raise cleanup_error
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=finish(args.output/'report.json',lambda:run(args.output),public_io.write_json,public_io.public_report)
    print(json.dumps({k:result[k] for k in ('passed','tests','numeric_summary','elapsed_s')},allow_nan=False))
    raise SystemExit(0 if result['passed']else 1)
