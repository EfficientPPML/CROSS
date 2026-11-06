"""Source-bound public metadata proof, without keys, ciphertexts or backend."""
from pathlib import Path
import argparse,io,json,os,sys,tarfile,time,xml.etree.ElementTree as ET
import export_decoder as e
from component_processes import OwnedChild
from cleanup import cleanup_signals
from terminal import terminal_scope
from finish import finish
HERE=Path(__file__).resolve().parent

def sources(root=HERE):
    root=Path(root);names=json.loads((root/'source_inventory.json').read_bytes())
    if type(names)is not list or names!=sorted(set(names)) or 'source_inventory.json'not in names:raise ValueError('fixed unique source inventory required')
    if any(type(n)is not str or Path(n).is_absolute()or '..'in Path(n).parts for n in names):raise ValueError('relative source names required')
    return {n:(root/n).read_bytes()for n in names}

def hashes(raw):return {n:e.digest(b)for n,b in raw.items()}

def archive(path,raw):
    with tarfile.open(path,'w:gz')as t:
        for n,b in sorted(raw.items()):
            item=tarfile.TarInfo(n);item.size=len(b);t.addfile(item,io.BytesIO(b))

def archive_check(path,expected):
    with tarfile.open(path)as t:
        rows=t.getmembers()
        if len(rows)!=len(expected)or len({m.name for m in rows})!=len(rows)or any(not m.isfile()or m.size>8<<20 for m in rows)or sum(m.size for m in rows)>32<<20:raise ValueError('bounded regular source archive required')
        if {m.name:e.digest(t.extractfile(m).read())for m in rows}!=expected:raise ValueError('source archive differs')
    return True

def public_gate(raw):
    pins=json.loads(raw['predecessor_pins.json']);origin=json.loads(raw['provenance.json'])
    if len(pins)!=8 or pins!=origin['eight_predecessor_input_pins']:raise ValueError('eight unchanged predecessor inputs required')
    expected=dict(pins)|origin['measured_numerical_sources_before_collector']|{n:r['sha256']for n,r in origin['helpers'].items()}
    for n,h in expected.items():
        if e.digest(raw[n])!=h:raise ValueError('frozen public source differs: '+n)
    return True

def plan_gate(plan):
    rows=plan.get('schedules')
    if plan.get('passed')is not True or type(rows)is not list or len(rows)!=512:raise ValueError('all512 schedules required')
    if {(r['K'],r['gain'])for r in rows}!={(K,g)for K in range(1,129)for g in (1,2,4,8)}:raise ValueError('fixed public schedule inventory differs')
    if any(r.get('real_polynomial_unchanged')is not True or r.get('metadata_relabeling')is not False for r in rows):raise ValueError('unchanged polynomial/no relabeling required')
    if len(plan.get('sensitivity',[]))!=16 or not plan.get('limitations'):raise ValueError('conditional sensitivity and limits required')
    return True

def run(output):
    output=Path(output).absolute();output.mkdir(mode=0o700,exist_ok=False);started=time.monotonic()
    report=dict(passed=False,complete=False,scope=__doc__,he=False,model_forward=False,private_payloads_read=False,wrapper_pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),termination_scope_completed=False,processes=[])
    children=[];expected={};primary=None;cleanup_error=None
    try:
        e.write_json(output/'report.json',report)
        raw=sources();public_gate(raw);expected=hashes(raw)
        archive(output/'sources.tar.gz',raw);archive_check(output/'sources.tar.gz',expected)
        snapshot=output/'execution';snapshot.mkdir()
        for n,b in raw.items():
            p=snapshot/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        env=dict(os.environ,PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4');env.pop('PYTHONPATH',None)
        report.update(source_count=len(expected),source_hashes_before=expected,source_archive_sha256=e.file_sha(output/'sources.tar.gz'))
        commands=[('tests',[sys.executable,'-m','pytest','-q','test_precision_plan.py','--junitxml='+str(output/'tests.xml')],'tests.log'),('plan',[sys.executable,str(snapshot/'precision_plan.py')],'plan.json')]
        for name,command,filename in commands:
            with e.owned_stream(output/filename,write=True)as stream:
                child=OwnedChild(command,cwd=snapshot,stdout=stream,env=env);children.append(child);report['processes'].append(dict(name=name,record=child.record));e.write_json(output/'report.json',report)
                if child.wait()!=0:raise RuntimeError(name+' child failed')
        suites=list(ET.parse(output/'tests.xml').getroot().iter('testsuite'))
        counts={n:sum(int(s.attrib.get(n,0))for s in suites)for n in ('tests','errors','failures','skipped')};report['tests']=counts
        if counts!={'tests':520,'errors':0,'failures':0,'skipped':0}:raise ValueError('exact520 checks required')
        plan=json.loads(e.public_report(output/'plan.json'));plan_gate(plan)
        report.update(plan_sha256=e.file_sha(output/'plan.json'),schedules=512,sensitivity_rows=16,passed=True,complete=True)
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
                    if cleanup_error is None or not isinstance(error,Exception)and isinstance(cleanup_error,Exception):cleanup_error=error
            for child in children:attempt('owned child cleanup',lambda child=child:child.close(primary=primary))
            after=attempt('current sources',lambda:hashes(sources()))
            copied=attempt('execution sources',lambda:{n:e.file_sha(output/'execution'/n)for n in expected})if expected else{}
            archive_ok=attempt('source archive',lambda:archive_check(output/'sources.tar.gz',expected))if expected else False
            attempt('public provenance readback',lambda:public_gate(sources()))
            plan_hash=attempt('public plan digest',lambda:e.file_sha(output/'plan.json'))if report.get('plan_sha256')else None
            if report.get('plan_sha256'):attempt('public plan readback',lambda:plan_gate(json.loads(e.public_report(output/'plan.json'))))
            process_ok=len(children)==2 and all(all(c.record.get(n)is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))for c in children)
            report.update(source_hashes_after=after,execution_source_hashes_after=copied,sources_unchanged=bool(expected)and expected==after==copied,source_archive_verified=archive_ok is True,all_direct_children_reaped=bool(children)and all(c.record.get('direct_child_reaped')is True for c in children),all_owned_groups_empty=bool(children)and all(c.record.get('group_empty_verified')is True for c in children),cleanup_completed=not errors and all(c.record.get('cleanup_completed')is True for c in children),cleanup_errors=[dict(operation=n,type=type(v).__name__)for n,v in errors],elapsed_s=time.monotonic()-started)
            report['plan_file_unchanged']=bool(plan_hash)and plan_hash==report.get('plan_sha256')
            report['passed']=bool(report['passed']and report['complete']and process_ok and report['sources_unchanged']and report['source_archive_verified']and report['plan_file_unchanged']and not errors)
            attempt('final report',lambda:e.write_json(output/'report.json',report))
            if primary is not None:
                for name,error in errors:primary.add_note(name+': '+type(error).__name__)
                raise primary
            if cleanup_error is not None:raise cleanup_error
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=finish(args.output/'report.json',lambda:run(args.output),e.write_json,e.public_report)
    print(json.dumps({n:result[n]for n in ('passed','tests','schedules','source_count','elapsed_s')}))
    raise SystemExit(0 if result['passed']else 1)
