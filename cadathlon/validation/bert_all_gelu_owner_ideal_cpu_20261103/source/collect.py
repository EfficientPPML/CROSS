"""Freeze and check all12 ideal GELU owners; no backend or model inference."""
from pathlib import Path
import argparse,hashlib,io,json,os,shutil,signal,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main(output):
    output.mkdir(exist_ok=False);snapshot=output/'snapshot';snapshot.mkdir();start=time.monotonic()
    paths=list((HERE/'checkout').rglob('*.py'))+list(HERE.glob('*.py'))
    paths+=[HERE/n for n in ('candidate_manifest.json','p60_profile_and_refresh.json','origin.json')]
    paths={str(p.relative_to(HERE)):p for p in sorted(paths)}
    before={n:sha(p) for n,p in paths.items()};report=dict(passed=False,scope=__doc__,encrypted_execution=False,
        source_hashes_before=before,cpu_affinity=sorted(os.sched_getaffinity(0)),cases_expected=72)
    process=None;primary=None
    try:
        for name,path in paths.items():
            target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        assert {n:sha(snapshot/n) for n in paths}==before
        with tarfile.open(output/'sources.tar.gz','w:gz') as t:
            for name in paths:
                b=(snapshot/name).read_bytes();i=tarfile.TarInfo(name);i.size=len(b);t.addfile(i,io.BytesIO(b))
        with tarfile.open(output/'sources.tar.gz') as t:
            ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(paths)
            assert all(m.isfile() for m in ms)
            assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==before
        report['source_archive_sha256']=sha(output/'sources.tar.gz');save(output/'report.json',report)
        case_dir=output/'cases';case_dir.mkdir();env=os.environ.copy();env['GELU_CASE_REPORT_DIR']=str(case_dir)
        with (output/'tests.log').open('w') as log:
            command=[sys.executable,'-m','pytest','-q','test_all_gelu.py','test_gelu64.py']
            report['command']=command
            process=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report['child_pid']=process.pid;save(output/'report.json',report);report['returncode']=process.wait()
        cases=[json.loads(p.read_text()) for p in sorted(case_dir.glob('*.json'))]
        assert report['returncode']==0 and len(cases)==72
        assert all(c['physical_slots_checked']==65536 and c['max_abs_vs_independent_chebval']<c['gate']==1e-10 for c in cases)
        assert {c['layer'] for c in cases}==set(range(12))
        for layer in range(12):
            rows=[c for c in cases if c['layer']==layer]
            assert len(rows)==6 and {c['level_choice'] for c in rows}=={'minimum',24,49}
        report.update(passed=True,cases=cases,physical_slots_checked=72*65536,
            max_abs_vs_independent_chebval=max(c['max_abs_vs_independent_chebval'] for c in cases),
            versions={'python':sys.version})
    except BaseException as error:
        primary=error;report['error']=dict(type=type(error).__name__,message=str(error));report['passed']=False
    finally:
        try:
            if process is not None and process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            report['child_reaped']=process is None or process.poll() is not None
            after={n:sha(p) for n,p in paths.items()};report['source_hashes_after']=after
            report['source_files_unchanged']=after==before
            report['snapshot_unchanged']={n:sha(snapshot/n) for n in paths}==before
            report['passed']=report['passed'] and report['source_files_unchanged'] and report['snapshot_unchanged']
        except BaseException as error:
            report['passed']=False
            if primary is None:primary=error
            else:primary.add_note('Cleanup/readback: '+repr(error))
        report['elapsed_s']=time.monotonic()-start;save(output/'report.json',report)
    if primary is not None:raise primary
    print(json.dumps({k:report[k] for k in ('passed','elapsed_s','max_abs_vs_independent_chebval','source_archive_sha256')}))
    return 0 if report['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();raise SystemExit(main(a.output))
