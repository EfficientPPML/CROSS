"""Freeze and check all24 ideal row-normalization owners; no backend or model inference."""
from pathlib import Path
import argparse,hashlib,io,json,os,shutil,signal,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main(output):
    output.mkdir(exist_ok=False);snapshot=output/'snapshot';snapshot.mkdir();start=time.monotonic()
    paths=list((HERE/'checkout').rglob('*.py'))+list(HERE.glob('*.py'))
    paths+=[HERE/n for n in ('candidate_manifest.json','p60_profile_and_refresh.json','actual_profile_audit.json','affine_pins.json','source_origin.json')]
    paths={str(p.relative_to(HERE)):p for p in sorted(paths)}
    before={n:sha(p) for n,p in paths.items()};report=dict(passed=False,scope=__doc__,encrypted_execution=False,
        source_hashes_before=before,cpu_affinity=sorted(os.sched_getaffinity(0)),cases_expected=288)
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
        from test_all_row_norm import MODEL,MODEL_SHA,file_sha
        asset_before=file_sha(MODEL);assert asset_before==MODEL_SHA
        report['external_checkpoint_sha256_before']=asset_before
        case_dir=output/'cases';case_dir.mkdir();env=os.environ.copy();env['NORM_CASE_REPORT_DIR']=str(case_dir)
        with (output/'tests.log').open('w') as log:
            command=[sys.executable,'-m','pytest','-q','test_all_row_norm.py','--tb=short']
            report['command']=command
            process=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report['child_pid']=process.pid;save(output/'report.json',report);report['returncode']=process.wait()
        asset_after=file_sha(MODEL);report['external_checkpoint_sha256_after']=asset_after
        report['external_checkpoint_unchanged']=asset_before==asset_after
        assert report['external_checkpoint_unchanged']
        cases=[json.loads(p.read_text()) for p in sorted(case_dir.glob('*.json'))]
        assert report['returncode']==0 and len(cases)==288
        assert all(c['slots_checked']==65536 and c['max_abs_vs_independent_chebval']<c['gate']==1e-8 for c in cases)
        sites={f'{i}.{role}' for i in range(12) for role in ('attention','mlp')}
        assert {c['site'] for c in cases}==sites
        for site in sites:
            rows=[c for c in cases if c['site']==site]
            assert len(rows)==12 and {c['level_choice'] for c in rows}=={'minimum','normal'}
            assert {c['valid_rows'] for c in rows}=={1,16} and len({c['source_scale'] for c in rows})==3
        report.update(passed=True,cases=cases,physical_slots_checked=288*65536,
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
