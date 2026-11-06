"""Source-bound public-model cache validation; no client inputs or HE backend."""
from pathlib import Path
import argparse,hashlib,io,json,os,shutil,signal,subprocess,sys,tarfile,time
from server_public_model import open_owned_directory,remove_owned_directory
HERE=Path(__file__).resolve().parent
FILES=('server_public_model.py','test_server_public_model.py','collect_server_model.py','server_tensor_pins.json','server_source_origin.json','server_inputs/candidate_manifest.json','server_inputs/p60_profile_and_refresh.json')
SNAPSHOT=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c')
ASSETS={'model.safetensors':'e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034','config.json':'ec2b6cd6ce741dda6c972aea96f9d1a449cbb4b5349e688c5370a9b2f6872d30'}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def main(out):
    out.mkdir(exist_ok=False);snapshot=out/'snapshot';snapshot.mkdir();start=time.monotonic();child=None;primary=None;work_fd=None;work_identity=None
    before={n:sha(HERE/n) for n in FILES};report=dict(passed=False,scope=__doc__,source_hashes_before=before,HE_executed=False,model_forward=False,cpu_affinity=sorted(os.sched_getaffinity(0)))
    try:
        for n in FILES:
            target=snapshot/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/n,target)
        with tarfile.open(out/'sources.tar.gz','w:gz') as t:
            for n in FILES:
                raw=(snapshot/n).read_bytes();info=tarfile.TarInfo(n);info.size=len(raw);t.addfile(info,io.BytesIO(raw))
        with tarfile.open(out/'sources.tar.gz') as t:
            ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==7
            assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==before
        assets={n:sha(SNAPSHOT/n) for n in ASSETS};assert assets==ASSETS
        report.update(source_archive_sha256=sha(out/'sources.tar.gz'),asset_hashes_before=assets)
        work=out/'test-work';work.mkdir();work_fd,work_identity=open_owned_directory(work)
        env=os.environ.copy();env['SERVER_MODEL_CASE_REPORT']=str(out/'case.json')
        command=[sys.executable,'-m','pytest','-q','test_server_public_model.py','--tb=short','--basetemp='+str(out/'test-work/pytest')]
        with (out/'tests.log').open('w') as log:
            child=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report.update(child_pid=child.pid,command=command);save(out/'report.json',report);report['returncode']=child.wait()
        assert report['returncode']==0
        case=json.loads((out/'case.json').read_text());assert case['passed'] and case['tensors']==196 and case['matrices']==74
        report.update(passed=True,case=case)
    except BaseException as error:primary=error;report.update(passed=False,error=dict(type=type(error).__name__,message=str(error)))
    finally:
        def attempt(label,fn):
            nonlocal primary
            try:return fn()
            except BaseException as error:
                report['passed']=False;report.setdefault('cleanup_errors',[]).append(dict(action=label,type=type(error).__name__,message=str(error)))
                if primary is None:primary=error
                else:primary.add_note(label+': '+repr(error))
        def reap():
            if child is not None and child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            return child is None or child.poll() is not None
        report['child_reaped']=attempt('reap child',reap)
        report['source_hashes_after']=attempt('source readback',lambda:{n:sha(HERE/n) for n in FILES})
        report['snapshot_hashes_after']=attempt('executed snapshot readback',lambda:{n:sha(snapshot/n) for n in FILES})
        report['asset_hashes_after']=attempt('asset readback',lambda:{n:sha(SNAPSHOT/n) for n in ASSETS})
        # Test-owned temporary public tensors are excluded from the evidence.
        if work_fd is not None:
            attempt('remove temporary test tensors',lambda:remove_owned_directory(out/'test-work',work_fd,work_identity))
            attempt('close owned test-work descriptor',lambda:os.close(work_fd))
        report['temporary_public_arrays_removed']=not (out/'test-work').exists()
        report['source_files_unchanged']=before==report['source_hashes_after']==report['snapshot_hashes_after']
        report['assets_unchanged']=report['asset_hashes_after']==ASSETS
        report['passed']=bool(report['passed'] and report['child_reaped'] and report['source_files_unchanged'] and report['assets_unchanged'] and report['temporary_public_arrays_removed'])
        report['elapsed_s']=time.monotonic()-start
        try:save(out/'report.json',report)
        except BaseException as error:
            if primary is None:raise
            primary.add_note('terminal report write: '+repr(error))
    if primary is not None:raise primary
    print(json.dumps({k:report[k] for k in ('passed','elapsed_s','source_archive_sha256')}));return 0 if report['passed'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();raise SystemExit(main(a.output))
