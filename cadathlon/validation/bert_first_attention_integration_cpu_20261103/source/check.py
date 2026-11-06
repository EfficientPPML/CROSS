"""Source-bound public-only first-attention planning checkpoint."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile,time,os
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def inventory():
    paths=list((HERE/'inputs').rglob('*.py'))+list((HERE/'inputs').rglob('*.json'))+list(HERE.glob('*.py'))
    paths += [HERE/name for name in ('README.md','source_origin.json','matrix_inventory.json','filesystem_observation.json')]
    return {str(p.relative_to(HERE)):sha(p) for p in sorted(paths)}
def main():
    start=time.monotonic();out=HERE/'source-bound-v1';out.mkdir(exist_ok=False);before=inventory()
    with tarfile.open(out/'sources.tar.gz','w:gz') as archive:
        for name in before:
            data=(HERE/name).read_bytes();entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
    with tarfile.open(out/'sources.tar.gz') as archive:
        members=archive.getmembers()
        assert len(members)==len(before) and {m.name for m in members}==set(before)
        assert all(m.isfile() and hashlib.sha256(archive.extractfile(m).read()).hexdigest()==before[m.name] for m in members)
    with (out/'tests.log').open('w') as log:
        tests=subprocess.run([sys.executable,'-m','pytest','-q','test_plan.py','--junitxml='+str(out/'tests.xml')],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
    with (out/'plan.log').open('w') as log:
        public=subprocess.run([sys.executable,'plan.py'],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
    after=inventory();r=dict(passed=tests.returncode==public.returncode==0 and before==after,
        scope='Public first-attention integration metadata, counts and interfaces only; no model forward/JAX/HE.',
        tests_exit_code=tests.returncode,plan_exit_code=public.returncode,elapsed_s=time.monotonic()-start,
        python=sys.version,cpu_affinity=sorted(os.sched_getaffinity(0)),source_hashes_before=before,source_hashes_after=after,
        source_files_unchanged=before==after,source_archive_verified=True,source_archive_sha256=sha(out/'sources.tar.gz'),
        plan_report_sha256=sha(HERE/'report.json'))
    (out/'validation.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:r[k] for k in ('passed','tests_exit_code','plan_exit_code','elapsed_s','source_files_unchanged','source_archive_sha256')},indent=2))
    return 0 if r['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
