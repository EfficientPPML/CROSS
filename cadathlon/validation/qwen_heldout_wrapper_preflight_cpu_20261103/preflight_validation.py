"""Source-bound wrapper tests; do not open held-out inputs or checkpoint weights."""
from pathlib import Path
import hashlib,importlib.util,io,json,os,subprocess,sys,tarfile,time
P=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    req=json.loads((P/'calibration_requirement.json').read_text())
    package=Path(importlib.util.find_spec('transformers').origin).parent
    paths={n:(package/n.removeprefix('software/transformers/') if n.startswith('software/transformers/') else P/n) for n in req['expected_runtime_source_hashes']}
    for name in ('validate_heldout.py','calibration_gate.py','calibration_requirement.json','test_validation_gate.py','preflight_validation.py'):paths[name]=P/name
    out=P/'wrapper-preflight-v2';out.mkdir(exist_ok=False);start=time.perf_counter()
    before={n:sha(p) for n,p in paths.items()}
    with tarfile.open(out/'sources.tar.gz','w:gz') as t:
        for n,p in paths.items():
            b=p.read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
    with tarfile.open(out/'sources.tar.gz') as t:
        members=t.getmembers()
        assert len(members)==len(before)==len({m.name for m in members})
        assert all(m.isfile() for m in members)
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==before
    command=[sys.executable,'-m','pytest','-q','test_candidate.py','test_arithmetic.py','test_native_capture.py','test_approximate.py','test_validation_gate.py']
    with (out/'tests.log').open('w') as f:proc=subprocess.run(command,cwd=P,stdout=f,stderr=subprocess.STDOUT)
    after={n:sha(p) for n,p in paths.items()}
    unchanged_runtime=all(before[n]==h for n,h in req['expected_runtime_source_hashes'].items())
    report={'passed':proc.returncode==0 and before==after and unchanged_runtime,'test_exit_code':proc.returncode,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,'calibrated_runtime_sources_identical':unchanged_runtime,'source_archive_sha256':sha(out/'sources.tar.gz'),'elapsed_s':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),'command':command,'validation_inputs_read':False,'checkpoint_model_evaluated':False,'approximate_full_model_evaluated':False,'scope':'Public/analytic/symbolic tests, tiny native-model hooks, completed-calibration prerequisite and strict validation wrapper failure tests only.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','test_exit_code','calibrated_runtime_sources_identical','source_archive_sha256','elapsed_s')}),flush=True)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
