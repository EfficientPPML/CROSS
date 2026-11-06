"""Source-bound validation collector tests; no checkpoint evaluation."""
from pathlib import Path
import hashlib,importlib.util,io,json,os,subprocess,sys,tarfile,time
import candidate_gate
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();args.output.mkdir(exist_ok=False)
    req=json.loads((P/'validation_requirement.json').read_text());package=Path(importlib.util.find_spec('transformers').origin).parent
    paths={n:(package/n.removeprefix('software/transformers/') if n.startswith('software/') else P/n) for n in req['expected_runtime_source_hashes']}
    for n in ('validate_heldout.py','calibration_gate.py','validation_gate.py','validation_requirement.json','validation_protocol.json','test_validation_gate.py','preflight_validation.py','validation_inputs/tokenized_fixtures.json','calibration_prerequisite/report.json','calibration_prerequisite/runtime_sources.tar.gz'):paths[n]=P/n
    start=time.perf_counter();before={n:sha(p) for n,p in paths.items()}
    with tarfile.open(args.output/'sources.tar.gz','w:gz') as archive:
        for n,p in paths.items():
            b=p.read_bytes();m=tarfile.TarInfo(n);m.size=len(b);archive.addfile(m,io.BytesIO(b))
    candidate_gate.verify_archive(args.output/'sources.tar.gz',before)
    command=[sys.executable,'-m','pytest','-q','test_validation_gate.py','test_collector.py']
    with (args.output/'tests.log').open('w') as log:result=subprocess.run(command,cwd=P,stdout=log,stderr=subprocess.STDOUT)
    after={n:sha(p) for n,p in paths.items()};same=all(before[n]==h for n,h in req['expected_runtime_source_hashes'].items())
    report={'passed':result.returncode==0 and before==after and same,'test_exit_code':result.returncode,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,
        'calibrated_runtime_sources_identical':same,'source_archive_sha256':sha(args.output/'sources.tar.gz'),'cpu_affinity':sorted(os.sched_getaffinity(0)),'elapsed_s':time.perf_counter()-start,'command':command,
        'validation_workload':json.loads((P/'validation_protocol.json').read_text())['workload'],'checkpoint_model_evaluated':False,'validation_outputs_read':False,'scope':'Wrapper prerequisite/lifecycle/input/metric tests only; validation input bytes checked, no model evaluation.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ('passed','test_exit_code','source_archive_sha256','elapsed_s')}),flush=True)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
