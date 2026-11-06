"""Source-bound candidate3 collector tests; no checkpoint/native model evaluation."""
from pathlib import Path
import hashlib,importlib.util,io,json,os,subprocess,sys,tarfile,time
import candidate_gate
P=Path(__file__).resolve().parent
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()

def paths():
    req=json.loads((P/'collector_requirement.json').read_text())
    names={p.name for p in P.glob('*.py')}|{'collector_requirement.json','origin.json','candidate-v3/manifest.json'}
    for group in ('selection_artifact_sha256','calibration_input_sha256','runtime_dependency_sha256'):names.update(req[group])
    names.update('selection/'+n for n in req['selection_input_sha256'])
    out={n:P/n for n in sorted(names)}
    package=Path(importlib.util.find_spec('transformers').origin).parent
    software=[*sorted((package/'models/qwen3').glob('*.py')),*[package/n for n in ('activations.py','cache_utils.py','modeling_attn_mask_utils.py','modeling_rope_utils.py','modeling_utils.py','modeling_outputs.py','configuration_utils.py','tokenization_utils_base.py','tokenization_utils_fast.py')]]
    out.update({'software/transformers/'+str(f.relative_to(package)):f for f in software})
    return out

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(exist_ok=False);start=time.perf_counter();p=paths();before={n:sha(f) for n,f in p.items()}
    with tarfile.open(args.output/'sources.tar.gz','w:gz') as archive:
        for n,f in p.items():
            b=f.read_bytes();m=tarfile.TarInfo(n);m.size=len(b);archive.addfile(m,io.BytesIO(b))
    candidate_gate.verify_archive(args.output/'sources.tar.gz',before)
    prerequisite=candidate_gate.verify(P)
    command=[sys.executable,'-m','pytest','-q','test_prerequisites.py','test_collector.py','test_arithmetic.py','test_native_capture.py']
    with (args.output/'tests.log').open('w') as log:run=subprocess.run(command,cwd=P,stdout=log,stderr=subprocess.STDOUT)
    after={n:sha(f) for n,f in p.items()};report={'passed':run.returncode==0 and before==after,'test_exit_code':run.returncode,
        'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,
        'source_archive_sha256':sha(args.output/'sources.tar.gz'),'selection_prerequisite':prerequisite,
        'elapsed_s':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),'command':command,
        'checkpoint_model_evaluated':False,'validation_outputs_read':False,'scope':'Prerequisite substitution, lifecycle, public arithmetic/depth and tiny native-capture tests only; no checkpoint model execution.'}
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ('passed','test_exit_code','source_archive_sha256','elapsed_s')}),flush=True)
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
