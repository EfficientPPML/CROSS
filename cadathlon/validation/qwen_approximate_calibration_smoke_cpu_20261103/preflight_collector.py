"""Seal the prepared collector before tests; no checkpoint model evaluation."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
P=Path(__file__).resolve().parent
names=['preflight_collector.py','check_approximate.py','arithmetic.py','numeric.py','native_capture.py','reference.py','test_candidate.py','test_arithmetic.py','test_native_capture.py','test_approximate.py','protocol.json','model_pins.json','corpus.json','tokenized_calibration_inputs.json','candidate-v1/manifest.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    import transformers
    package=Path(transformers.__file__).parent
    software=[*sorted((package/'models/qwen3').glob('*.py')),*[package/n for n in ('activations.py','cache_utils.py','modeling_attn_mask_utils.py','modeling_rope_utils.py','modeling_utils.py','modeling_outputs.py','configuration_utils.py','tokenization_utils_base.py','tokenization_utils_fast.py')]]
    manifest=json.loads((P/'candidate-v1/manifest.json').read_text())
    all_names=list(dict.fromkeys(names+list(manifest['source_input_sha256'])))
    paths={n:P/n for n in all_names};paths.update({'software/transformers/'+str(f.relative_to(package)):f for f in software})
    out=P/'collector-preflight-v2';out.mkdir(exist_ok=False);start=time.perf_counter();before={n:sha(f) for n,f in paths.items()}
    with tarfile.open(out/'sources.tar.gz','w:gz') as t:
        for n,f in paths.items():b=f.read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
    with (out/'tests.log').open('w') as f:r=subprocess.run([sys.executable,'-m','pytest','-q','test_candidate.py','test_arithmetic.py','test_native_capture.py','test_approximate.py'],cwd=P,stdout=f,stderr=subprocess.STDOUT)
    after={n:sha(f) for n,f in paths.items()}
    report={'passed':r.returncode==0 and before==after,'test_exit_code':r.returncode,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,'source_archive_sha256':sha(out/'sources.tar.gz'),'elapsed_s':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),'checkpoint_model_evaluated':False,'approximate_full_model_evaluated':False,'scope':'Analytic/symbolic tests, tiny native models, strict lifecycle and failure contracts only.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'source_count':len(before),'source_archive_sha256':report['source_archive_sha256'],'elapsed_s':report['elapsed_s']}));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
