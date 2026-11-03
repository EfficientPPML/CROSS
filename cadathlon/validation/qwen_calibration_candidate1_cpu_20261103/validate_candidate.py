"""Source-bound public grid, coefficient and symbolic-graph checks only."""
from pathlib import Path
import hashlib,io,json,subprocess,sys,tarfile,time
P=Path(__file__).resolve().parent
names=['numeric.py','select_candidate.py','arithmetic.py','test_candidate.py','test_arithmetic.py','validate_candidate.py','protocol.json','selection_inputs.json','input_origin.json','reciprocal_schedule.json','candidate-v1/manifest.json','candidate-v1/assessments.json']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=P/'candidate-validation';out.mkdir(exist_ok=False);start=time.perf_counter();before={n:sha(P/n) for n in names}
    with tarfile.open(out/'sources.tar.gz','w:gz') as a:
        for n in names:
            b=(P/n).read_bytes();i=tarfile.TarInfo(n);i.size=len(b);a.addfile(i,io.BytesIO(b))
    with (out/'tests.log').open('w') as f:r=subprocess.run([sys.executable,'-m','pytest','-q','test_candidate.py','test_arithmetic.py'],cwd=P,stdout=f,stderr=subprocess.STDOUT)
    after={n:sha(P/n) for n in names};report={'passed':r.returncode==0 and before==after,'test_exit_code':r.returncode,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,'source_archive_sha256':sha(out/'sources.tar.gz'),'elapsed_s':time.perf_counter()-start,'full_model_evaluation':False,'scope':'Analytic grids, exact rounded initial certificates, strict domain failures, explicit primitive parity/depth and zero-payload symbolic full graph only.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
