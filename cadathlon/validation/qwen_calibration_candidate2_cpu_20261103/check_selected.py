from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=P/'selected-preflight-v1';out.mkdir(exist_ok=False);start=time.perf_counter()
    manifest=json.loads((P/'candidate-v2/manifest.json').read_text());names=list(manifest['source_input_sha256'])+['candidate-v2/manifest.json','candidate-v2/assessments.json','test_selected.py','check_selected.py']
    before={n:sha(P/n) for n in names}
    assert all(before[n]==h for n,h in manifest['source_input_sha256'].items())
    with tarfile.open(out/'sources.tar.gz','w:gz') as t:
        for n in before:b=(P/n).read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
    with (out/'tests.log').open('w') as f:result=subprocess.run([sys.executable,'-m','pytest','-q','test_domain_rule.py','test_selection_contract.py','test_selected.py'],cwd=P,stdout=f,stderr=subprocess.STDOUT)
    after={n:sha(P/n) for n in names}
    report={'passed':result.returncode==0 and before==after,'test_exit_code':result.returncode,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,'source_archive_sha256':sha(out/'sources.tar.gz'),'elapsed_s':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),'manifest_sha256':sha(P/'candidate-v2/manifest.json'),'model_evaluated':False,'scope':'All28 selected SiLU rules through the unchanged strict arithmetic on fixed full/central/endpoint analytic grids; source/certificate/frozen-rule tests. No model or validation evaluation.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'source_count':len(before),'source_archive_sha256':report['source_archive_sha256'],'elapsed_s':report['elapsed_s']}));return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
