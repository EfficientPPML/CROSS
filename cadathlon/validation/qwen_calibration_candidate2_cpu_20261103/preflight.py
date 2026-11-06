from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=P/'selection-preflight-v2';out.mkdir(exist_ok=False);start=time.perf_counter()
    protocol=json.loads((P/'protocol.json').read_text());before={n:sha(P/n) for n in protocol['source_input_sha256']};assert before==protocol['source_input_sha256'];before['protocol.json']=sha(P/'protocol.json')
    with tarfile.open(out/'sources.tar.gz','w:gz') as t:
        for n in before:b=(P/n).read_bytes();i=tarfile.TarInfo(n);i.size=len(b);t.addfile(i,io.BytesIO(b))
    with (out/'tests.log').open('w') as f:r=subprocess.run([sys.executable,'-m','pytest','-q','test_selection_contract.py','test_domain_rule.py'],cwd=P,stdout=f,stderr=subprocess.STDOUT)
    after={n:sha(P/n) for n in before}
    report={'passed':r.returncode==0 and before==after,'test_exit_code':r.returncode,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,'source_archive_sha256':sha(out/'sources.tar.gz'),'elapsed_s':time.perf_counter()-start,'cpu_affinity':sorted(os.sched_getaffinity(0)),'model_evaluated':False,'coefficient_selection':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'passed':report['passed'],'sources':len(before),'source_archive_sha256':report['source_archive_sha256']}));return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
