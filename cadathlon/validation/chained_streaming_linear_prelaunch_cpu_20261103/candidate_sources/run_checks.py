"""Source-bound CPU ideal/metadata tests only; no key generation or encryption."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import check_component as driver

def main():
    output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=False)
    raw=driver.sources();before={name:driver.sha(data) for name,data in raw.items()}
    with tarfile.open(output/'sources.tar.gz','w:gz') as tar:
        for name,data in raw.items():
            entry=tarfile.TarInfo(name);entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
    command=[sys.executable,'-m','pytest','-q',str(HERE/'test_chained.py'),str(HERE/'test_streaming_ideal.py'),str(HERE/'test_driver.py'),
        *(str(HERE/'checkout/cadathlon/tests'/name) for name in ('test_packed_transformer.py','test_streaming_linear.py','test_streaming_linear_contracts.py','test_packed_qwen.py')),
        '--junitxml='+str(output/'junit.xml')]
    start=time.perf_counter()
    with (output/'run.log').open('w') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,cwd=HERE)
    after={name:driver.sha(data) for name,data in driver.sources().items()}
    with tarfile.open(output/'sources.tar.gz') as tar:
        verified={m.name:driver.sha(tar.extractfile(m).read()) for m in tar.getmembers()}==before
    report=dict(scope=__doc__,passed=result.returncode==0 and before==after and verified,
        command=command,returncode=result.returncode,source_hashes_before=before,source_hashes_after=after,
        source_files_unchanged=before==after,source_files=len(before),source_archive_verified=verified,
        source_archive_sha256=driver.sha((output/'sources.tar.gz').read_bytes()),
        elapsed_s=time.perf_counter()-start,cpu_affinity=sorted(os.sched_getaffinity(0)),
        environment={k:os.environ.get(k) for k in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        key_generation_executed=False,ciphertext_evaluation_executed=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','returncode','source_files','source_files_unchanged','source_archive_verified','elapsed_s')}))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
