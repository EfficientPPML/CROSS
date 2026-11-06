"""Source-bound mocked checkpoint; does not construct or execute cryptography."""
from pathlib import Path
import hashlib,json,os,platform,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent
NAMES=('prepared_rotation.py','loader_prepared_rotation.py','key_contexts.py','client_keys.py',
       'test_prepared_rotation.py','test_prepared_loader.py','test_key_contexts.py','test_client_keys.py',
       'base/encrypted_rotation.py','base/encrypted_packed_qwen.py',
       'server_inputs/p60_profile_and_refresh.json','server_inputs/twelve_layer_plan.json','origins.json','check.py')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def main():
    output=HERE/'source-bound-v2';output.mkdir(exist_ok=False)
    before={name:sha((HERE/name).read_bytes()) for name in NAMES}
    with tarfile.open(output/'sources.tar.gz','w:gz') as archive:
        for name in NAMES:archive.add(HERE/name,arcname=name,recursive=False)
    command=[sys.executable,'-m','pytest','-q','test_key_contexts.py','test_prepared_rotation.py','test_prepared_loader.py','--junitxml='+str(output/'tests.xml')]
    env=os.environ.copy();env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',JAX_PLATFORMS='cpu',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.monotonic()
    with (output/'tests.log').open('wb') as log:
        child=subprocess.Popen(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT)
        code=child.wait()
    after={name:sha((HERE/name).read_bytes()) for name in NAMES}
    with tarfile.open(output/'sources.tar.gz','r:gz') as archive:
        members={member.name:sha(archive.extractfile(member).read()) for member in archive.getmembers()}
    report=dict(passed=code==0 and before==after==members,scope='Actual authentication/loader code with tiny mocked producer/context/device and prepared cache; no actual JAX/context/key/HE',
        command=command,cwd=str(HERE),python=platform.python_version(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        child_pid=child.pid,child_return_code=code,child_reaped=True,elapsed_s=time.monotonic()-started,
        source_hashes_before=before,source_hashes_after=after,archive_member_hashes=members,
        source_archive_sha256=sha((output/'sources.tar.gz').read_bytes()),
        source_files_unchanged=before==after,archive_bytes_verified=before==members,
        actual_loader_wiring=True,actual_encrypted_parity=False,main_sources_changed=False)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('passed','child_return_code','child_reaped','elapsed_s','source_archive_sha256')}))
    return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
