"""Portable public evidence/overlay verification, no runtime imports."""
from pathlib import Path
import hashlib,json,subprocess,tarfile,tempfile
HERE=Path(__file__).resolve().parent
REPORT_SHA='42193921c7360eb6fec76e857d636214b787f61d1397a761104624b723ffc439'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def run():
    r=read(HERE/'source-bound-v1/report.json');s=read(HERE/'summary.json')
    assert sha(HERE/'source-bound-v1/report.json')==REPORT_SHA
    assert r['passed'] and r['tests']==dict(tests=123,failures=0,errors=0,skipped=0)
    assert r['returncode']==0 and r['child_reaped'] and r['owned_test_directory_removed']
    assert r['source_files_unchanged'] and r['source_archive_verified'] and not r['cleanup_errors']
    assert r['threaded_process_signal_atomicity_established'] is False
    assert r['primary_identity_outside_entered_guard_guaranteed'] is False
    before=r['source_hashes_before'];assert before==r['source_hashes_after'] and len(before)==65
    archive=HERE/'source-bound-v1/sources.tar.gz';assert sha(archive)==r['source_archive_sha256']==s['source_archive_sha256']
    with tarfile.open(archive) as t:
        ms=t.getmembers();assert len(ms)==65==len({m.name for m in ms})
        assert all(m.isfile() and m.size<=8<<20 and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in ms)
        assert sum(m.size for m in ms)<32<<20
        raw={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}==before
    assert before['candidate/key_contexts.py']==s['runtime_candidate_sha256']
    assert before['before/key_contexts.py']==s['runtime_base_sha256']
    assert before['candidate/cleanup.py']==s['single_fire_wrapper_sha256']
    assert sha(HERE/'candidate.patch')==s['runtime_patch_sha256']
    with tempfile.TemporaryDirectory(prefix='verify-loader-overlay-') as d:
        root=Path(d);path=root/'runtime/stage/key_contexts.py';path.parent.mkdir(parents=True);path.write_bytes(raw['before/key_contexts.py'])
        subprocess.run(['git','apply',str(HERE/'candidate.patch')],cwd=root,check=True,capture_output=True)
        assert path.read_bytes()==raw['candidate/key_contexts.py']
    term=read(HERE/'source-bound-v1/terminal.json');assert term['session_id']==84909 and term['exit_code']==0 and term['reaped'] and term['child_pid_absent']
    probe=read(HERE/'worker_probe_record.json')
    assert probe['returncode']==0 and probe['reaped'] and probe['source_hashes_before']==probe['source_hashes_after']==before
    assert probe['result']['counterexample_confirmed'] and probe['result']['observations']==[dict(handler_on_main=True,main_thread_mask_contains_signal=True)]
    assert '2 failed, 8 deselected' in (HERE/'negative-v1.log').read_text()
    assert '59 passed, 1 error' in (HERE/'development-v1.log').read_text()
    assert '52 passed' in (HERE/'development-v2.log').read_text() and 'KeyboardInterrupt' in (HERE/'development-v2.log').read_text()
    assert s['live_runtime_promoted'] is False and s['actual_keys_context_ciphertext_model_execution'] is False
    print(json.dumps(dict(passed=True,source_files=65,scoped_tests=123,original_failures_preserved=True,threaded_process_atomicity=False)))
if __name__=='__main__':run()
