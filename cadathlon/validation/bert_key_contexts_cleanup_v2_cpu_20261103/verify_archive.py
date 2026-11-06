from pathlib import Path
import hashlib,json,subprocess,tarfile,tempfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def archive(p,expected):
    with tarfile.open(p) as t:
        ms=t.getmembers();assert len(ms)==len(expected)==len({m.name for m in ms})
        assert sum(m.size for m in ms)<32<<20
        assert all(m.isfile() and m.size<=8<<20 and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in ms)
        raw={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}==expected
    return raw
def run():
    r=read(HERE/'source-bound-v1/report.json');s=read(HERE/'summary.json')
    assert sha(HERE/'source-bound-v1/report.json')=='76e571e67a811ec1ce00814e9428d20755d2b35795e46f534ba3cb3de88d0284'
    assert r['passed'] and r['tests']==dict(tests=127,errors=0,failures=0,skipped=0)
    assert r['returncode']==0 and r['child_reaped'] and r['owned_test_directory_removed'] and not r['cleanup_errors']
    assert r['source_files_unchanged'] and r['source_archive_verified']
    before=r['source_hashes_before'];assert len(before)==77 and before==r['source_hashes_after']
    assert sha(HERE/'source-bound-v1/sources.tar.gz')==r['source_archive_sha256']==s['source_archive_sha256']
    raw=archive(HERE/'source-bound-v1/sources.tar.gz',before)
    assert before['candidate/key_contexts.py']==s['candidate_sha256']
    assert before['candidate/cleanup.py']==s['wrapper_sha256']
    for name,base,key in [('candidate.patch','prerequisites/original_key_contexts.py','full_patch_sha256'),('incremental.patch','before/key_contexts.py','incremental_patch_sha256')]:
        assert sha(HERE/name)==s[key]
        with tempfile.TemporaryDirectory(prefix='verify-key-loader-v2-') as d:
            root=Path(d);p=root/'runtime/stage/key_contexts.py';p.parent.mkdir(parents=True);p.write_bytes(raw[base])
            subprocess.run(['git','apply',str(HERE/name)],cwd=root,check=True,capture_output=True)
            assert p.read_bytes()==raw['candidate/key_contexts.py']
    terminal=read(HERE/'source-bound-v1/terminal.json');assert terminal['session_id']==15402 and terminal['exit_code']==0 and terminal['reaped'] and terminal['child_pid_absent']
    old=read(HERE/'prior-v1/report.json')
    assert sha(HERE/'prior-v1/report.json')=='42193921c7360eb6fec76e857d636214b787f61d1397a761104624b723ffc439'
    assert old['tests']['tests']==123 and old['passed']
    assert sha(HERE/'prior-v1/sources.tar.gz')==old['source_archive_sha256']=='0bbaf92207c376c32c4fbcf18fb26e62924bf3e4bdfd002125bc646a17e5b126'
    oldraw=archive(HERE/'prior-v1/sources.tar.gz',old['source_hashes_before'])
    assert oldraw['candidate/key_contexts.py']==raw['before/key_contexts.py']
    assert '4 failed' in (HERE/'negative-mask-v1.log').read_text()
    assert '2 failed, 8 deselected' in (HERE/'prior-v1/negative-v1.log').read_text()
    assert 'KeyboardInterrupt' in (HERE/'prior-v1/development-v2.log').read_text()
    probe=read(HERE/'prior-v1/worker_probe_record.json');assert probe['result']['counterexample_confirmed']
    assert s['threaded_process_atomicity'] is False and s['runtime_promoted'] is False
    print(json.dumps(dict(passed=True,sources=77,scoped_tests=127,prior_proofs_preserved=True,threaded_process_atomicity=False)))
if __name__=='__main__':run()
