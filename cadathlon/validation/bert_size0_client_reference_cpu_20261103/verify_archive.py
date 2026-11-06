"""Offline evidence readback; optional private files never printed."""
from pathlib import Path
import argparse,hashlib,json,math,stat,tarfile
HERE=Path(__file__).resolve().parent
REPORT_SHA='b1309f39d7554542213a05c89864628472a76a3644b60d6633bf8eb3e741e1d9'
FIXTURE_SHA='2218282c9d4ef468a448d00d6e7afeebc9c3a9fa0e9f3834227afe299ea28e10'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_bytes())
def source(p,expected):
    with tarfile.open(p) as t:
        ms=t.getmembers()
        assert len(ms)==len(expected)==len({m.name for m in ms})
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.size<8<<20 for m in ms)
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==expected
def run(private=False):
    r=read(HERE/'actual-v1/report.json');p=read(HERE/'preflight-v1/report.json')
    assert sha(HERE/'actual-v1/report.json')==REPORT_SHA
    assert r['passed'] is True and p['passed'] is True
    assert r['source_hashes_before']==r['source_hashes_after']==p['source_hashes_before']==p['source_hashes_after']
    assert len(r['source_hashes_before'])==23 and p['tests']==dict(tests=20,errors=0,failures=0,skipped=0)
    assert r['preflight_report_sha256']==sha(HERE/'preflight-v1/report.json')
    for branch,x,session in [('actual-v1',r,94161),('preflight-v1',p,76137)]:
        assert x['source_files_unchanged'] and x['assets_unchanged'] and x['original_fixture_unchanged']
        assert x['no_jax_or_he_modules_loaded'] and not x['cleanup_errors'] and not x.get('error')
        assert sha(HERE/branch/'sources.tar.gz')==x['source_archive_sha256']
        source(HERE/branch/'sources.tar.gz',x['source_hashes_before'])
        terminal=read(HERE/branch/'terminal.json')
        assert terminal['session_id']==session and terminal['exit_code']==0 and terminal['session_reaped']
    assert r['record_count']==1 and r['key_count']==32 and r['approximate_complete']
    assert len(r['domain_observations'])==156
    assert all(x[k]==0 for x in r['domain_observations'] for k in ('below','above','nonfinite'))
    exact=r['exact_fp64_parity'];assert exact['gate']==1e-12 and len(exact['layers'])==12
    assert all(x[k]['passed'] and x[k]['max_abs']<=1e-12 for x in exact['layers'] for k in ('attention','block'))
    assert all(exact[k]['passed'] and exact[k]['max_abs']<=1e-12 for k in ('pooler','logits'))
    assert r['client_embedding_native_parity']=={'max_abs':0.,'passed':True}
    assert r['approximate_vs_official_fp32']['passed'] and r['prediction_agreement_official_fp32']
    assert r['official_native']['dtype']=='float32' and r['official_native']['attention_implementation']=='sdpa'
    f=read(HERE/'prerequisites/fixture_report.json');assert sha(HERE/'prerequisites/fixture_report.json')==FIXTURE_SHA==r['fixture_report_sha256']
    assert sha(HERE/'prerequisites/fixture_sources.tar.gz')==f['source_archive_sha256']
    source(HERE/'prerequisites/fixture_sources.tar.gz',f['source_hashes_before'])
    assert f['input_file_sha256']==r['original_private_files_before']==r['original_private_files_after']
    if private:
        for root,identity in [(Path(r['private_workspace']),r['private_workspace_identity']),(Path(f['private_workspace']),f['private_workspace_identity'])]:
            s=root.lstat();assert stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode)==0o700
            assert {'device':s.st_dev,'inode':s.st_ino}==identity
        path=Path(r['private_reference_path']);s=path.lstat()
        assert stat.S_ISREG(s.st_mode) and stat.S_IMODE(s.st_mode)==0o600 and sha(path)==r['private_reference_sha256']
        a=read(path);input_path=Path(f['preprocessed_input_path'])
        assert sha(input_path)==a['input_sha256']==r['input_sha256']
        record=json.loads(input_path.read_text().strip());assert a['target_idx']==record['target_idx']
        for name in ('native_fp32_logits','native_fp64_logits','independent_exact_fp64_logits','approximate_fp64_logits'):
            assert len(a[name])==2 and all(type(x)is float and math.isfinite(x) for x in a[name])
        u,v=a['approximate_fp64_logits'],a['native_fp32_logits']
        assert all(abs(x-y)<=.02+.01*abs(y) for x,y in zip(u,v))
        assert (u[1]>=u[0])==(v[1]>=v[0])
        files={x.relative_to(Path(f['private_workspace'])).as_posix():sha(x) for x in Path(f['private_workspace']).rglob('*') if x.is_file()}
        assert files==f['input_file_sha256']
    print(json.dumps(dict(passed=True,sources=23,tests=20,domains=156,private_checked=private,scope='client-only one-sample reference; no model execution by verifier')))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--private',action='store_true');run(p.parse_args().private)
