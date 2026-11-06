"""Offline whitelist/readback verification; never reads private contents by default."""
from pathlib import Path
import argparse
import hashlib
import json
import stat
import tarfile


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify(root,private=False):
    expected=json.loads((root/'artifact_hashes.json').read_text())
    actual={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
    assert actual==expected,'artifact inventory or bytes differ'
    before=None
    for directory in ('preflight-v2','actual-v1'):
        report=json.loads((root/directory/'report.json').read_text())
        assert report['passed'] and report['sources_unchanged']
        assert report['source_hashes_before']==report['source_hashes_after']
        assert len(report['source_hashes_before'])==22
        if before is not None:assert before==report['source_hashes_before']
        before=report['source_hashes_before']
        path=root/directory/'sources.tar.gz'
        assert sha(path)==report['source_archive_sha256']
        with tarfile.open(path) as t:
            members=t.getmembers()
            assert len(members)==22 and {m.name for m in members}==set(before)
            for m in members:
                assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                assert hashlib.sha256(t.extractfile(m).read()).hexdigest()==before[m.name]
    tests=json.loads((root/'preflight-v2/report.json').read_text())
    assert tests['collected']==20 and tests['failed']==0 and tests['exitstatus']==0
    r=json.loads((root/'actual-v1/report.json').read_text())
    assert sha(root/'actual-v1/report.json')=='2218282c9d4ef468a448d00d6e7afeebc9c3a9fa0e9f3834227afe299ea28e10'
    assert (r['size'],r['base_seed'],r['generator_seed'],r['record_count'])==(0,42,191664963,1)
    assert r['selection_byte_parity'] and r['tokenization_byte_parity'] and r['model_construction_and_forward_guards']
    assert r['assets_unchanged'] and r['asset_file_sha256_before']==r['asset_file_sha256_after']
    assert r['upstream_originals_unchanged'] and r['upstream_original_sha256_before']==r['upstream_original_sha256_after']
    assert r['token_lengths']==[128] and r['valid_token_counts']==[32]
    assert r['private_workspace_permissions']=='0o700' and r['private_file_permissions_all_0600']
    assert not r['cleanup_errors'] and r['private_workspace_retained']
    terminal=json.loads((root/'terminal.json').read_text())
    for name in ('actual','preflight'):assert terminal[name]['reaped'] and terminal[name]['exit_code']==0
    if private:
        work=Path(r['private_workspace']);info=work.lstat()
        assert stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode)==0o700
        assert {'device':info.st_dev,'inode':info.st_ino}==r['private_workspace_identity']
        files={str(p.relative_to(work)) for p in work.rglob('*') if p.is_file()}
        assert files==set(r['input_file_sha256'])
        for name,digest in r['input_file_sha256'].items():
            p=work/name;s=p.lstat()
            assert stat.S_ISREG(s.st_mode) and stat.S_IMODE(s.st_mode)==0o600 and sha(p)==digest
        p=Path(r['preprocessed_input_path'])
        assert p.is_relative_to(work) and sha(p)=='c30ffc79d8150cf6cf3d544bca23a07150d8a0d0f7b7924788e91df3aa55991f'
    return {'passed':True,'sources':22,'tests':20,'private_checked':private,'scope':'tokenizer-only client fixture; no model or HE'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--private',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.private),sort_keys=True))
