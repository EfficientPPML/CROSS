from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    expected=json.loads((ROOT/'artifact_hashes.json').read_bytes())
    assert {p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__'not in p.parts}==expected
    r=json.loads((ROOT/'report.json').read_bytes());assert r['passed'] is True
    assert r['source_hashes_before']==r['source_hashes_after'];assert len(r['source_hashes_before'])==13
    assert sha(ROOT/'sources.tar.gz')==r['source_archive_sha256']=='73d40ff4d0c8de6bd360d08eded028a0e0603f9f6ed381cb6352dc6d975acb44'
    with tarfile.open(ROOT/'sources.tar.gz') as t:
        ms=t.getmembers();assert len(ms)==13==len({m.name for m in ms});assert all(m.isfile() for m in ms)
        source={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(v).hexdigest() for n,v in source.items()}==r['source_hashes_before']
    assert sha(ROOT/'audit.py')==r['source_hashes_before']['audit.py']
    for row in r['storage']:
        q=2*(row['level']+1);parts=(q+12)//13
        assert row['q_count']==q and row['effective_parts']==parts
        assert row['full_digits_uint32_bytes']==parts*131072*(q+14)*4
    print(json.dumps(dict(passed=True,source_count=13,scope='public source/integer inventory only; no HE or parity execution')))
if __name__=='__main__':main()
