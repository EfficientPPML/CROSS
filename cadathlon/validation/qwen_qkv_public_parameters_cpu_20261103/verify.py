"""Offline metadata/source verification; no tensor bodies or model imports."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for name,count in [('result-v1',4),('fd-successor-v2',3)]:
    base=P/name;r=json.loads((base/'report.json').read_bytes())
    assert r['passed']is True and len(r['source_hashes_before'])==count
    assert r['source_hashes_before']==r['source_hashes_after']
    assert sha(base/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(base/'sources.tar.gz') as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==count and all(m.isfile() for m in ms)
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==r['source_hashes_before']
r=json.loads((P/'result-v1/report.json').read_bytes())
assert r['operation_complete']is True and r['archive_verified']is True and len(r['arrays'])==84
assert r['asset_sha256_before']==r['asset_sha256_after']
assert r['held_checkpoint_sha256_after']=='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
assert sha(P/'result-v1/parameter_manifest.json')==r['parameter_manifest_sha256']=='c707068e1a247f4ef14fda3493a3e2edf9dea2f11ec19e94531a64d80dbcf48b'
m=json.loads((P/'result-v1/parameter_manifest.json').read_bytes())
assert m['matrices']==r['arrays']
for layer in range(28):
    for kind in ('q','k','v'):
        row=m['matrices'][f'model.layers.{layer}.self_attn.{kind}_proj.weight']
        assert row['shape']==[2048 if kind=='q' else 1024,1024]
        assert row['source_dtype']=='BF16' and row['dtype']=='float64' and row['promotion_exact']is True
s=ET.parse(P/'fd-successor-v2/tests.xml').getroot().findall('testsuite')
counts={n:sum(int(x.get(n,0)) for x in s) for n in ('tests','errors','failures','skipped')}
assert counts==dict(tests=14,errors=0,failures=0,skipped=0)
assert '1 failed' in (P/'fd-negative-v1/tests.log').read_text()
print('PASS: 84 public tensor metadata records, exact export sources, and separate descriptor cleanup regression.')
