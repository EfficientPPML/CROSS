from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=json.loads((ROOT/'artifact_sha256.json').read_text())
assert all(sha(ROOT/name)==expected for name,expected in manifest.items())
counts={}
for label in ('scale-v1','guard-v2','corrected-initial-preflight','corrected-normalized-preflight'):
 report=json.loads((ROOT/label/'report.json').read_text());before=report['source_hashes_before']
 assert report['passed'] is True and report['source_hashes_after']==before
 assert sha(ROOT/label/'sources.tar.gz')==report['source_archive_sha256']
 with tarfile.open(ROOT/label/'sources.tar.gz') as archive:
  members=archive.getmembers()
  assert len(members)==len(before) and {m.name for m in members}==set(before) and all(m.isfile() for m in members)
  restored={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}
 assert restored==before
 counts[label]=len(before)
print(json.dumps({'passed':True,'artifact_count':len(manifest),'source_counts':counts},indent=2))
