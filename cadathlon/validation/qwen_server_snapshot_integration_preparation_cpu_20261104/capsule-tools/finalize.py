"""Preserve the verifier-authoring correction and refresh generated capsule indices."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
C=Path('/tmp/qwen_server_snapshot_integration_preparation_cpu_20261104')
H=lambda raw:hashlib.sha256(raw).hexdigest()
def canonical(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
orig=json.loads((C/'copy_origins.json').read_bytes())
for source,dest in [(B/'verify.py','verify.py'),(B/'verify_initial_grant_shape.py','capsule-tools/verify_initial_grant_shape.py'),(B/'verifier-history.json','capsule-tools/verifier-history.json'),(B/'finalize.py','capsule-tools/finalize.py')]:
    raw=source.read_bytes();target=C/dest;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    orig['files'][dest]=dict(source=str(source),sha256=H(raw),bytes=len(raw))
(C/'copy_origins.json').write_bytes(canonical(orig))
history=C/'HISTORY.md'
note='\nCapsule verifier authoring: an initial public readback stopped because it assumed the root grant authorization field was boolean. The exact field is a fixed scope string; the corrected verifier compares that string. The initial verifier and correction note are preserved under capsule-tools. No candidate source, proof, grant or execution was changed or rerun.\n'
assert note not in history.read_text();history.write_text(history.read_text()+note)
names=sorted(p.relative_to(C).as_posix()for p in C.rglob('*')if p.is_file())
(C/'WHITELIST.json').write_bytes(canonical(dict(schema='closed-public-capsule-whitelist-v1',files=names)))
(C/'SHA256SUMS').write_text(''.join(H((C/n).read_bytes())+'  '+n+'\n'for n in names if n!='SHA256SUMS'))
print(json.dumps(dict(files=len(names),index_sha256=H((C/'SHA256SUMS').read_bytes()),whitelist_sha256=H((C/'WHITELIST.json').read_bytes()))))
