from pathlib import Path
import ast,hashlib,json,tarfile
OUT=Path(__file__).parent
A=Path('/tmp/qwen-server-device-integration-runtime-v3-20261104');B=Path('/tmp/qwen-server-device-integration-runtime-v4-20261104')
AH=Path('/tmp/qwen-server-device-integration-handoff-v3-20261104');BH=Path('/tmp/qwen-server-device-integration-handoff-v4-20261104')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(v):return(json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
a=json.loads((AH/'sources.json').read_bytes());b=json.loads((BH/'sources.json').read_bytes())
assert len(a)==402 and len(b)==403
assert sha(BH/'sources.json')=='56ae461ee94652a66a83ebac56ed4bc13f3f4a52da28cfce44f8c6898b9b4bf0'
assert sha(BH/'sources.tar.gz')=='0ed4491696edb89892f4a37a18f76f57362d2c92e171ffeb340ee356d7e8cb4a'
for root,mapping in ((A,a),(B,b)):
 assert {p.relative_to(root).as_posix()for p in root.rglob('*')if p.is_file()}==set(mapping)
 for name,value in mapping.items():
  p=root/name;assert not p.is_symlink()and sha(p)==value
with tarfile.open(BH/'sources.tar.gz','r:gz')as archive:
 members=archive.getmembers();assert len(members)==403 and all(m.isfile()for m in members)
 assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()for m in members}==b
changed=sorted(n for n in a if a[n]!=b[n]);added=sorted(set(b)-set(a))
assert changed==['qwen_device_policy.py','qwen_server_source_inventory.json']and added==['device_native_open_flags_successor.json']
text=(B/'qwen_device_policy.py').read_text();old=(A/'qwen_device_policy.py').read_text()
block="""        # Named-node metadata is checked before opening and the actual FD is
        # checked afterward. The open itself must neither follow a substituted
        # link, block on a substituted FIFO, nor create/truncate a replacement.
        flags=row['open_flags'];required=os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK
        if flags&os.O_ACCMODE not in(os.O_RDONLY,os.O_RDWR):
            raise ValueError('native device access mode must be RDONLY or RDWR')
        if flags&required!=required or flags&~(os.O_ACCMODE|required):
            raise ValueError('native device requires only nonblocking no-follow close-on-exec flags')
"""
assert text.count(block)==1 and text.replace('import os\n','',1).replace(block,'',1)==old
assert ast.dump(ast.parse(text.replace('import os\n','',1).replace(block,'',1)),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
assert b['qwen_device_policy.py']=='e5e50757e67ca935af8a65a680d9538ac5f5161a6cf4199016a5d13a4af0c4cb'
refs=[BH/'handoff.json',BH/'sources.json',BH/'sources.tar.gz',B/'qwen_device_policy.py',B/'qwen_server_dependencies.py',B/'qwen_server_native_authority.py',B/'qwen_server_readiness.json',Path('/tmp/qwen-server-device-integration-root-review-20261104/correction403-review.json')]
# Readiness path is source inventory-owned; report actual name if baseline uses another.
refs=[p for p in refs if p.is_file()]
report=dict(schema='qwen-full28-native403-peer-source-review-v1',passed=True,source_only=True,source_count=403,parent_count=402,parent_unchanged_count=400,changed=changed,added=added,current_archive_exact=True,parent_preserved=True,policy_delta='Exact os import plus mode/required/allowed flag validation only; no caller flag rewriting.',findings=[],required_flags=['O_CLOEXEC','O_NOFOLLOW','O_NONBLOCK'],allowed_modes=['O_RDONLY','O_RDWR'],strict_integer_precedes_flags=True,existing_wrapper='Original flags are passed unchanged; native path metadata is checked before original os.open, and actual character FD major/minor is checked afterward. Failed postcheck retires original returned FD with shared primary-error handling.',limits=['No device opened or probed; no factual native driver support/lease/capacity claim.','No observed predecessor runtime failure.','Current native authority and actual-entry denial remain selected.','Prior device30/runtime36 proofs do not establish these new semantic flag checks.'],candidate_imports=False,tests=False,probes=False,refs=[dict(path=str(p),sha256=sha(p))for p in refs])
(OUT/'review.json').write_bytes(dump(report));print(json.dumps(dict(passed=True,review=str(OUT/'review.json'),sha256=sha(OUT/'review.json'),source_count=403)))
