from pathlib import Path
import hashlib,json
from analyze import derive
p=Path(__file__).resolve().parent
pins=json.loads((p/'artifact_sha256.json').read_bytes())
assert set(pins)=={x.relative_to(p).as_posix() for x in p.rglob('*') if x.is_file() and '__pycache__'not in x.parts and x.name!='artifact_sha256.json'}
for n,h in pins.items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h,n
assert derive()==json.loads((p/'analysis.json').read_bytes())
print('VERIFIED:public reciprocal sensitivity diagnosis; original encrypted run remains FAIL')
