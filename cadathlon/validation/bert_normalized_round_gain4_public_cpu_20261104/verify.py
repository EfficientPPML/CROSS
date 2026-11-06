from pathlib import Path
import hashlib,json
from verify_gain4 import verify
root=Path(__file__).resolve().parent
result=verify(root)
pins=json.loads((root/'artifact_hashes.json').read_bytes())
assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in pins.items())
terminal=json.loads((root/'terminal_observation.json').read_bytes())
assert terminal['authoritatively_reaped']is True and terminal['exit_code']==0 and terminal['all_producer_pids_absent']is True
print(json.dumps(result,sort_keys=True))
