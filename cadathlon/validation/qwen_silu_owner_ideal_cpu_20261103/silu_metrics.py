"""Collector-enabled aggregate-only ideal metrics; no slot-array persistence."""
from pathlib import Path
import json,os

def record(name,*,error,domains,result,scope):
    destination=os.environ.get('QWEN_SILU_METRICS_PATH')
    if destination is None:return
    p=Path(destination);data=json.loads(p.read_bytes()) if p.exists() else {}
    if name in data:raise ValueError('duplicate ideal fixture record')
    data[name]=dict(max_abs_error=float(error),gate=1e-8,physical_slots_checked=65536,
        domains=domains,counts=result.plan['counts'],input_scale=result.plan['input_scale'],
        output_level=result.output.level,output_scale=result.output._ckks_scale,
        refresh_scales=result.plan['refresh_scales'],scope=scope)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(p)
