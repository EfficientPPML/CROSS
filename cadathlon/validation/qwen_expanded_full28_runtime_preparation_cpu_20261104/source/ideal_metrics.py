"""Aggregate synthetic-fixture diagnostics only; never save tensor bodies."""
from pathlib import Path
import json,math,os
def record(*,layer,sequence,positions,domains,max_abs_error,counts,metadata,refreshes,padding_probe=False):
    path=os.environ.get('QWEN_JOIN_METRICS_PATH')
    if path is None:return
    if not math.isfinite(max_abs_error) or max_abs_error>1e-8:raise ValueError('unchanged all-slot ideal gate failed')
    if not all(row['passed']is True for row in domains):raise ValueError('fixture domain failure')
    output=Path(path);reports=json.loads(output.read_bytes()) if output.exists() else {}
    key=f'layer{layer}.S{sequence}'
    if key in reports:raise ValueError('duplicate genuine-weight fixture result')
    reports[key]=dict(layer=layer,sequence=sequence,original_positions=list(positions),physical_slots_checked=len(metadata)*65536,
        max_abs_error=max_abs_error,gate=1e-8,domains=domains,counts=dict(counts),actual_metadata=list(metadata),
        outer_refreshes=list(refreshes),invalid_refresh_lane_probe=padding_probe,
        scope='Fixed public synthetic input rows, actual pinned public QKV/gamma, ideal primitive and identity-refresh doubles.')
    temp=output.with_suffix('.tmp');temp.write_text(json.dumps(reports,indent=2,allow_nan=False)+'\n');temp.replace(output)
