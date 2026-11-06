"""Bounded public source/design readback only; no runtime imports."""
import argparse
import hashlib
import json
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main(current):
    index=json.loads((P/'files.sha256.json').read_bytes())
    names={str(p.relative_to(P))for p in P.rglob('*')if p.is_file()and p.name!='files.sha256.json'}
    assert names==set(index)
    assert all(sha((P/n).read_bytes())==h for n,h in index.items())
    rows=json.loads((P/'reviewed_sources.json').read_bytes());assert len(rows)==20
    for row in rows:
        assert sha((P/row['copy']).read_bytes())==row['sha256']
        if current:assert sha(Path(row['source']).read_bytes())==row['sha256']
    r=json.loads((P/'report.json').read_bytes())
    assert r['reviewed_source_count']==20 and r['sources_unchanged']is True
    for k in ['runtime_imports','tests_run','private_or_ciphertext_body_reads','queue_or_grant_changes','new_comparison_role_implemented','original_b5_recovered','original_b5_throw_site_proved','new_actual_execution_authorized']:assert r[k]is False
    assert json.loads((P/'execution_authorization.json').read_bytes())['ready']is False
    print(json.dumps({'passed':True,'scope':'source-only design','selected_sources':20,'current_checked':current,'actual_execution':False},sort_keys=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--current',action='store_true');main(p.parse_args().current)
