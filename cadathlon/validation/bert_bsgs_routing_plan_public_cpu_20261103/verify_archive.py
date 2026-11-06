"""Public artifact/source/count verification; no external or backend reads."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    inventory=json.loads((ROOT/'artifact_hashes.json').read_bytes())
    assert {p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__'not in p.parts}==inventory
    r=json.loads((ROOT/'report.json').read_bytes());b=r['source_hashes_before']
    assert all(r[n]is True for n in ('passed','terminal_scope_completed','cleanup_completed','process_reaped','source_files_unchanged','source_archive_verified'))
    assert len(b)==r['source_count']==10 and b==r['source_hashes_after']==r['snapshot_hashes']
    assert {n:sha(ROOT/'source'/n) for n in b}==b
    assert r['returncode']==0 and r['encrypted_execution']is False
    assert r['full_slot_comparisons']==768 and r['compared_slots']==768*65536
    assert sha(ROOT/'sources.tar.gz')==r['source_archive_sha256']=='6db86228c46611578ae20dff305473545fbd8d058964fd890cdd28542b31800e'
    with tarfile.open(ROOT/'sources.tar.gz') as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==10
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..'not in Path(m.name).parts for m in ms)
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}==b
    ss=list(ET.parse(ROOT/'tests.xml').getroot().iter('testsuite'))
    assert {n:sum(int(s.get(n,0)) for s in ss) for n in ('tests','failures','errors','skipped')}==r['test_counts']==dict(tests=392,failures=0,errors=0,skipped=0)
    p=json.loads((ROOT/'public_plan.json').read_bytes());cost=json.loads((ROOT/'source/cost_review.json').read_bytes())
    assert cost['passed']is True and cost['baseline_K128_rotations']==p['old_total_rotations_excluding_bootstrap']==906829
    assert p['per_map']==dict(original_internal_rotations=26,proposed_internal_rotations=11,plaintext_terms_before=28,plaintext_terms_after=28,ciphertext_additions_before=26,ciphertext_additions_after=26,rescale_calls_before=2,rescale_calls_after=7,critical_rescale_depth_before=2,critical_rescale_depth_after=2)
    assert p['full_k128_routing_maps']==12*8*128*2==24576
    assert p['saved_internal_rotations']==24576*15 and p['extra_rescale_calls']==24576*5
    assert p['proposed_total_rotations_excluding_bootstrap']==906829-24576*15==538189
    keys=p['key_inventory'];assert len(keys['original'])==66 and len(keys['proposed'])==36 and set(keys['proposed'])<=set(keys['original'])
    terminal=json.loads((ROOT/'terminal.json').read_bytes())
    assert terminal==dict(session_id=65329,exit_code=0,reaped=True,producer_processes_absent=True,current_sources_verified=True,original_sources_verified=True)
    print(json.dumps(dict(passed=True,sources=10,tests=392,physical_slot_comparisons=50331648,encrypted_execution=False)))
if __name__=='__main__':main()
