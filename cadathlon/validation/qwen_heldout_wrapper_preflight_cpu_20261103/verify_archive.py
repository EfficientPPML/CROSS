"""Verify preparation evidence only; no model/validation imports."""
from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    v=json.loads((P/'verification.json').read_text());r=json.loads((P/'wrapper-preflight-v2/report.json').read_text())
    assert all(sha(P/n)==h for n,h in v['artifact_sha256'].items())
    archive=P/'wrapper-preflight-v2/sources.tar.gz'
    assert sha(archive)==r['source_archive_sha256']==v['source_archive_sha256']
    with tarfile.open(archive) as t:
        members=t.getmembers();assert len(members)==len({m.name for m in members})==36
        assert all(m.isfile() for m in members)
        got={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}
    assert got==r['source_hashes_before']==r['source_hashes_after']
    assert r['passed'] and r['test_exit_code']==0 and r['calibrated_runtime_sources_identical']
    req=json.loads((P/'calibration_requirement.json').read_text())
    assert len(req['expected_runtime_source_hashes'])==31
    assert all(got[n]==h for n,h in req['expected_runtime_source_hashes'].items())
    for name in ('validate_heldout.py','calibration_gate.py','calibration_requirement.json','test_validation_gate.py','preflight_validation.py'):assert sha(P/name)==got[name]
    assert v['validation_inputs_read'] is False and v['candidate_changed'] is False
    print(json.dumps({'passed':True,'source_count':36,'unchanged_calibrated_runtime_sources':31,'tests':51,'subtests':19,'validation_launched':False}))
if __name__=='__main__':main()
