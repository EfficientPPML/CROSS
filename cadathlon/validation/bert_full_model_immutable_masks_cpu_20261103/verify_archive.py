"""Archive-only exact source and aggregate gate readback; no numerical rerun."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def members(path):
    with tarfile.open(path) as tar:
        rows=tar.getmembers();assert all(m.isfile() for m in rows) and len(rows)==len({m.name for m in rows})==137
        return {m.name:sha(tar.extractfile(m).read()) for m in rows}
def verify():
    summary=json.loads((ROOT/'summary.json').read_text());new=members(ROOT/'measured/sources.tar.gz');old=members(ROOT/'provenance/parent_sources.tar.gz')
    assert sha((ROOT/'measured/sources.tar.gz').read_bytes())==summary['source_archive_sha256']
    assert sha((ROOT/'provenance/parent_sources.tar.gz').read_bytes())==summary['parent_archive_sha256']
    assert {n for n in new if old[n]!=new[n]}==set(summary['changed_sources'])
    assert set(old)==set(new)
    for name in ('measured','smoke'):
        report=json.loads((ROOT/name/'report.json').read_text())
        assert new==report['source_hashes_before']==report['source_hashes_after']==report['snapshot_source_hashes_after']==members(ROOT/name/'sources.tar.gz')
        assert report['source_files_unchanged'] and report['source_archive_verified'] and report['asset_files_unchanged'] and not report['cleanup_errors']
        assert report['process_cleanup']['reaped'] and report['process_cleanup']['group_gone'] and report['test_exit_code']==0
        if name=='smoke':assert not report['complete'] and not report['passed'] and report['tests']==dict(tests=44,errors=0,failures=0,skipped=0)
        else:
            assert report['passed'] and report['complete'] and report['selection'] is None and report['tests']==dict(tests=46,errors=0,failures=0,skipped=0)
            assert {c['K'] for c in report['cases']}=={15,128}
            for c in report['cases']:
                assert c['passed'] and c['all_connections_are_actual_output_objects'] and c['ideal_gate']==1e-7 and c['native_prediction_agrees']
                assert c['native_logit_gate']=='abs(error)<=0.02+0.01*abs(native)' and c['output_level']==2
                assert all(d['passed'] for d in c['actual_domain_observations'])
                assert max(d['max_abs'] for d in c['stage_errors'])<=1e-7
                assert c['logits_all_slot_max_abs']<=1e-7 and c['pooler_all_slot_max_abs']<=1e-7
    inventory=json.loads((ROOT/'archive_inventory.json').read_text())
    assert {p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file() and p.name!='archive_inventory.json' and '__pycache__' not in p.parts}==inventory
    print(json.dumps(dict(passed=True,tests=46,cases=[15,128],sources=137,runtime_modules_changed=1,HE=False)))
if __name__=='__main__':verify()
