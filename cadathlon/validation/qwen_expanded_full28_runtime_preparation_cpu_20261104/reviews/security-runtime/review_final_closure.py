"""Final declared source closure readback only; no candidate imports/tests."""
from pathlib import Path
import ast, datetime, hashlib, json, tarfile

ROOT=Path('/tmp/qwen-expanded-full-model-runtime-final-20261104')
OLD=Path('/tmp/qwen-expanded-full-model-runtime-v3-20261104')
PROOF=Path('/tmp/qwen-expanded-full-model-preflight-20261104')
DRIVER=Path('/tmp/qwen-expanded-full-preparation-driver-v3-20261104')
OUT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
    raw=(PROOF/'sources.json').read_bytes();mapping=json.loads(raw)
    assert sha(raw)=='b048ea18abcb887d491f5190401445e6fc5a0d3c72f1dabd6ad9475893f7def6' and len(mapping)==198
    archive=(PROOF/'sources.tar.gz').read_bytes()
    assert sha(archive)=='315df2659e81f80b08efd98b8f5a42ccaf272eca91a918223c38d0acb5cfc02a'
    for root in (ROOT,PROOF/'execution'):
        files=[p for p in root.rglob('*') if p.is_file()]
        assert not any(p.is_symlink() for p in root.rglob('*'))
        assert {p.relative_to(root).as_posix() for p in files}==set(mapping)
        assert {n:sha((root/n).read_bytes()) for n in mapping}==mapping
    with tarfile.open(PROOF/'sources.tar.gz') as tar:
        members=tar.getmembers()
        assert len(members)==len({m.name for m in members})==198 and all(m.isfile() for m in members)
        assert {m.name:sha(tar.extractfile(m).read()) for m in members}==mapping
    parent_map=Path('/tmp/qwen-expanded-full-model-runtime-v3-sources-20261104.json').read_bytes()
    parent=json.loads(parent_map)
    assert len(parent)==191 and {n:sha((OLD/n).read_bytes()) for n in parent}==parent
    origin=json.loads((ROOT/'final_origin.json').read_bytes())
    assert origin['parent_manifest_sha256']==sha(parent_map)
    additions=set(mapping)-set(parent)
    assert additions==set(origin['added_files'])=={'prepare_expanded.py','freeze_io.py','cleanup_origin.py','helper_provenance.json','driver_history.json','selected_preparation.json','final_origin.json'}
    changes=[n for n,h in parent.items() if mapping[n]!=h]
    assert changes==['PREPARATION.md']
    for name in ('prepare_expanded.py','freeze_io.py','cleanup_origin.py','helper_provenance.json','driver_history.json'):
        assert (ROOT/name).read_bytes()==(DRIVER/name).read_bytes()
    assert mapping['prepare_expanded.py']=='1a7c49dff76aba6a0a1c492f043ac1c5248217fd909b02e6df1330e659572c14'
    select=json.loads((ROOT/'selected_preparation.json').read_bytes())
    inventory=json.loads((ROOT/'successor_inventory.json').read_bytes())
    assert select['test_files']==inventory['selected_tests'] and select['expected_tests']==inventory['expected_selected_tests']==176
    assert select['cpu_affinity']==[112,113,114,115] and select['python_flags']==['-I','-B']
    assert select['actual_mode_exists']is False and select['external_owner_receipt_required']is True
    sizes=[(ROOT/n).stat().st_size for n in mapping]
    assert sum(sizes)==11581760 and max(sizes)==3312334
    report=dict(schema='qwen-expanded-full-final-closure-security-review-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        passed=True,scope='Source/public metadata closure comparison only. No imports, tests, certificate recomputation, model/input/private bodies or HE.',
        source_count=198,source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive),source_bytes=sum(sizes),largest_source_bytes=max(sizes),
        original_execution_archive_exact=True,declared_inventory_exact=True,parent_count=191,parent_manifest_sha256=sha(parent_map),
        changed_parent_files=changes,added_source_hashes={n:mapping[n] for n in sorted(additions)},runtime_and_generated_inputs_unchanged=True,
        selection=select,prior_source_review_sha256=sha((OUT/'review-v3.json').read_bytes()),prior_driver_review_sha256=sha((OUT/'review-driver-v3.json').read_bytes()),
        authorization='Existing root conditional grant for one176 public/synthetic preparation run is satisfied by this exact closure. Transformer owns launch/reap and independent outer process receipt.',
        limitations=['This is not a176-test result. Report/source/archive and authoritative outer exit/reap/group absence must be checked after the run.',
            'No actual model/calibration/heldout/row0/HE authority or encrypted refresh/head accuracy follows.'])
    (OUT/'review-final-closure.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True,sources=198,receipt_sha256=sha((OUT/'review-final-closure.json').read_bytes()))))
if __name__=='__main__':main()
