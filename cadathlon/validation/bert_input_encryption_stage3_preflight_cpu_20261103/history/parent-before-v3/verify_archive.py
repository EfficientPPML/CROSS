"""Verify both preparation generations without importing crypto/model packages."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def checkpoint(folder,count,tests):
    report=json.loads((folder/'report.json').read_text());expected=report['source_hashes_start']
    assert report['passed']is True and len(expected)==count and 'preflight_driver.py' in expected
    assert expected==report['source_hashes_end']==report['snapshot_hashes_end']==report['archive_member_hashes']
    archive=folder/'sources.tar.gz';assert sha(archive.read_bytes())==report['source_archive_sha256']
    with tarfile.open(archive) as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})==count and all(m.isfile() for m in members)
        raw={m.name:tar.extractfile(m).read() for m in members};assert {n:sha(b) for n,b in raw.items()}==expected
    origin=json.loads(raw['source_origin.json']);old={'runtime/'+n:h for n,h in origin['first_handler_runtime_source_hashes'].items()}|origin['additional_source']
    assert len(old)==59 and all(expected[n]==h for n,h in old.items())
    suites=list(ET.parse(folder/'tests.xml').getroot().iter('testsuite'))
    assert sum(int(s.attrib.get('tests',0)) for s in suites)==tests
    assert not any(int(s.attrib.get(n,0)) for s in suites for n in ('failures','errors','skipped'))
    probe=json.loads((folder/'real_import_guard.json').read_text())
    assert probe['passed']is True and probe['denial_count']==probe['restored_count']==21
    assert len(set(probe['actual_entrypoints_denied']))==21
    assert not any(probe[n] for n in ('context_constructed','keys_generated','model_constructed','asset_payloads_read','encrypt_called','decrypt_called'))
    return raw

def verify():
    old=checkpoint(ROOT/'measured',76,115);new=checkpoint(ROOT/'successor-v2/measured',79,120)
    assert set(new)-set(old)=={'key_source_manifest.json','key_source_archive_fixture.tar.gz','test_gate_revision.py'}
    assert {n for n in old if old[n]!=new[n]}=={'key_gate.py','model_gate.py','collect.py','test_model_gate.py','source_origin.json','preflight_driver.py'}
    assert all(old[n]==new[n] for n in old if n.startswith('runtime/'))
    assert (ROOT/'history/negative-plan-size-v1/model_gate.py').read_bytes()==old['model_gate.py']
    assert json.loads(ROOT.joinpath('summary.json').read_text())['actual_stage3_started']is False
    assert json.loads(ROOT.joinpath('successor-v2/summary.json').read_text())['actual_stage3_started']is False
    manifest=json.loads(new['key_source_manifest.json']);assert len(manifest)==58
    import io
    with tarfile.open(fileobj=io.BytesIO(new['key_source_archive_fixture.tar.gz'])) as tar:
        ms=tar.getmembers();assert len(ms)==58
        assert {m.name:sha(tar.extractfile(m).read()) for m in ms}==manifest
        assert next(m.size for m in ms if m.name=='stage/server_inputs/twelve_layer_plan.json')==6461050
    inventory=json.loads((ROOT/'archive_inventory.json').read_text())
    actual={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file() and p.name!='archive_inventory.json' and '__pycache__' not in p.parts}
    assert actual==inventory
    print(json.dumps(dict(passed=True,v1=dict(proof_sources=76,mock_tests=115),v2=dict(proof_sources=79,candidate_sources=78,mock_tests=120),actual_entrypoint_denials_each=21,actual_stage3=False)))
if __name__=='__main__':verify()
