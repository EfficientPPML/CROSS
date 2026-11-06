"""Portable source/archive verifier; never imports candidate runtime."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(label,count,tests=None):
    base=ROOT/label;r=json.loads((base/'report.json').read_text());assert r['passed'] is True
    expected=r['source_hashes_before'];assert len(expected)==count and expected==r['source_hashes_after']
    archive=base/'sources.tar.gz';assert digest(archive)==r['source_archive_sha256']
    with tarfile.open(archive) as tar:
        members=tar.getmembers();assert len(members)==count and len({m.name for m in members})==count and all(m.isfile() for m in members)
        raw={m.name:tar.extractfile(m).read() for m in members}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}==expected
    if tests is not None:
        xml=ET.parse(base/'tests.xml').getroot();suites=list(xml.iter('testsuite'))
        assert sum(int(s.get('tests',0)) for s in suites)==tests
        assert all(int(s.get(k,0))==0 for s in suites for k in ('failures','errors'))
    return r,raw

def main():
    final,raw=verify('preflight-v2',219,113);probe,probe_raw=verify('installed-guards-v1',218);verify('historical-scaffold-v1',214,61)
    candidate={n:b for n,b in raw.items() if not n.startswith('external/')}
    assert len(candidate)==217 and candidate=={n:b for n,b in probe_raw.items() if not n.startswith('external/')}
    assert final['original_198_changed_only_adopted_overlays'] is True
    origin=json.loads(candidate['source_origin.json']);prior=origin['assembly_source_hashes'];assert len(prior)==198
    changes={n:hashlib.sha256(candidate[n]).hexdigest() for n,h in prior.items() if hashlib.sha256(candidate[n]).hexdigest()!=h}
    assert changes==origin['reviewed_runtime_overlays']==final['adopted_runtime_overlays']
    assert set(changes)=={'runtime/stage/key_contexts.py','runtime/stage/full_model_engine.py'}
    assert probe['denied_count']==25 and len(probe['denied_entrypoints'])==25 and probe['all25_restored'] is True
    assert probe['allowed_server_entrypoints_unchanged'] is True and probe['distinct_keygen_aliases'] is True
    assert probe['loaded_local_module_count']==len(probe['loaded_local_modules'])==69
    for row in probe['loaded_local_modules'].values():assert hashlib.sha256(candidate[row['source']]).hexdigest()==row['sha256']
    public=json.loads((ROOT/'preflight-v2/public_gates.json').read_text());assert public['passed'] is True
    assert digest(ROOT/'preflight-v2/public_gates.json')==final['public_gate_report_sha256']
    assert public['tile_count']==8 and public['payload_bytes_attested']==838860800 and public['no_jax_or_torch_imported'] is True
    assert (ROOT/'preflight-v2/check_preflight.py').read_bytes()==raw['external/preflight.py']
    assert (ROOT/'preflight-v2/check_actual_public_gates.py').read_bytes()==raw['external/public_gates.py']
    assert (ROOT/'installed-guards-v1/check_installed_guards.py').read_bytes()==probe_raw['external/installed_guard_probe.py']
    inventory=json.loads((ROOT/'artifact_manifest.json').read_text())
    assert {p.relative_to(ROOT).as_posix():digest(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'}==inventory
    print('PASS:113 mocked checks,219/218 exact sources,25 installed denials,69 module hashes; no actual stage4 claim.')
if __name__=='__main__':main()
