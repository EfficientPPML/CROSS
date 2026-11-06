from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def tar_sources(path,expected):
    with tarfile.open(path) as t:
        ms=t.getmembers();assert len(ms)==len(expected)==len({m.name for m in ms})
        assert all(m.isfile() and m.size<=8<<20 and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in ms)
        assert sum(m.size for m in ms)<=32<<20
        raw={m.name:t.extractfile(m).read() for m in ms}
    assert {n:sha(b) for n,b in raw.items()}==expected
    return raw
def verify():
    folder=ROOT/'measured';r=json.loads((folder/'report.json').read_bytes());s=json.loads((ROOT/'summary.json').read_bytes())
    assert sha((folder/'report.json').read_bytes())==s['report_sha256']
    expected=r['source_hashes_start'];assert r['passed'] and len(expected)==81
    assert expected==r['source_hashes_end']==r['snapshot_hashes_end']==r['archive_member_hashes']
    assert sha((folder/'sources.tar.gz').read_bytes())==r['source_archive_sha256']==s['source_archive_sha256']=='13d798b3cae5e34ca3713372aad0d1527f7d2f1ee5e0268a4502e5d82634eed8'
    raw=tar_sources(folder/'sources.tar.gz',expected)
    assert raw['preflight_driver.py']==(ROOT/'preflight_driver.py').read_bytes()
    assert raw['source_origin.json']==(ROOT/'source_origin.json').read_bytes()
    origin=json.loads(raw['source_origin.json']);old={'runtime/'+n:h for n,h in origin['first_handler_runtime_source_hashes'].items()}|origin['additional_source']
    assert len(old)==59
    overlay={'runtime/stage/key_contexts.py':'7eb793deeccae8cd5a1e09f368f2465d39bfed21b379aefca46474bf6532c205'}
    assert r['reviewed_runtime_overlay']==overlay
    assert {n for n in old if expected[n]!=old[n]}==set(overlay)
    assert all(expected[n]==h for n,h in overlay.items())
    assert r['preserved_prior_runtime_sources']==58 and r['preserved_prior_runtime_sources_unchanged']
    assert expected['cleanup.py']=='1798455cdd88d4bbac5e1769e6b3226676ccb9901e7be6faf383e7ad063c1ef6'
    suites=list(ET.parse(folder/'tests.xml').getroot().iter('testsuite'))
    assert sum(int(x.attrib.get('tests',0)) for x in suites)==r['mock_tests']==135
    assert not any(int(x.attrib.get(n,0)) for x in suites for n in ('failures','errors','skipped'))
    probe=json.loads((folder/'real_import_guard.json').read_bytes())
    assert probe['passed'] and probe['denial_count']==probe['restored_count']==21
    assert not any(probe[n] for n in ('context_constructed','keys_generated','model_constructed','asset_payloads_read','encrypt_called','decrypt_called'))
    for branch,code in [('negative-terminal-scope-v1',2),('negative-terminal-scope-v2',1),('negative-main-exit-v1',1)]:
        p=ROOT/'history'/branch;h=json.loads((p/'report.json').read_bytes())
        assert h['expected_failure'] and h['returncode']==code
        assert sha((p/'sources.tar.gz').read_bytes())==h['source_archive_sha256']
        assert h['source_hashes_before']==h['source_hashes_after']
        tar_sources(p/'sources.tar.gz',h['source_hashes_before'])
    term=json.loads((folder/'terminal.json').read_bytes());assert term['session_id']==79107 and term['exit_code']==0 and term['reaped'] and term['pid_absent']
    assert s['actual_stage3_started'] is False
    print(json.dumps(dict(passed=True,v3_proof_sources=81,candidate_sources=80,mock_tests=135,actual_entrypoint_denials=21,actual_stage3_in_this_proof=False)))
if __name__=='__main__':verify()
