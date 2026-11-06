"""Verify recorded preflight only; never construct keys or a crypto context."""
from pathlib import Path
import hashlib,json,math,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def archive(path,expected):
    with tarfile.open(path) as t:
        members=t.getmembers();assert len(members)==len(expected)==len({m.name for m in members}) and all(m.isfile() for m in members)
        raw={m.name:t.extractfile(m).read() for m in members}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}==expected
    return raw

def main():
    pins=json.loads((ROOT/'artifact_sha256.json').read_text());assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in pins.items())
    for name,tests,count in (('preflight-v1',22,57),('preflight-v2',29,58)):
        directory=ROOT/name;r=json.loads((directory/'report.json').read_text());p=json.loads((directory/'public-preflight/report.json').read_text())
        assert r['passed'] and r['source_files_unchanged'] and r['source_archive_verified'] and r['actual_key_generation'] is False
        assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==count
        raw=archive(directory/'sources.tar.gz',r['source_hashes_before']);archive(directory/'public-preflight/sources.tar.gz',p['source_hashes_start'])
        assert hashlib.sha256((directory/'sources.tar.gz').read_bytes()).hexdigest()==r['source_archive_sha256']
        assert hashlib.sha256((directory/'public-preflight/report.json').read_bytes()).hexdigest()==r['public_preflight_sha256']
        assert p['source_hashes_start']==p['source_hashes_end']==r['source_hashes_before']==p['isolated_execution_source_hashes']
        assert p['passed'] and p['execute'] is False and p['actual_key_generation'] is False and p['workspace_retained'] is False and p['cleanup_completed']
        assert r['tests']==dict(tests=tests,failures=0,errors=0,skipped=0)
        assert sum(int(x.attrib['tests']) for x in ET.parse(directory/'tests.xml').getroot())==tests
        profile=json.loads(raw['stage/server_inputs/p60_profile_and_refresh.json'])['profile'];q=profile['q'];aux=profile['p'];N=profile['degree'];d=profile['dnum']
        assert (N,d,len(q),len(aux))==(131072,8,100,14) and math.prod(q+aux).bit_length()<=3523
        alpha=(len(q)+d-1)//d;assert math.prod(aux)>=max(math.prod(q[i:i+alpha]) for i in range(0,len(q),alpha))
        rotations=json.loads(raw['stage/server_inputs/twelve_layer_plan.json'])['union_keys'];assert len(rotations)==130 and rotations==sorted(set(rotations))
        assert p['resource']['public_array_bytes']==125380329472 and p['resource']['private_array_bytes']==52428800
        assert p['resource']['public_required_bytes']+p['resource']['private_required_bytes']==126508630016
    assert r['source_archive_sha256']=='ff851d346502ed190798dcdfee5703c0d83ddbd929e878aa0ea83084a657ab00'
    assert not list(ROOT.rglob('*.npy')) and not list(ROOT.rglob('*.npz'))
    print(json.dumps(dict(passed=True,sources=58,tests=29,actual_key_generation=False,scope='public and mocked preflight only')))
if __name__=='__main__':main()
