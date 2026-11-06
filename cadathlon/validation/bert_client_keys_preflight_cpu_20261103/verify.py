"""Offline recorded key-helper evidence and exact public resource checks."""
from pathlib import Path
import hashlib,json,math,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    pins=json.loads((ROOT/'artifact_sha256.json').read_text());assert all(sha(ROOT/n)==h for n,h in pins.items())
    r=json.loads((ROOT/'report.json').read_text());assert r['passed'] and r['test_child_reaped'] and not r['cleanup_errors']
    assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==40
    assert r['source_files_unchanged'] and r['source_archive_verified']
    assert sha(ROOT/'sources.tar.gz')==r['source_archive_sha256']=='85dd7e6d1db17e6d380fb9e6b154c8d803331bb05fbd5c8404399314b6b0ec7f'
    with tarfile.open(ROOT/'sources.tar.gz') as t:
        rows=t.getmembers();assert len(rows)==len({x.name for x in rows})==40 and all(x.isfile() for x in rows)
        raw={x.name:t.extractfile(x).read() for x in rows}
        assert {n:hashlib.sha256(b).hexdigest() for n,b in raw.items()}==r['source_hashes_before']
    p=r['parameters'];q=p['q_towers'];aux=p['p_towers'];n=p['degree'];d=p['dnum']
    assert (n,d,len(q),len(aux))==(131072,8,100,14)
    assert math.prod(q+aux).bit_length()<=3523
    alpha=(len(q)+d-1)//d;assert math.prod(aux)>=max(math.prod(q[i:i+alpha]) for i in range(0,len(q),alpha))
    rotations=r['rotation_inventory'];assert len(rotations)==130 and rotations==sorted(set(rotations))
    assert rotations==json.loads(raw['report.json'])['union_keys']
    assert raw['report.json']==raw['server_inputs/twelve_layer_plan.json']
    parts=(len(q)+alpha-1)//alpha;public=(2*len(q)+(len(rotations)+1)*2*parts*(len(q)+len(aux)))*n*4;private=len(q)*n*4
    assert r['resources']['public_array_bytes']==public==125380329472
    assert r['resources']['private_array_bytes']==private==52428800
    assert r['tests']==dict(tests=42,failures=0,errors=0,skipped=0)
    assert sum(int(x.attrib['tests']) for x in ET.parse(ROOT/'tests.xml').getroot())==42
    assert r['actual_key_generation'] is False and r['encrypted_execution'] is False and r['fake_payloads_archived'] is False
    assert not list(ROOT.rglob('*.npy')) and not list(ROOT.rglob('*.npz'))
    print(json.dumps(dict(passed=True,sources=40,tests=42,rotation_count=130,scope='public consistency and recorded mocked preflight; no actual keygen')))
if __name__=='__main__':main()
