"""Offline source/evidence verifier; no backend or external payload reads."""
from pathlib import Path
import ast,hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(name):return json.loads((ROOT/name).read_bytes())
def archive(name,pins):
    with tarfile.open(ROOT/name) as t:
        members=t.getmembers();assert len(members)==len({m.name for m in members})==len(pins)
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..'not in Path(m.name).parts and m.size<8<<20 for m in members)
        assert sum(m.size for m in members)<16<<20
        raw={m.name:t.extractfile(m).read() for m in members}
    assert {n:sha(v) for n,v in raw.items()}==pins
    return raw

def main():
    pins=read('artifact_hashes.json');names={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__'not in p.parts and p.name!='artifact_hashes.json'}
    assert set(pins)==names
    for name,digest in pins.items():assert sha((ROOT/name).read_bytes())==digest,name
    r=read('report.json');before=r['source_hashes_before']
    assert len(before)==r['source_count']==46 and before==r['source_hashes_after']==r['snapshot_hashes']
    for flag in ('passed','terminal_scope_completed','cleanup_completed','process_reaped','source_files_unchanged','source_archive_verified'):assert r[flag]is True
    assert r['cleanup_errors']==[]
    assert sha((ROOT/'sources.tar.gz').read_bytes())==r['source_archive_sha256']=='9220fd29d40cebe4a8f8fedc840bd6a37faf0d52a5cce7a03ea904db37f856e5'
    raw=archive('sources.tar.gz',before);origin=json.loads(raw['origins.json'])
    assert all(sha(raw[n])==h for n,h in origin['fixed_input_hashes'].items())
    assert sha(raw['grouped_rotation.py'])=='7e042d5bbb2e10cff2e223cbab19542ee1d95e1f4b21b8c3f44a3d7aeb3b594b'
    for p in (ROOT/'review').iterdir():assert p.read_bytes()==raw[p.name]
    old=read('parent/report.json');oldraw=archive('parent/sources.tar.gz',old['source_hashes_before'])
    assert old['passed']is True and len(oldraw)==45
    assert sha((ROOT/'parent/sources.tar.gz').read_bytes())==origin['successor']['parent_archive_sha256']=='8ca3b2fef3e2f685da7248dabf649f75d806b2cea70d963fd0c15abb6ffbd47f'
    assert set(raw)-set(oldraw)=={'test_backing_successor.py'}
    assert {n for n in oldraw if oldraw[n]!=raw[n]}=={'grouped_rotation.py','collect_public.py','origins.json','DESIGN.md'}
    negative=archive('negative-v1/sources.tar.gz',read('negative-v1/source_hashes.json'))
    assert negative['grouped_rotation.py']==oldraw['grouped_rotation.py']
    functions=lambda data:{n.name:ast.dump(n) for n in ast.parse(data).body if isinstance(n,ast.FunctionDef)}
    tests=functions(negative['test_backing_successor.py']);current=functions(raw['test_backing_successor.py'])
    assert len(tests)==4 and all(current[n]==body for n,body in tests.items())
    assert '4 failed' in (ROOT/'negative-v1/tests.log').read_text()
    counts=r['test_counts'];assert counts==dict(tests=58,failures=0,errors=0,skipped=0)
    suites=ET.parse(ROOT/'tests.xml').getroot().findall('testsuite')
    assert {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in counts}==counts
    parity=read('public-parity/report.json');assert parity==r['public_result'] and parity['passed'] is parity['no_context_keygen_modules_loaded'] is True
    assert [(x['N'],x['Q']) for x in parity['fixtures']]==[(n,q) for n in (16,256) for q in (12,14,26,28,52,100)]
    assert parity['cpu_affinity']==[168,169,170,171] and parity['total_comparisons']==372
    nonzero=0;groups=0
    for row in parity['fixtures']:
        assert row['input_unchanged']is row['retired']is True and row['P']==14
        assert [x['comparisons'] for x in row['checks']]==[11,15,5]
        for check in row['checks']:
            assert check['equal']is check['group_digit_kernel_wrappers_collected']is True
            count=sum(i%(row['N']//2)!=0 for i in check['requests']);nonzero+=count
            assert check['decompositions']==int(count>0);groups+=check['decompositions']
            assert check['result_bytes']==count*2*row['N']*row['Q']*4
    assert nonzero==258 and groups==30
    assert all(p['reaped']is True and p['returncode']==0 for p in r['processes'])
    t=read('terminal.json');assert t['session_id']==15818 and t['exit_code']==0 and t['reaped']is t['numeric_pids_absent']is True
    assert t['negative_repro_exit_code']==1 and t['negative_repro_failures']==4
    print(json.dumps(dict(passed=True,local_sources=46,mock_checks=58,public_comparisons=372,original_negative_repros=4,actual_secure_HE=False)))
if __name__=='__main__':main()
