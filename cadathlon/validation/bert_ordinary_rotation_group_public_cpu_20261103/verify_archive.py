"""Portable source/evidence checks only; no JAX, payload or external reads."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(name):return json.loads((ROOT/name).read_bytes())
def archive(name,pins):
    with tarfile.open(ROOT/name) as t:
        members=t.getmembers()
        assert len(members)==len(set(m.name for m in members))==len(pins)
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.size<8<<20 for m in members)
        assert sum(m.size for m in members)<16<<20
        raw={m.name:t.extractfile(m).read() for m in members}
    assert {n:sha(v) for n,v in raw.items()}==pins
    return raw

def main():
    pins=read('artifact_hashes.json')
    names={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__'not in p.parts and p.name!='artifact_hashes.json'}
    assert set(pins)==names
    for n,h in pins.items():assert sha((ROOT/n).read_bytes())==h,n
    report=read('report.json');sources=report['source_hashes_before']
    assert report['passed'] is report['source_files_unchanged'] is report['source_archive_verified'] is report['cleanup_completed'] is report['terminal_scope_completed'] is True
    assert len(sources)==report['source_count']==45
    assert sources==report['source_hashes_after']==report['snapshot_hashes']
    assert report['cleanup_errors']==[] and report['process_reaped'] is True
    assert sha((ROOT/'sources.tar.gz').read_bytes())==report['source_archive_sha256']=='8ca3b2fef3e2f685da7248dabf649f75d806b2cea70d963fd0c15abb6ffbd47f'
    raw=archive('sources.tar.gz',sources)
    origin=json.loads(raw['origins.json'])
    assert all(sha(raw[n])==h for n,h in origin['fixed_input_hashes'].items())
    assert origin['raw_seam_archive_sha256']=='2d8d4f9454f013d11c3ab2d7d2bf288834abf009b6b86c6616bebb9aa7c0c081'
    for p in (ROOT/'review').iterdir():assert raw[p.name]==p.read_bytes()
    summary=read('summary.json');assert sha(raw['grouped_rotation.py'])==summary['source_sha256']=='31f27dd528bb4b0ad0c44b05e4795b22ab01c862c70bae17a43c48108a8444fb'
    assert summary['actual_secure_HE'] is summary['runtime_promoted'] is False
    parity=read('public-parity/report.json');assert parity==report['public_result'] and parity['passed'] is parity['no_context_keygen_modules_loaded'] is True
    assert parity['cpu_affinity']==[168,169,170,171]
    assert [(r['N'],r['Q']) for r in parity['fixtures']]==[(n,q) for n in (16,256) for q in (12,14,26,28,52,100)]
    assert parity['total_comparisons']==sum(c['comparisons'] for row in parity['fixtures'] for c in row['checks'])==372
    nonzero=0;decompositions=0
    for row in parity['fixtures']:
        assert row['P']==14 and row['input_unchanged'] is row['retired'] is True
        assert [c['group'] for c in row['checks']]==['head11','key15','edges']
        assert [c['comparisons'] for c in row['checks']]==[11,15,5]
        for c in row['checks']:
            assert c['equal'] is c['group_digit_kernel_wrappers_collected'] is True and c['mode']=='digits'
            count=sum(i%(row['N']//2)!=0 for i in c['requests']);nonzero+=count
            assert c['decompositions']==int(count>0);decompositions+=c['decompositions']
            assert c['result_bytes']==count*2*row['N']*row['Q']*4
            expected=((row['Q']+12)//13)*row['N']*(row['Q']+14)*4 if count else 0
            assert c['digit_bytes']==expected
    assert summary['nonzero_completions']==nonzero==258 and summary['zero_input_alias_checks']==114
    assert summary['decomposition_groups']==decompositions==30
    counts=report['test_counts'];assert counts==dict(tests=51,failures=0,errors=0,skipped=0)
    suites=ET.parse(ROOT/'tests.xml').getroot().findall('testsuite')
    assert {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in counts}==counts
    assert len(report['processes'])==2 and all(p['reaped'] is True and p['returncode']==0 for p in report['processes'])
    terminal=read('terminal.json');assert terminal['session_id']==75497 and terminal['exit_code']==0 and terminal['reaped'] is terminal['all_numeric_pids_absent'] is True
    historical=archive('historical/development-final-v1-sources.tar.gz',read('historical/development-final-v1-source-hashes.json'))
    changes=[n for n in raw if raw[n]!=historical[n]]
    assert changes==['test_collector.py']
    assert raw['test_collector.py']==historical['test_collector.py'].replace(b"root/'check_public.py'",b"root/'check_public_group.py'")
    print(json.dumps(dict(passed=True,local_sources=45,mock_checks=51,public_comparisons=372,nonzero_completions=258,zero_aliases=114,actual_secure_HE=False)))
if __name__=='__main__':main()
