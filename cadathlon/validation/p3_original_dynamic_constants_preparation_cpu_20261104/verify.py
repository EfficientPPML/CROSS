"""Portable public-byte verifier; no runtime import or test execution."""
from pathlib import Path
import argparse
import ast
import difflib
import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
REPORT='686ed1945b4223db8e2eda125ca102213a1bbe965713f9a9a1921d37735e3487'
TAR='91f0faf4fc40c9955a5184efa42b363c7fd9d29b8aefe9fb074a60458c04bf69'
CANDIDATE='b93c89318086138b82c7b2d30cb525390a431b17253a0dfcc301c6f88caafeb1'
ORIGINAL='f7e1b4f2261d5ebf8223fe3cbdf206018bdf7c5a8fd8e394a67ea909e0659d29'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(name):return json.loads((HERE/name).read_bytes())

def verify(current=None):
    files=read('artifact_hashes.json')
    assert set(files)=={p.name for p in HERE.iterdir() if p.is_file() and p.name!='artifact_hashes.json'}
    assert all(not(HERE/n).is_symlink() and sha((HERE/n).read_bytes())==h for n,h in files.items())
    before=read('sources-before.json');after=read('sources-after.json')
    assert before==after and len(before)==100
    packed=(HERE/'sources.tar.gz').read_bytes();assert sha(packed)==TAR
    with tarfile.open(fileobj=io.BytesIO(packed),mode='r:gz') as handle:
        members=handle.getmembers()
        assert len(members)==100 and len({m.name for m in members})==100
        assert all(m.isfile() and m.size<=8*1024**2 for m in members)
        assert sum(m.size for m in members)<=32*1024**2
        raw={m.name:handle.extractfile(m).read() for m in members}
    assert {n:sha(b) for n,b in raw.items()}==before
    origins=read('copied_source_origins.json');assert len(origins)==96
    assert all(before[n]==v['sha256'] for n,v in origins.items())
    old=(HERE/'baseline_original.py').read_bytes();new=(HERE/'baseline_dynamic_constants.py').read_bytes()
    assert sha(old)==ORIGINAL and sha(new)==CANDIDATE
    assert old==raw['checkout/cadathlon/non_agent_pset/p3_cnn/src/baseline.py']
    assert new==raw['baseline_dynamic_constants.py']
    a=ast.parse(old);b=ast.parse(new)
    calls=[n for n in ast.walk(b) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Mapping']
    assert len(calls)==1 and calls[0].keywords[-1].arg=='dynamic_constants'
    assert ast.literal_eval(calls[0].keywords[-1].value) is True
    calls[0].keywords.pop()
    assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    patch=''.join(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='baseline_original.py',tofile='baseline_dynamic_constants.py'))
    assert (HERE/'candidate.patch').read_text()==patch
    report_raw=(HERE/'report.json').read_bytes();assert sha(report_raw)==REPORT
    report=json.loads(report_raw);terminal=read('owner-terminal.json')
    assert report['passed'] is True and report['pytest_exit_code']==0 and report['tests_collected']==10
    assert report['cpu_affinity']==[120,121,122,123] and report['backend']=='cpu' and report['device_platforms']==['cpu']
    assert report['cryptographic_construction_attempts']==[]
    assert report['optional_accelerator_import_denials']==['libtpu']
    assert report['calibrated_speed_claim'] is False and report['actual_original_profile_comparison'] is False
    records=report['records'];assert len(records)==6 and [r['case'] for r in records]==[f'test{i}' for i in range(1,7)]
    assert [r['batch'] for r in records]==[1,2,4,1,2,4]
    assert [r['model'] for r in records]==['lenet']*3+['alexnet_tiny']*3
    for row in records:
        assert all(row[k] is True for k in ('public_clear_fixture_matches','weights_unchanged','packing_unchanged','mapping_is_recording_stub'))
        ring=row['ring'];assert (ring['degree'],ring['dnum'],ring['depth'],len(ring['q_towers']),len(ring['p_towers']))==(32768,4,7,16,4)
        assert len(row['frontend_digest'])==len(row['packing_fingerprint'])==64
    xml=list(ET.parse(HERE/'tests.xml').getroot().iter('testcase'))
    assert len(xml)==10 and all(x.find('failure') is None and x.find('error') is None and x.find('skipped') is None for x in xml)
    names=[x.attrib['name'] for x in xml]
    assert names.count('test_exact_one_keyword_delta')==1
    assert sum(n.startswith('test_original_six_frontends_and_packings_are_identical[') for n in names)==6
    assert set(n for n in names if not n.startswith('test_original_six_') and n!='test_exact_one_keyword_delta')=={
        'test_concurrent_cleanup_defers_global_cache_clear_until_joined','test_cache_deferral_flushes_after_exception','test_resident_constants_and_multiple_inputs_match_direct_program'}
    assert (terminal['session_id'],terminal['tool_chunk'],terminal['exit_code'])==(35135,'140f65',0)
    assert all(terminal[k] is True for k in ('passed','reaped','pid_absent','group_absent','source_before_after_tar_equal','original_current_source_bytes_unchanged'))
    assert terminal['report_sha256']==REPORT and terminal['source_archive_sha256']==TAR and terminal['candidate_sha256']==CANDIDATE
    assert (terminal['source_count'],terminal['copied_original_source_count'],terminal['tests'],terminal['public_frontend_packing_cases'],terminal['unchanged_existing_dynamic_constant_tests'],terminal['exact_ast_delta_checks'])==(100,96,10,6,3,1)
    if current is not None:
        current=Path(current)
        actual={str(p.relative_to(current)):sha(p.read_bytes()) for p in current.rglob('*') if p.is_file()}
        assert actual==before
    return dict(passed=True,sources=100,tests=10,frontend_packing_cases=6,existing_clear_compilation_tests=3,one_keyword_delta_checks=1,current_checked=current is not None,no_runtime_import=True,no_he_or_tpu_result=True,no_timing_gain_claim=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--current',type=Path)
    args=parser.parse_args();print(json.dumps(verify(args.current),sort_keys=True))
