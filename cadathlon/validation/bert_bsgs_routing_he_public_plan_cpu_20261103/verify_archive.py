"""Portable public evidence verification; never reads original private paths."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,runpy,tarfile,tempfile,xml.etree.ElementTree as ET

def sha(raw):return hashlib.sha256(raw).hexdigest()

def verify(root,recompute=False):
    root=Path(root);pins=json.loads((root/'archive_manifest.json').read_text())
    assert {p.name for p in root.iterdir() if p.is_file()}==set(pins)|{'archive_manifest.json'}
    for n,d in pins.items():assert sha((root/n).read_bytes())==d,n
    report=json.loads((root/'report.json').read_bytes());plan=json.loads((root/'plan.json').read_bytes())
    assert report['passed']is True and report['source_files_unchanged']is True and report['source_archive_verified']is True
    expected=report['source_hashes_before'];assert len(expected)==8 and expected==report['source_hashes_after']
    assert expected==json.loads((root/'source_hashes_before.json').read_bytes())
    raw=(root/'sources.tar.gz').read_bytes();assert sha(raw)==report['source_archive_sha256']
    with tarfile.open(root/'sources.tar.gz') as archive:
        members=archive.getmembers();assert len(members)==len({m.name for m in members})==8
        assert all(m.isfile() and 0<m.size<=8<<20 and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members)
        assert sum(m.size for m in members)<=16<<20
        bodies={m.name:archive.extractfile(m).read() for m in members}
    assert {n:sha(v) for n,v in bodies.items()}==expected
    origins=json.loads((root/'input_sources.json').read_bytes());assert len(origins)==4
    assert json.loads(bodies['input_sources.json'])==origins
    for n,d in origins.items():assert sha(bodies['inputs/'+n])==d['sha256']
    assert sha((root/'plan.json').read_bytes())==report['plan_sha256']
    cases=ET.parse(root/'tests.xml').findall('.//testcase')
    assert len(cases)==report['tests']==25 and report['pytest_exit_code']==0
    assert all(c.find(n)is None for c in cases for n in ('failure','error','skipped'))
    terminal=json.loads((root/'terminal_observation.json').read_bytes())
    assert terminal['test_exit_code']==terminal['readback_tool_exit_code']==0
    assert terminal['test_session_id']is None and terminal['test_pid']is None and terminal['test_process_reaped']is None
    assert plan['checkpoint_count']==18 and plan['compared_physical_slots']==1179648
    assert plan['route_totals']==dict(rotations=72,public_products=168,ciphertext_additions=156,composite_rescales=42,critical_public_levels=12)
    assert plan['storage']['total_stored_payload_bytes']==348*(1<<20)
    if recompute:
        with tempfile.TemporaryDirectory(prefix='public-bsgs-plan-verify-') as directory:
            base=Path(directory)
            for name,body in bodies.items():
                path=base/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
            namespace=runpy.run_path(str(base/'plan.py'),run_name='verified_public_plan')
            assert namespace['build_plan']()==plan
    return dict(passed=True,sources=8,tests=25,checkpoints=18,public_plan_recomputed=recompute,
                no_encrypted_execution=True,separate_process_pid_and_reap_history='not captured')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recompute',action='store_true');args=parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parent,args.recompute),sort_keys=True))
