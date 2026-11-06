"""Offline/current public-only verification; never instantiate a backend."""
from pathlib import Path
import argparse,hashlib,io,json,tarfile,xml.etree.ElementTree as E
HERE=Path(__file__).resolve().parent
PINS={'disabled':('42407ad135289bd0655aa3e0d912e6b8780952f0a4762b0c4f8eae07d8fb4aeb','7ef701f2052bba3da5b59c2c9394713940df63e337c1e90363dea970d6ed020f',162), 'ready':('6fbf898fbcb8a884e2f2c7f1e2308dfd04171be990bfde82a20200d20f00e19c','382933c982162e6d83c7d5cc366e59b7abfd674ef408efcc529c9ba13db912ce',163)}
def sha(b):return hashlib.sha256(b).hexdigest()
def read_json(path):return json.loads(path.read_bytes())
def unpack(raw,expected):
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as tar:
        rows=tar.getmembers();assert len(rows)==len({r.name for r in rows})==len(expected)
        assert all(r.isfile()and r.size<=8<<20 and not Path(r.name).is_absolute()and '..'not in Path(r.name).parts for r in rows)
        assert sum(r.size for r in rows)<=32<<20
        result={r.name:tar.extractfile(r).read()for r in rows}
    assert {n:sha(b)for n,b in result.items()}==expected
    return result

def verify(current=False):
    inventory=read_json(HERE/'artifact_hashes.json')
    actual={p.relative_to(HERE).as_posix()for p in HERE.rglob('*')if p.is_file()and p.name!='artifact_hashes.json'and '__pycache__'not in p.parts}
    assert set(inventory)==actual
    assert all(sha((HERE/n).read_bytes())==h for n,h in inventory.items())
    proof_pins=read_json(HERE/'proof_pins.json');sources={}
    for label,(report_sha,archive_sha,count)in PINS.items():
        folder=HERE/label;raw=(folder/'report.json').read_bytes();assert sha(raw)==report_sha
        report=json.loads(raw);assert report['execute']is False
        assert all(report[k]is True for k in ['passed','source_files_unchanged','source_archive_verified','terminal_scope_completed','termination_scope_completed','all_direct_children_reaped','all_owned_groups_empty','process_cleanup_completed','no_private_or_ciphertext_payload_read','no_encrypted_execution'])
        assert report['cleanup_errors']==[] and not report.get('error')
        expected=report['source_hashes_start'];assert len(expected)==count and expected==report['source_hashes_end']==read_json(folder/'source_manifest.json')
        blob=(folder/'sources.tar.gz').read_bytes();assert sha(blob)==archive_sha==report['source_archive_sha256']
        sources[label]=unpack(blob,expected)
        for name,num in [('mock_tests',415),('installed_denials',10)]:
            assert report[name]==dict(tests=num,exit_code=0,errors=0,failures=0,skipped=0)
            cases=E.parse(folder/(name+'.xml')).findall('.//testcase');assert len(cases)==num
            assert all(all(c.find(k)is None for k in ['failure','error','skipped'])for c in cases)
        assert len(report['children'])==2
        for row in report['children']:
            assert row['direct_returncode']==0 and type(row['direct_returncode'])is int
            assert all(row[k]is True for k in ['direct_child_reaped','group_empty_verified','cleanup_completed'])and row['cleanup_errors']==[]
        observation=read_json(folder/'terminal_observation.json');assert observation['authoritative_tool_exit_code']==0 and observation['authoritative_reaped']is True and observation['children_absent_at_readback']is True
        if current:
            root=Path(proof_pins[label]['lab']);snapshot=root/'source-bound-preflight-v1/execution_source'
            for n,h in expected.items():assert sha((root/n).read_bytes())==sha((snapshot/n).read_bytes())==h
            assert not any(Path('/proc',str(row['pid'])).exists()for row in report['children'])
    before,after=sources['disabled'],sources['ready'];origin=read_json(HERE/'ready_successor_origin.json')
    assert set(after)-set(before)=={'ready_successor_origin.json'}and set(before)-set(after)==set()
    changed={n for n in before if before[n]!=after[n]};assert changed==set(origin['changes'])=={'DESIGN.md','README.md','execution_authorization.json','gain4_component_overlays.json','source_inventory.json'}
    assert json.loads(before['execution_authorization.json'])=={'ready':False}and json.loads(after['execution_authorization.json'])=={'ready':True}
    for name,row in origin['changes'].items():assert sha(before[name])==row['before']and sha(after[name])==row['after']
    predecessor=json.loads(before['routing_predecessor/report.json']);old=unpack(before['routing_predecessor/sources.tar.gz'],predecessor['source_hashes_start'])
    retired='public_plan/inputs/full_key_plan.json';active='runtime/stage/server_inputs/twelve_layer_plan.json'
    assert retired not in before and old[retired]==before[active]==after[active]
    for source in [before,after]:
        plan=json.loads(source['proof_plan.json']);assert len(plan['checkpoints'])==54
        assert len(plan['cases'])==2 and plan['counts']['ciphertext_products']==48
        ideal=json.loads(source['gain4_ideal/report.json']);pinned=ideal['source_hashes_before'];unpack(source['gain4_ideal/sources.tar.gz'],pinned)
        assert len(pinned)==131 and ideal['passed']is True and pinned==ideal['source_hashes_after']
        original=json.loads(source['gain4_component_origin.json'])
        for name,h in original['new_gain4_files'].items():assert sha(source[name])==h
    return {'passed':True,'disabled_sources':162,'ready_sources':163,'tests_per_proof':425,'current':current,'actual_HE':False}
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--current',action='store_true');args=parser.parse_args();print(json.dumps(verify(args.current),sort_keys=True))
