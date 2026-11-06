"""Offline source/report audit. No imports of runtime, model readers or arrays."""
from pathlib import Path
import argparse,hashlib,json,math,tarfile
EXPECTED_ARCHIVE='4b5e11f598d25dce089b97ec87640ef40752bfd390a05d1ec16e2ad4a605cfb4'
EXPECTED_OWNER='0e7c7e67b80628a2e0acfa4e4b7fc0b734a26926b037f45ddac6e10948300896'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def run(root):
    report=json.loads((root/'report.json').read_text());terminal=json.loads((root/'terminal.json').read_text())
    assert report['passed'] is True and report['complete'] is True and report['selection'] is None
    assert report['tests']==dict(tests=46,errors=0,failures=0,skipped=0) and report['test_exit_code']==0
    assert terminal['exec_session']==98208 and terminal['authoritative_exit_code']==0 and terminal['session_reaped'] is True
    assert report['cleanup_errors']==[] and report['process_cleanup']['reaped'] is True and report['process_cleanup']['group_gone'] is True
    before=report['source_hashes_before'];assert len(before)==137==report['source_count']
    assert before==report['source_hashes_after']==report['snapshot_source_hashes_after']
    assert report['source_files_unchanged'] is True and report['source_archive_verified'] is True
    assert report['asset_files_unchanged'] is True and report['asset_hashes_before']==report['asset_hashes_after']
    assert before['encrypted_bert_full_model.py']==EXPECTED_OWNER
    archive=root/'sources.tar.gz';assert digest(archive)==EXPECTED_ARCHIVE==report['source_archive_sha256']
    with tarfile.open(archive) as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})==137 and all(m.isfile() for m in members)
        assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members}==before
    assert len(report['cases'])==2 and {c['K'] for c in report['cases']}=={15,128}
    for case in report['cases']:
        K=case['K'];T=(K+15)//16
        assert case['passed'] is True and case['ideal_gate']==1e-7 and case['all_connections_are_actual_output_objects'] is True
        assert case['output_level']==2 and case['native_prediction_agrees'] is True
        assert case['native_logit_gate']=='abs(error)<=0.02+0.01*abs(native)'
        assert math.isfinite(case['native_logit_max_abs']) and case['native_logit_max_abs']<.02
        assert len(case['stage_errors'])==24 and {(x['layer'],x['phase']) for x in case['stage_errors']}=={(i,p) for i in range(12) for p in ('attention','block')}
        assert all(math.isfinite(x['max_abs']) and 0<=x['max_abs']<=1e-7 and x['physical_slots']==T*65536 for x in case['stage_errors'])
        assert all(math.isfinite(case[k]) and 0<=case[k]<=1e-7 for k in ('pooler_all_slot_max_abs','logits_all_slot_max_abs'))
        assert case['physical_stage_slots_checked']==(24*T+2)*65536
        assert case['primitive_counts']==case['public_operation_plan']['counts']
        assert case['identity_refresh_calls']==case['public_operation_plan']['identity_refreshes']
        assert case['public_operation_plan']['full_refresh_cost_included'] is False
        assert case['actual_domain_observations'] and all(x['passed'] is True and x['domain'][0]<=x['minimum']<=x['maximum']<=x['domain'][1] for x in case['actual_domain_observations'])
        assert json.loads((root/f'joined-K{K}.json').read_text())==case
    expected=json.loads((root/'artifact_hashes.json').read_text())
    assert {p.relative_to(root).as_posix():digest(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__'not in p.parts}==expected
    historical=json.loads((root/'historical_artifacts.json').read_text())
    assert all(digest(root/n)==h for n,h in historical.items())
    return dict(passed=True,source_count=137,tests=46,cases=[15,128],scope='Offline exact-source and reported ideal-gate audit; no encrypted execution or ranked-model claim.')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path,nargs='?',default=Path(__file__).resolve().parent);args=parser.parse_args()
    print(json.dumps(run(args.archive),indent=2))
