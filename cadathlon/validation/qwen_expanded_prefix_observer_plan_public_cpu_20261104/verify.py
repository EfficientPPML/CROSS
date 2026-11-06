"""Portable public readback; never reads model/token bodies or executes archived code."""
from pathlib import Path
import argparse,hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
REPORT='913b73c7ee89f76224562b2e9f713380c97236048f691d6d853bc9dda3af120c'
ARCHIVE='ef91719157c2443dd7a59a1a564c070666a18461cd5c7a13675c3ca1851c04ae'
PLAN='0d6d6df8874997c9f1f1905201441f771e6233d8c81fad4171ecc5f3170fb935'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def verify(current=None):
    assert sha(P/'report.json')==REPORT and sha(P/'sources.tar.gz')==ARCHIVE and sha(P/'plan.json')==PLAN
    manifest=json.loads((P/'artifact_hashes.json').read_bytes())
    assert all(sha(P/n)==h for n,h in manifest.items())
    r=json.loads((P/'report.json').read_bytes());expected=r['source_hashes_before']
    assert len(expected)==38 and expected==r['source_hashes_after']==r['execution_source_hashes_before']==r['execution_source_hashes_after']
    for name in ('passed','complete','source_files_unchanged','execution_files_unchanged','cleanup_completed','terminal_scope_completed','termination_scope_completed','zero_audited_private_model_body_reads'):
        assert r[name] is True,name
    assert r['HE_executed'] is False and r['model_forward'] is False
    assert r['source_archive_sha256']==ARCHIVE
    assert r['tests']==dict(tests=62,errors=0,failures=0,skipped=0)
    suites=ET.parse(P/'tests.xml').getroot().findall('testsuite')
    assert {n:sum(int(s.get(n,0)) for s in suites) for n in ('tests','errors','failures','skipped')}==r['tests']
    with tarfile.open(P/'sources.tar.gz','r:gz') as tar:
        members=tar.getmembers()
        assert len(members)==len({m.name for m in members})==38 and {m.name for m in members}==set(expected)
        assert all(m.isfile() and 0<=m.size<=8<<20 and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert sum(m.size for m in members)<16<<20
        assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members}==expected
    for name in ('audit.json','plan-audit.json'):
        audit=json.loads((P/name).read_bytes());assert audit['passed'] is True and audit['denied_attempts']==[] and audit['backend_modules']==[]
    assert len(r['children'])==2
    for child in r['children']:
        assert type(child['direct_returncode']) is int and child['direct_returncode']==0
        assert all(child[n] is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
        assert child['cleanup_errors']==[]
    terminal=json.loads((P/'terminal.json').read_bytes())
    assert terminal['session_id']==56658 and terminal['authoritative_tool_exit_code']==0 and terminal['authoritative_tool_reaped'] is True
    assert terminal['children']==r['children'] and all(x is True for x in terminal['independent_absence'].values())
    assert terminal['collector_pid_recorded'] is False and terminal['collector_pid'] is None
    plan=json.loads((P/'plan.json').read_bytes())
    assert plan['passed'] is True and plan['execution_authorized'] is False
    assert plan['workload']['entries']==3200 and plan['workload']['original416_members']==416
    assert plan['initial_generator']['new_candidate_budget_consumed']==0 and plan['initial_generator']['maximum_forward_contexts']==64
    assert plan['initial_selection_requires_complete_native_exact3200'] is True
    assert plan['native_exact_and_baseline_ranges_must_remain_separate'] is True
    assert all(plan[n] is False for n in ('calibration_or_validation_bodies_read','model_or_tokenizer_executed','coefficient_selection_performed'))
    if current is not None:
        assert {n:sha(Path(current)/n) for n in expected}==expected
    return dict(passed=True,sources=38,tests=62,current_checked=current is not None,scope='public/synthetic only; no observer/model/HE execution')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--current',type=Path);args=parser.parse_args()
    print(json.dumps(verify(args.current),sort_keys=True))
