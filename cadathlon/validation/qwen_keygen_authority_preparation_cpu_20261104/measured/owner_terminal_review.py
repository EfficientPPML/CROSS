"""Owner readback of public proof only; does not import or rerun tested sources."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import tarfile
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
sha = lambda b: hashlib.sha256(b).hexdigest()
pretty = lambda x: (json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def inventory(root):
    paths = sorted(root.rglob('*'))
    assert all(not p.is_symlink() and (p.is_file() or p.is_dir()) for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in paths if p.is_file()}


def absent(pid, *, group=False):
    try:
        (os.killpg if group else os.kill)(pid,0)
    except ProcessLookupError:
        return True
    return False


def main():
    config_raw=(HERE/'run_config.json').read_bytes()
    assert sha(config_raw)=='ea619c255bac1009a4622a5c4c0905e4c5cb1c09cfe72c5aec1a0f5f85e1a25b'
    config=json.loads(config_raw)
    report_raw=(HERE/'report.json').read_bytes();report=json.loads(report_raw)
    audit=json.loads((HERE/'worker-audit.json').read_bytes())
    live=json.loads((HERE/'live-observation.json').read_bytes())
    raw=(HERE/'sources-before.json').read_bytes();expected=json.loads(raw)
    assert sha(raw)=='3b2ddbcde655dc3485a36b160ed97f166ef6aad69512dd824bc409545a13db48'
    assert raw==(json.dumps(expected,sort_keys=True,separators=(',',':'))+'\n').encode()
    assert len(expected)==39
    assert inventory(Path(config['source_root']))==inventory(HERE/'execution')==expected
    assert json.loads((HERE/'sources-after.json').read_bytes())==expected
    assert json.loads((HERE/'execution-after.json').read_bytes())==expected
    archive=HERE/'sources.tar.gz'
    assert sha(archive.read_bytes())=='532c9f51972f851f535f373115022876fa2553af12f21ed84617de83e0f877b2'
    with tarfile.open(archive) as tar:
        members=tar.getmembers()
        assert len(members)==len({m.name for m in members})==39
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert {m.name:sha(tar.extractfile(m).read()) for m in members}==expected
    parent=json.loads((HERE/'execution/parent28_sources.json').read_bytes())
    assert len(parent)==28 and all(expected[n]==h for n,h in parent.items())
    assert inventory(Path('/tmp/qwen-keygen-authority-boundary-20261104'))==parent
    xml=ET.parse(HERE/'tests.xml').getroot()
    cases=[dict(classname=n.attrib['classname'],name=n.attrib['name']) for n in xml.iter('testcase')]
    counts={name:len(list(xml.iter(tag))) for name,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
    assert counts==dict(tests=48,failures=0,errors=0,skipped=0)==report['tests']
    assert cases==config['selected_xml_cases']==report['selected_xml_cases']
    assert audit['collected_nodes']==config['selected_nodes']==report['selected_nodes']
    assert len(audit['collected_nodes'])==len(set(audit['collected_nodes']))==48
    assert audit['denied']==audit['loaded_backend']==[] and audit['numpy_imported']is False
    assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']=={}
    assert audit['forbidden_roots']==config['forbidden_roots'] and 'numpy'in audit['forbidden_roots']
    assert audit['affinity']==[112,113,114,115]
    assert audit['pid']==live['worker_pid']==43440 and audit['pgid']==live['worker_pgid']==43440
    assert live['outer_pid']==live['outer_pgid']==43435
    assert type(audit['pytest_exit_code'])is int and audit['pytest_exit_code']==0
    assert report['passed']is True and report['sources_unchanged']is True
    assert report['child_returncode']==0 and report['child_reaped']is True and report['child_group_absent']is True
    assert all(report[n]is False for n in ('actual_execution_authorized','actual_key_generation','backend_context_construction','model_execution','encrypted_execution','numpy_imported'))
    assert report['synthetic_only']is True
    assert sha((HERE/'executed_driver.py').read_bytes())==report['driver_sha256']=='c9dba84612a97028575307cf5ff123ad812ebd3438881536a63b559e5e270d6f'
    assert sha((HERE/'worker_driver.py').read_bytes())==report['worker_driver_sha256']=='b2ec88e0d56252c3dd290d3e563d50875ab0afac3da81768f82172cc3b6b31f1'
    states={str(pid):dict(pid_absent=absent(pid),group_absent=absent(pid,group=True)) for pid in (43435,43440)}
    assert all(all(row.values())for row in states.values())
    race=dict(schema='qwen-pure-authority-owner-observation-race-v1',tool_chunk='223595',exit_code=1,
        action='Second optional live ps capture after the earlier successful d56d20 external observation',
        observed_error='subprocess.CalledProcessError: ps -p 43435,43440 returned 1 because both processes had already exited',
        owner_live_file_written=False,source_or_proof_mutated=False,proof_test_failure=False,
        initial_live_observation_tool='d56d20',authoritative_owner_terminal_tool='c646b3')
    (HERE/'late-live-observation-race.json').write_bytes(pretty(race))
    evidence=['report.json','tests.xml','tests.log','worker-audit.json','live-observation.json',
              'sources-before.json','sources-after.json','execution-after.json','sources.tar.gz',
              'executed_driver.py','worker_driver.py','run_config.json','launch-proposal.json',
              'source-review.json','late-live-observation-race.json','owner_terminal_review.py']
    result=dict(schema='qwen-keygen-pure-authority-owner-terminal-v1',passed=True,
        checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),owner_session=22566,
        owner_launch_chunk='4fdb05',owner_terminal_chunk='c646b3',owner_tool_exit_code=0,owner_reaped=True,
        first_external_live_observation_chunk='d56d20',first_external_live_observation=dict(
            outer_pid=43435,outer_pgid=43435,worker_pid=43440,worker_pgid=43440,both_active=True,worker_cpu_percent=98.2),
        process_states=states,child_direct_returncode=0,child_reaped=True,tests=counts,
        elapsed_s=report['elapsed_s'],source_count=39,all_parent28_unchanged=True,
        all_current_execution_start_end_archive_equal=True,exact_collection_and_xml_order=True,
        no_backend_or_numpy_import=True,audit_denials=[],actual_authority=False,
        scope='Public synthetic byte-gate proof only; no genuine ModelAdmission, body integrity, generation, RNG or role authorization',
        files={n:sha((HERE/n).read_bytes()) for n in evidence})
    raw=pretty(result)
    with (HERE/'owner-terminal.json').open('xb') as f:f.write(raw)
    handoff=dict(schema='qwen-keygen-pure-authority-terminal-handoff-v1',proof_root=str(HERE),
        source_map_sha256=sha((HERE/'sources-before.json').read_bytes()),
        source_archive_sha256=sha(archive.read_bytes()),report_sha256=sha(report_raw),
        owner_receipt_sha256=sha(raw),owner_receipt='owner-terminal.json',
        owner_script_sha256=sha(Path(__file__).read_bytes()),source_count=39,tests=48,passed=True,
        owner_session=22566,owner_terminal_chunk='c646b3',exit_code=0,reaped=True,
        no_actual_role_authorized=True,root_independent_terminal_review_pending=True)
    with (HERE/'terminal-handoff.json').open('xb') as f:f.write(pretty(handoff))
    print(json.dumps(dict(report_sha256=sha(report_raw),owner_sha256=sha(raw),handoff_sha256=sha(pretty(handoff)),
        source_count=39,tests=counts,elapsed_s=report['elapsed_s'],process_states=states),sort_keys=True))


if __name__=='__main__':main()
