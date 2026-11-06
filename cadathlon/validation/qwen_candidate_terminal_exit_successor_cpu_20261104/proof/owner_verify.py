"""Public final source/terminal readback; never follows model/private descriptors."""
from pathlib import Path
import hashlib,json,os,tarfile,xml.etree.ElementTree as ET

ROOT=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-successor-20261104')
PROOF=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-preflight-v1-20261104')
MANIFEST=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-sources-v1-20261104.json')
LIVE=Path('/tmp/qwen-candidate-terminal-contract-review-20261104/full-preflight-live-observation.json')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
parse=lambda path:json.loads(path.read_bytes())

def main():
    raw=MANIFEST.read_bytes();assert sha(raw)=='45cd92473d9ba558288651d59d4f089c7fb9ba37650802848cad24687beaa437'
    expected=json.loads(raw);assert len(expected)==141
    parent=parse(PROOF/'report.json');worker=parse(PROOF/'worker/report.json')
    assert parent['worker']==worker and parent['worker_report_sha256']==sha((PROOF/'worker/report.json').read_bytes())
    for row in(parent,worker):
        for field in('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged',
                     'terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'):
            assert row[field]is True
        assert row['source_manifest_sha256']==sha(raw) and row['source_hashes_start']==row['source_hashes_end']==expected
        assert row['audit_denials']==[] and 'error_type'not in row
    assert parent['child_returncode']==0 and parent['child_reaped']is True and parent['child_group_absent']is True
    for root in(ROOT,PROOF/'execution_source'):
        paths={p.relative_to(root).as_posix():p for p in root.rglob('*')if p.is_file()}
        assert set(paths)==set(expected)
        assert all(not p.is_symlink()and sha(p.read_bytes())==expected[n]for n,p in paths.items())
    archive=PROOF/'sources.tar.gz';assert sha(archive.read_bytes())==parent['source_archive']['sha256']
    seen={}
    with tarfile.open(archive,'r:gz')as tar:
        for member in tar:
            assert member.isfile()and member.name in expected and member.name not in seen
            seen[member.name]=sha(tar.extractfile(member).read())
    assert seen==expected
    old=parse(Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json'))
    assert {n for n in old if old[n]!=expected[n]}=={'entry.py','preflight_expectations.json'}
    assert set(expected)-set(old)=={'test_terminal_exit.py','TERMINAL_SUCCESSOR.md'}
    xml=ET.parse(PROOF/'worker/tests.xml').getroot()
    assert len(list(xml.iter('testcase')))==worker['tests']==781
    assert all(not list(xml.iter(n))for n in('failure','error','skipped'))
    assert worker['installed_import_denials']==['torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla']
    live=parse(LIVE);observed=live['observed']
    assert observed[0]['pid']==parent['pid']==2988850 and observed[0]['pgid']==2988850
    assert observed[1]['pid']==parent['child_pid']==worker['pid']==2988851 and observed[1]['pgid']==parent['child_pgid']==2988851
    for row in observed:
        assert not Path('/proc',str(row['pid'])).exists()
        try:os.killpg(row['pgid'],0)
        except ProcessLookupError:pass
        else:raise AssertionError('recorded group remains')
    receipt=dict(schema='qwen-candidate-terminal-preflight-owner-v1',passed=True,session=29701,exit_code=0,reaped=True,
        authoritative_tool_chunk='797e97',source_count=141,tests=781,installed_denials=10,
        source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive.read_bytes()),
        report_sha256=sha((PROOF/'report.json').read_bytes()),worker_report_sha256=sha((PROOF/'worker/report.json').read_bytes()),
        observed_owned_processes=observed,all_observed_pids_and_groups_absent=True,
        wrapper_pid=None,wrapper_status=live['wrapper_status'],parent137_unchanged=True,numeric_six_unchanged=True,
        actual_candidate_outcome_read=False,actual_private_body_read=False,actual_model_execution=False,new_execution_grant=False)
    (PROOF/'terminal_observation.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:receipt[k]for k in('passed','source_count','tests','report_sha256','source_archive_sha256','worker_report_sha256')}))

if __name__=='__main__':main()
