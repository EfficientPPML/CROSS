"""Verify public proof/source bytes only; never follow private descriptors."""
from pathlib import Path
import argparse,hashlib,io,json,tarfile,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
CURRENT=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-successor-20261104')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def load(name):return json.loads((HERE/name).read_bytes())

def main(current=False):
    index=load('SHA256SUMS.json')
    paths={p.relative_to(HERE).as_posix():p for p in HERE.rglob('*')if p.is_file()}
    assert not any(p.is_symlink()for p in HERE.rglob('*'))
    assert set(paths)==set(index)|{'SHA256SUMS.json'}
    assert all(sha(paths[n].read_bytes())==pin for n,pin in index.items())
    raw=(HERE/'proof/source_manifest.json').read_bytes();expected=json.loads(raw)
    assert len(expected)==141 and sha(raw)=='45cd92473d9ba558288651d59d4f089c7fb9ba37650802848cad24687beaa437'
    rraw=(HERE/'proof/report.json').read_bytes();wraw=(HERE/'proof/worker/report.json').read_bytes();r=json.loads(rraw);w=json.loads(wraw)
    assert sha(rraw)=='c847b64d3cada0add4071fafe5c6ce79f2b70b0db6da7abcde41adf48f853ab9'
    assert sha(wraw)=='316b887ae7acdb1c00db66e927371542cfffcab3358e620a0a4b7d478f28a155'
    for row in(r,w):
        for key in('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'):assert row[key]is True
        assert row['source_hashes_start']==row['source_hashes_end']==expected and row['source_manifest_sha256']==sha(raw)
        assert row['audit_denials']==[]and 'error_type'not in row
    assert r['worker']==w and r['worker_report_sha256']==sha(wraw)
    assert type(r['child_returncode'])is int and r['child_returncode']==0 and r['child_reaped']is True and r['child_group_absent']is True
    assert r['actual_candidate_stage']is False and r['model_forward']is False
    assert w['tests']==781 and w['failures']==w['errors']==w['skipped']==w['pytest_returncode']==0
    assert w['installed_import_denials']==['torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla']
    xml=ET.parse(HERE/'proof/worker/tests.xml').getroot();assert len(list(xml.iter('testcase')))==781
    assert all(not list(xml.iter(name))for name in('failure','error','skipped'))
    archive=(HERE/'proof/sources.tar.gz').read_bytes();assert sha(archive)=='04edc1012ef8d503b878979c21cdeb997c2dd6e098a9407e8d59d9328ddb6c12'
    assert sha(archive)==r['source_archive']['sha256']and len(archive)==r['source_archive']['bytes']
    sources={};total=0
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
        for member in tar:
            assert member.isfile()and member.name in expected and member.name not in sources
            cap=(64 if member.name=='candidate_proof/selector-assessments.json'else 8)<<20
            assert 0<member.size<=cap
            total+=member.size;assert total<=96<<20
            data=tar.extractfile(member).read();assert sha(data)==expected[member.name];sources[member.name]=data
    assert set(sources)==set(expected)
    oldraw=(HERE/'parent139_sources.json').read_bytes();assert sha(oldraw)=='ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a'
    parent=json.loads(oldraw);assert len(parent)==139
    assert {n for n in parent if parent[n]!=expected[n]}=={'entry.py','preflight_expectations.json'}
    assert set(expected)-set(parent)=={'test_terminal_exit.py','TERMINAL_SUCCESSOR.md'}
    assert json.loads(sources['preflight_expectations.json'])['synthetic_tests']==781
    assert json.loads(sources['execution_authorization.json'])=={'execute':False}
    terminal=load('proof/terminal_observation.json')
    assert terminal['passed']is True and terminal['session']==29701 and terminal['exit_code']==0 and terminal['reaped']is True
    assert terminal['all_observed_pids_and_groups_absent']is True
    assert [(x['pid'],x['pgid'])for x in terminal['observed_owned_processes']]==[(2988850,2988850),(2988851,2988851)]
    old=load('history/identical-cli-predecessor43/report.json');new=load('history/focused-successor43/report.json')
    assert old['expected_boundary_failures']==36 and old['controls_passed']==7 and old['total_tests']==43
    assert new['passed']is True and new['tests']==43 and new['xml_bad']==0
    assert old['probe_sha256']==expected['test_terminal_exit.py']==new['source_hashes_start']['test_terminal_exit.py']
    for version in(1,2,3):
        negative=load(f'history/negative-late-close-v{version}/review.json')
        assert negative['finding_confirmed']is True and negative['actual_candidate_outcome_read']is False
        assert negative['expected_requirement_failures']==version
    if current:
        assert not any(p.is_symlink()for p in CURRENT.rglob('*'))
        assert {p.relative_to(CURRENT).as_posix():sha(p.read_bytes())for p in CURRENT.rglob('*')if p.is_file()}==expected
    print(json.dumps(dict(passed=True,indexed_public_files=len(index),source_count=141,parent137_unchanged=True,
        synthetic_tests=781,installed_import_denials=10,identical_probe_tests=43,current_checked=current,
        private_descriptors_followed=False),sort_keys=True))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',action='store_true');main(p.parse_args().current)
