"""Portable public evidence verification. Does not import producer code."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import tarfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(name):return (ROOT/name).read_bytes()
def load(name):return json.loads(read(name))
def require(value,label):
    if not value:raise ValueError(label)

def proof(version,count,tests,failures):
    prefix=f'preflight-v{version}/';parent=load(prefix+'report.json');worker=load(prefix+'worker/report.json')
    expected=load(prefix+'source_manifest.json');owner=load(prefix+'owner_terminal.json')
    require(len(expected)==count,'source count')
    raw=read(prefix+'sources.tar.gz');found={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as archive:
        for member in archive:
            require(member.isfile() and member.name in expected and member.name not in found and 0<member.size<=8<<20,'source member')
            found[member.name]=archive.extractfile(member).read()
    require({name:sha(data)for name,data in found.items()}==expected,'exact source archive')
    require(parent['source_archive']['sha256']==sha(raw)==owner['source_archive_sha256'],'source archive binding')
    require(parent['worker_report_sha256']==sha(read(prefix+'worker/report.json')) and parent['worker']==worker,'worker binding')
    require(owner['report_sha256']==sha(read(prefix+'report.json')),'owner report binding')
    for row in (parent,worker):
        require(row['passed'] is (failures==0) and row['complete'] is True,'scientific outcome')
        for key in ('cleanup_completed','signal_restoration_completed','source_files_unchanged',
            'terminal_after_body_cleanup_and_signal_restoration_attempts',
            'acceptance_requires_independent_exit0_reap_and_exact_report_hash'):
            require(row[key] is True,key)
        require(row['source_hashes_start']==row['source_hashes_end']==expected and row['audit_denials']==[],'source/audit completion')
    require(type(parent['child_returncode'])is int and parent['child_returncode']==(1 if failures else 0),'child return code')
    require(parent['child_reaped'] is True and parent['child_group_absent'] is True,'direct wait and group absence')
    require(owner['exit_code']==parent['child_returncode'] and owner['reaped'] is True and owner['producer_pids_absent'] is True,'outer terminal evidence')
    require(worker['tests']==tests and worker['failures']==failures and worker['errors']==worker['skipped']==0,'exact test totals')
    require(set(worker['installed_import_denials'])=={'torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla'},'ten denied imports')
    tree=ET.fromstring(read(prefix+'worker/tests.xml'))
    require(len(tree.findall('.//testcase'))==tests and len(tree.findall('.//failure'))==failures and not tree.findall('.//error') and not tree.findall('.//skipped'),'XML totals')
    require(parent['actual_candidate_stage'] is False and parent['model_forward'] is False,'public preparation only')
    require(json.loads(found['candidate_binding.json'])['ready'] is False and json.loads(found['execution_authorization.json'])=={'execute':False},'pending admission/no grant')
    return expected,found

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--current',type=Path);args=parser.parse_args()
    pins=load('artifact_hashes.json')
    actual={str(path.relative_to(ROOT)):sha(path.read_bytes())for path in ROOT.rglob('*')if path.is_file() and path.name!='artifact_hashes.json'}
    require(actual==pins,'exact public artifact inventory/hashes')
    old,_=proof(1,113,686,3);expected,found=proof(2,115,696,0)
    for name in ('negative-v1','successor-v1'):
        prefix='shield-origin/'+name+'/';report=load(prefix+'report.json')
        require(report['sources_before']==report['sources_after'],'shield probe source equality')
        require(report['archive_sha256']==sha(read(prefix+'sources.tar.gz')),'shield probe archive')
        require([r['passed']for r in report['checks']]==([False,False,True]if name=='negative-v1'else[True]*3),'preserved shield outcomes')
    for file in ('reviews/transformer-v2.json','numeric-development/handoff.json'):
        receipt=load(file)
        for name,pin in receipt['sources'].items():require(expected[name]==pin,'reviewed source binding: '+name)
    require(json.loads(found['preflight_expectations.json'])['excluded_tests']==['test_entry.py'],'declared legacy test exclusion')
    overlay=json.loads(found['observer61_overlay.json'])
    require(len(overlay['unchanged_sources'])==59 and set(overlay['overlays'])=={'entry.py','preflight_expectations.json'},'original61 delta')
    for name,pin in overlay['unchanged_sources'].items():require(expected[name]==pin,'unchanged observer source')
    for name,row in overlay['overlays'].items():require(expected[name]==row['checker_sha256'],'declared observer overlay')
    if args.current:
        current={str(path.relative_to(args.current)):sha(path.read_bytes())for path in args.current.rglob('*')if path.is_file()}
        require(current==expected,'current exact115 sources')
    print('PASS: 115 sources, 696 synthetic tests, 10 import denials; pending candidate admission. Preserved 113-source failure and shield negatives verified.')

if __name__=='__main__':main()
