"""Portable public evidence verification; never import measured runtime."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_bytes())
def need(v,m):
    if not v:raise ValueError(m)
def tarbytes(path):
    with tarfile.open(path,'r:gz')as t:
        items=t.getmembers();need(all(x.isfile()for x in items),'Regular source members')
        result={x.name:t.extractfile(x).read()for x in items};need(len(result)==len(items),'Unique source members');return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--current',action='store_true');a=ap.parse_args()
    index=load(ROOT/'artifact_hashes.json');actual={str(p.relative_to(ROOT))for p in ROOT.rglob('*')if p.is_file()and p.name!='artifact_hashes.json'and '__pycache__'not in p.parts}
    need(actual==set(index),'Exact artifact inventory')
    for n,h in index.items():need(sha(ROOT/n)==h,'Artifact '+n)
    p=ROOT/'preflight171';m=load(p/'source_manifest.json');b=tarbytes(p/'sources.tar.gz');need(len(m)==171,'171 sources');need({n:hashlib.sha256(v).hexdigest()for n,v in b.items()}==m,'Exact tar')
    r=load(p/'report.json');w=load(p/'worker/report.json');o=load(p/'owner-terminal.json');need(r['worker']==w and r['worker_report_sha256']==sha(p/'worker/report.json'),'Worker link')
    need(r['source_archive']['sha256']==sha(p/'sources.tar.gz'),'Tar link')
    for d in (r,w):
        need(d['source_manifest_sha256']==sha(p/'source_manifest.json'),'Manifest link')
        need(d['source_hashes_start']==m==d['source_hashes_end'],'Source start/end')
        for k in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'):need(d[k]is True,k)
        need(d['audit_denials']==[]and d['model_forward']is False,'No audit/model')
    need(r['actual_heldout_stage']is False and w['validation_inputs_read']is False,'No actual workload')
    need(r['child_returncode']==0 and r['child_reaped']is r['child_group_absent']is True,'Child closure')
    need(w['tests']==563 and w['errors']==w['failures']==w['skipped']==0,'563 tests')
    need(w['installed_import_denials']==['torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla'],'Ten installed denials')
    x=ET.parse(p/'worker/tests.xml').getroot();cases=x.findall('.//testcase');need(len(cases)==563 and not any(x.findall('.//'+n)for n in ('error','failure','skipped')),'XML checks')
    need(len([c for c in cases if c.get('classname','').endswith('test_heldout_terminal_exit')])==85,'85 shared/provenance tests included')
    need(o['session']==97252 and o['exit_code']==0 and o['reaped']is True,'Authoritative owner reap')
    need(o['collector']=={'pid':2992574,'pgid':2992574,'absent':True}and o['worker']=={'pid':2992576,'pgid':2992576,'absent':True},'Actual observed groups')
    for k,f in (('report_sha256','report.json'),('worker_report_sha256','worker/report.json'),('source_archive_sha256','sources.tar.gz')):need(o[k]==sha(p/f),'Owner '+k)
    origin=load(ROOT/'heldout_terminal_origin.json');need(all(m[n]==h for n,h in origin['inherited139_sources'].items()),'All139 unchanged')
    changes=[n for n,h in origin['parent167_sources'].items()if m[n]!=h]
    need(changes==['heldout_entry.py','heldout_preflight_expectations.json','test_heldout_no_grant.py'],'Three declared old-file overlays')
    need(json.loads(b['execution_authorization.json'])=={'execute':False},'Disabled actual execution')
    for sub,v in (('draft171-85','v3'),('testassert171-6','v4')):
        hroot=ROOT/'history'/sub;hm=load(hroot/'source_manifest.json');scope=load(hroot/'scope.json');rebuilt=dict(b)
        for n in scope['exact_overlays']:rebuilt[n]=(hroot/'overlays'/n).read_bytes()
        need({n:hashlib.sha256(rebuilt[n]).hexdigest()for n in hm}==hm,'Exact development source reconstruction')
        tx=ET.parse(hroot/'tests.xml').getroot();need(len(tx.findall('.//testcase'))==scope['tests']and not any(tx.findall('.//'+n)for n in ('error','failure','skipped')),'Development test scope')
        if a.current:
            old=Path('/tmp/qwen-heldout1088-runtime-'+v+'-20261104');need({n:sha(old/n)for n in hm}==hm,'Current earlier tree')
    n=ROOT/'negative167';nm=load(n/'before.json');nb=tarbytes(n/'sources.tar.gz');need({k:hashlib.sha256(v).hexdigest()for k,v in nb.items()}==nm==load(n/'after.json'),'Negative exact sources')
    nx=ET.parse(n/'tests.xml').getroot();need(len(nx.findall('.//testcase'))==4 and len(nx.findall('.//failure'))==2 and not nx.findall('.//error'),'Two negatives plus two controls')
    need(load(n/'systemexit1-receipt.json')['stderr_bytes']==0,'Silent injected exit1 retained')
    if a.current:
        for base in (Path('/tmp/qwen-heldout1088-runtime-v5-20261104'),Path('/tmp/qwen-heldout1088-runtime-preflight-v5-20261104/execution_source')):need({n:sha(base/n)for n in m}==m,'Current final original/execution')
    print(json.dumps(dict(passed=True,public_artifacts=len(index),sources=171,tests=563,installed_denials=10,inherited139_unchanged=True,current_readback=a.current,actual_execution=False),sort_keys=True))
if __name__=='__main__':main()
