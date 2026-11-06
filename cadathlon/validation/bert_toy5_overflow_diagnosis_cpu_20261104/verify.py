"""Verify public failure/diagnostic evidence without executing decoder code."""
import argparse,hashlib,json,math,tarfile
from pathlib import Path
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_bytes())
def need(v,m):
    if not v:raise ValueError(m)
def contents(p):
    with tarfile.open(p,'r:gz')as t:
        ms=t.getmembers();need(all(x.isfile()for x in ms),'Regular source members')
        out={x.name:t.extractfile(x).read()for x in ms};need(len(out)==len(ms),'Unique inventory');return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--current',action='store_true');a=ap.parse_args()
    idx=load(ROOT/'artifact_hashes.json');actual={str(p.relative_to(ROOT))for p in ROOT.rglob('*')if p.is_file()and p.name!='artifact_hashes.json'and '__pycache__'not in p.parts};need(actual==set(idx),'Exact public inventory')
    for n,h in idx.items():need(sha(ROOT/n)==h,'Artifact '+n)
    p=ROOT/'original-failure';proc=load(p/'process.json');num=load(p/'stdout.json');device=load(p/'security_device.json');saved=load(p/'preservation.json')
    need(proc['passed']is num['passed']is device['passed']is False,'Original failure preserved')
    need(proc['returncode']==1 and num['error_type']=='OverflowError'and num['error']=='int too large to convert to float','Original error unchanged')
    need(proc['result']==num,'Numeric/process link')
    for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed','source_files_unchanged'):need(proc[n]is True,'Recorded cleanup '+n)
    for n,h in saved['case_hashes'].items():need(sha(p/n)==h,'Preserved case hash')
    source=load(p/'snapshot_manifest.json')['source_hashes'];tb=contents(p/'frozen199-sources.tar.gz');need(len(source)==199 and {n:hashlib.sha256(b).hexdigest()for n,b in tb.items()}==source,'Exact199 source archive')
    need(sha(p/'frozen199-sources.tar.gz')==saved['source_archive_sha256']=='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b','Original tar pin')
    need([e['boundary']for e in device['events']]==['input','output'],'Both recorded device boundaries')
    out=device['events'][-1];ct=out['ciphertexts'][0];need(ct['shape']==[1,2,256,512,40]and ct['nsd']==1,'Actual output metadata')
    need(ct['scale'].hex()=='0x1.ea3b603ea0001p+55','Actual output scale')
    for e in device['events']:
        for r in e['actual_rings']:
            bits=sum(math.log2(q)for q in r['q_towers']+r['p_towers']);need(r['degree']==131072 and len(r['q_towers'])==100 and len(r['p_towers'])==24 and bits<=3523 and abs(bits-r['log2_QP'])<1e-8,'Recorded public ring/security geometry')
        for c in e['ciphertexts']:need(c['devices']and all(d['platform']=='tpu'for d in c['devices']),'TPU boundary payload metadata')
    q=math.prod(out['actual_rings'][0]['q_towers'][:40]);need(q.bit_length()==1096 and 2**1024<q//2,'Public CRT range')
    scale_num,scale_den=ct['scale'].as_integer_ratio();need(math.isfinite((2**1024*scale_den)/scale_num),'Finite ratio despite intfloat overflow')
    try:(q//2*scale_den)/scale_num
    except OverflowError:pass
    else:raise ValueError('Expected nonrepresentable upper public range')
    resources=load(p/'resources.json');need(resources==proc['resources'],'Resource summary link');samples=[json.loads(line)for line in(p/'resource_samples.jsonl').read_text().splitlines()];need(len(samples)==resources['samples']and max(x['aggregate_rss_bytes']for x in samples)==resources['sampled_peak_process_tree_rss_bytes'],'Sampled RSS accounting')
    log=(p/'stderr.log').read_text();need(log.count('bounded homomorphic refresh complete')==8,'Eight recorded refresh completions')
    maps=[]
    for name,nt,nf,lab in [('preserved-synthetic18',18,1,'/tmp/bert-toy5-overflow-diagnosis-20261104'),('synthetic19',19,0,'/tmp/bert-toy5-overflow-diagnosis-v2-20261104')]:
        d=ROOT/name;m=load(d/'proof-sources-before.json');need(m==load(d/'proof-sources-after.json')and len(m)==9,'Nine-source before/after')
        b=contents(d/'proof-sources.tar.gz');need({n:hashlib.sha256(v).hexdigest()for n,v in b.items()}==m,'Exact synthetic source archive')
        x=ET.parse(d/'tests.xml').getroot();need(len(x.findall('.//testcase'))==nt and len(x.findall('.//failure'))==nf and not x.findall('.//error')and not x.findall('.//skipped'),'Synthetic test scope')
        origin=json.loads(b['sources/source_origin.json'])
        for n,entry in origin.items():need(hashlib.sha256(b['sources/'+n]).hexdigest()==source[entry['frozen_path']]==entry['sha256'],'Frozen source copy')
        if a.current:need({n:sha(Path(lab)/n)for n in m}==m,'Current synthetic source readback')
        maps.append(m)
    need([n for n in maps[0]if maps[0][n]!=maps[1][n]]==['sources/test_conversion.py'],'Only test correction between synthetic revisions')
    report=load(ROOT/'synthetic19/report.json');need(report['actual_throw_site_proven']is False and report['isolated_candidate_installed']is False and report['actual_reexecution']is False,'Unproved actual cause/no installed repair')
    owner=load(ROOT/'owner-test-observation.json');need(owner['v2']['direct_tool_chunk']=='e5d48b'and owner['v2']['exit_code']==0 and owner['v2']['pid_or_pgid_captured']is False,'Honest direct tool scope')
    if a.current:need({n:sha(Path('/tmp/cross-optimized-twelve-20261103/checkout')/n)for n in source}==source,'Current199 frozen sources')
    print(json.dumps(dict(passed=True,original_case_passed=False,frozen_sources=199,diagnostic_sources=9,synthetic_tests=19,preserved_predecessor_tests=18,preserved_predecessor_failures=1,current_readback=a.current,actual_throw_site_proven=False,actual_reexecution=False,public_artifacts=len(idx)),sort_keys=True))
if __name__=='__main__':main()
