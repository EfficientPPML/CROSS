"""Whitelist terminal aggregate/source reference-production evidence only."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,math,re,stat,sys,tarfile
LAB=Path('/tmp/bert-ranked-client-reference-20261103');RUN=LAB/'actual-v1'
def require(value,message):
    if not value:raise ValueError(message)
def read(path,limit=32<<20):
    info=path.lstat();require(stat.S_ISREG(info.st_mode) and info.st_size<=limit,'bounded regular public artifact required')
    raw=path.read_bytes();require(len(raw)==info.st_size,'public artifact changed');return raw
def digest(path):return hashlib.sha256(read(path)).hexdigest()
def aggregate(value):
    prohibited={'target_idx','target_ids','labels','input_ids','token_type_ids','logits','native_fp32_logits','native_fp64_logits','approximate_fp64_logits','domain_observations','embeddings'}
    if isinstance(value,dict):
        require(not set(value)&prohibited,'private-valued report field forbidden')
        for v in value.values():aggregate(v)
    elif isinstance(value,list):
        for v in value:aggregate(v)
    elif isinstance(value,float):require(math.isfinite(value),'nonfinite aggregate field')
def main(destination,observed_exit_code):
    r=json.loads(read(RUN/'report.json'));aggregate(r)
    require(all(not Path('/proc',str(pid)).exists() for pid in (1879300,1879301)),'actual process still exists')
    require((observed_exit_code==0)==(r.get('passed')is True),'numeric exit and report disagree')
    require(r.get('execute')is True and r.get('actual_native_started')is True,'not actual native preparation')
    require(r['cpu_affinity']==[124,125,126,127],'actual CPU policy differs')
    expected=json.loads(read(LAB/'preflight-v1/report.json'))['candidate_source_hashes']
    require(len(expected)==41 and r['source_hashes_before']==expected,'source identity differs')
    require(digest(RUN/'sources.tar.gz')==r['source_archive_sha256'],'archive digest differs')
    with tarfile.open(RUN/'sources.tar.gz') as t:
        members=t.getmembers();require(len(members)==len(expected)==len({m.name for m in members}),'archive inventory differs')
        require(all(m.isfile() and not PurePosixPath(m.name).is_absolute() and '..' not in PurePosixPath(m.name).parts for m in members),'unsafe source archive')
        require({m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}==expected,'exact archived bytes differ')
    sys.path.insert(0,str(LAB));import ranked_collect as c
    current=c.hashes(c.source_paths());require(current==expected==r['source_hashes_after'],'current/before/after mismatch')
    if r.get('official_reference_complete')is True:
        require(all(r.get(n)is True for n in r['reference_acceptance_requires']),'reference-only acceptance incomplete')
        require(r['official_evaluation_complete'] and r['progress']['completed_unique']==408 and len(r['cases'])==9,'complete unique/case accounting differs')
        require([(x['size'],x['seed'],x['count']) for x in r['cases']]==[(s,seed,count) for s,count in ((1,100),(2,200),(3,408)) for seed in (42,43,44)],'fixed ordered schedule differs')
        require(sum(x['count'] for x in r['cases'])==2124,'case reuse total differs')
        names={'master_references.jsonl','private_manifest.json'}|{f'size{s}-seed{seed}-references.jsonl' for s in (1,2,3) for seed in (42,43,44)}
        require(set(r['private_reference_files'])==names,'private file descriptor inventory differs')
        require(r['native_policy']['official_attention']=='sdpa' and r['native_policy']['diagnostic_attention']=='eager' and r['native_policy']['torch_threads']==4,'actual native policy differs')
    if r.get('passed')is True:
        require(r['exact_reference_parity_passed'] and r['approximation_validation_passed'],'outer result hides a diagnostic failure')
        require(all(all(case[n]is True for n in ('official_reference_complete','approximation_complete','approximate_numeric_gate_passed','prediction_agreement','ranked_accuracy_floor_passed','ranked_accuracy_drop_passed','exact_reference_parity_passed')) for case in r['cases']),'a ranked case failed')
    timing=read(LAB/'actual-v1.time').decode();match=re.search(r'Exit status: (\d+)\s*$',timing)
    require(match is not None and int(match[1])==observed_exit_code,'external exit differs')
    destination.mkdir(exist_ok=False)
    for name in ('report.json','sources.tar.gz'):(destination/name).write_bytes(read(RUN/name))
    for name in ('actual-v1.log','actual-v1.time'):(destination/name).write_bytes(read(LAB/name))
    (destination/'launch.json').write_bytes(read(Path('/tmp/ranked-reference-launch-20261103.json')))
    result=dict(passed=r['passed'],official_reference_complete=r['official_reference_complete'],exact_reference_parity_passed=r['exact_reference_parity_passed'],approximation_validation_passed=r['approximation_validation_passed'],
        observed_exit_code=observed_exit_code,all_owned_pids_absent=True,pids=[1879300,1879301],source_count=41,current_before_after_archive_equal=True,
        source_archive_sha256=r['source_archive_sha256'],report_sha256=digest(RUN/'report.json'),
        scope='Aggregate/source archiving only; no original or produced private reference bytes read or copied. Not HE or benchmark timing evidence.')
    (destination/'terminal_readback.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (destination/'archive_driver.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);p.add_argument('--observed-exit-code',type=int,required=True);main(**vars(p.parse_args()))
