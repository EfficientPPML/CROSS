"""Offline source/timing metadata verifier; no benchmark execution."""
import argparse,hashlib,json,statistics,tarfile
from pathlib import Path
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda n:json.loads((P/n).read_bytes())
def main(current):
    index=load('files.sha256.json')
    assert {str(f.relative_to(P))for f in P.rglob('*')if f.is_file()and f.name!='files.sha256.json'}==set(index)
    assert all(sha((P/n).read_bytes())==h for n,h in index.items())
    assert not any(n.endswith('.npy')for n in index)
    m=load('sources-before.json');assert len(m)==210 and m==load('sources-after.json')
    with tarfile.open(P/'sources.tar.gz','r:gz')as t:
        members=t.getmembers();assert len(members)==210 and all(x.isfile()for x in members)
        bodies={x.name:t.extractfile(x).read()for x in members}
        assert {n:sha(b)for n,b in bodies.items()}==m
    plan=load('plan.json');r=load('run-v1/report.json');review=load('owner-readback/review.json')
    assert r['passed']is True and r['complete']is True and r['source_unchanged']is True
    assert r['source_hashes_start']==m==r['source_hashes_end']
    assert r['source_manifest_sha256']==sha((P/'sources-before.json').read_bytes())
    assert review['benchmark_report_sha256']==sha((P/'run-v1/report.json').read_bytes())
    assert review['source_archive_sha256']==sha((P/'sources.tar.gz').read_bytes())
    assert r['jax_backend']=='cpu'and r['device_platforms']==['cpu']and r['tpu_plugin_loaded_at_end']is False
    assert r['device_access_denials']==[]and r['import_guard_denials']==['libtpu']
    assert r['ciphertext_or_key_or_model_operation']is False and r['actual_toy5_recovery']is False
    assert all(m[x['path']]==x['sha256']for x in r['loaded_local_modules'].values())
    assert len(plan['original_runtime207_sources'])==207
    assert all(m['runtime207/'+n]==h for n,h in plan['original_runtime207_sources'].items())
    assert len(r['cases'])==2
    for case,s in zip(r['cases'],review['summary'],strict=True):
        assert case['name']==s['name']and len(case['calls'])==10
        wanted=[('warmup',None,name)for name in plan['warmup_order']]+[('measured',i,name)for i,order in enumerate(plan['measured_pair_orders'])for name in order]
        assert [(x['phase'],x['pair'],x['decoder'])for x in case['calls']]==wanted
        assert all(x['exact_binary64_agreement']is True and x['elapsed_ns']>0 and x['elapsed_seconds']==x['elapsed_ns']/1e9 for x in case['calls'])
        assert len({x['output_sha256']for x in case['calls']})==1
        times={name:[x['elapsed_seconds']for x in case['calls']if x['phase']=='measured'and x['decoder']==name]for name in ['old','candidate']}
        assert times['old']==s['old_seconds']and times['candidate']==s['candidate_seconds']
        ratios=[c/o for c,o in zip(times['candidate'],times['old'],strict=True)]
        assert ratios==s['paired_candidate_over_old_ratios']and statistics.median(ratios)==s['median_observed_paired_ratio']
    datasets=load('dataset_manifest.json');assert datasets['files']==r['dataset_files']
    assert len(datasets['files'])==6 and sum(x['bytes']for x in datasets['files'].values())==51905280
    owner=load('owner.json');assert owner['session']==28130 and owner['exit_code']==0 and owner['reaped']is True
    assert owner['pid']==r['pid']==3011310 and owner['pgid']==r['pgid']==3011309
    if current:
        root=Path('/tmp/ckks-decode-clear-cpu-microbenchmark-20261104')
        assert all(sha((root/n).read_bytes())==h for n,h in m.items())
        old=Path('/tmp/ckks-scaled-crt-decoder-candidate-v2-20261104/source-bound-v1/execution')
        assert all(sha((old/n).read_bytes())==h for n,h in plan['original_runtime207_sources'].items())
        for n,d in datasets['files'].items():
            b=(Path(datasets['retained_directory'])/n).read_bytes();assert len(b)==d['bytes']and sha(b)==d['sha256']
    print(json.dumps(dict(passed=True,sources=210,unchanged_predecessor_sources=207,decode_calls=20,bit_agreement=True,synthetic_dataset_bodies_archived=False,current_checked=current,speed_claim=False),sort_keys=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--current',action='store_true');main(p.parse_args().current)
