"""Source/dataset/timing readback only; no backend import or benchmark repeat."""
from pathlib import Path
import hashlib,json,os,statistics,tarfile
P=Path('/tmp/ckks-decode-clear-cpu-microbenchmark-20261104')
OUT=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
def absent(pid,group=False):
    try:(os.killpg if group else os.kill)(pid,0)
    except ProcessLookupError:return True
    return False
m=json.loads((P/'sources-before.json').read_bytes())
assert len(m)==210
assert all(sha((P/n).read_bytes())==h for n,h in m.items())
with tarfile.open(P/'sources.tar.gz','r:gz')as t:
    members=t.getmembers();assert len(members)==210 and all(x.isfile()for x in members)
    assert {x.name:sha(t.extractfile(x).read())for x in members}==m
plan=json.loads((P/'plan.json').read_bytes())
assert len(plan['original_runtime207_sources'])==207
F=Path('/tmp/ckks-scaled-crt-decoder-candidate-v2-20261104/source-bound-v1/execution')
assert all(sha((F/n).read_bytes())==h==m['runtime207/'+n]for n,h in plan['original_runtime207_sources'].items())
r=json.loads((P/'run-v1/report.json').read_bytes())
assert r['passed']is True and r['complete']is True and r['source_unchanged']is True
assert r['source_hashes_start']==m==r['source_hashes_end']
assert r['source_manifest_sha256']==sha((P/'sources-before.json').read_bytes())
assert r['jax_backend']=='cpu'and r['device_platforms']==['cpu']and r['tpu_plugin_loaded_at_end']is False
assert r['device_access_denials']==[]and r['import_guard_denials']==['libtpu']
assert all(x['sha256']==m[x['path']]for x in r['loaded_local_modules'].values())
assert set(r['dataset_files'])=={p.name for p in(P/'run-v1/synthetic_datasets').iterdir()}
for n,row in r['dataset_files'].items():
    raw=(P/'run-v1/synthetic_datasets'/n).read_bytes();assert len(raw)==row['bytes']and sha(raw)==row['sha256']
assert len(r['cases'])==2
summary=[]
for case,expected in zip(r['cases'],plan['cases'],strict=True):
    assert case['name']==expected['name']and case['q']==expected['q']and case['scale']==expected['scale']
    assert case['residues_unchanged']is True and len(case['calls'])==10
    schedule=[('warmup',None,name)for name in plan['warmup_order']]+[('measured',i,name)for i,order in enumerate(plan['measured_pair_orders'])for name in order]
    assert [(x['phase'],x['pair'],x['decoder'])for x in case['calls']]==schedule
    assert all(x['exact_binary64_agreement']is True and type(x['elapsed_ns'])is int and x['elapsed_ns']>0 and x['elapsed_seconds']==x['elapsed_ns']/1e9 for x in case['calls'])
    assert len({x['output_sha256']for x in case['calls']})==1
    times={name:[x['elapsed_seconds']for x in case['calls']if x['phase']=='measured'and x['decoder']==name]for name in ['old','candidate']}
    ratios=[times['candidate'][i]/times['old'][i]for i in range(4)]
    summary.append(dict(name=case['name'],old_seconds=times['old'],candidate_seconds=times['candidate'],old_median_seconds=statistics.median(times['old']),candidate_median_seconds=statistics.median(times['candidate']),paired_candidate_over_old_ratios=ratios,median_observed_paired_ratio=statistics.median(ratios),warmups=[x for x in case['calls']if x['phase']=='warmup'],exact_outputs_equal=True))
assert absent(r['pid'])and absent(r['pgid'],group=True)
(P/'sources-after.json').write_text(json.dumps(m,sort_keys=True,indent=2)+'\n')
result=dict(passed=True,source_count=210,unchanged_runtime207=True,source_current_archive_start_end_equal=True,source_manifest_sha256=sha((P/'sources-before.json').read_bytes()),source_archive_sha256=sha((P/'sources.tar.gz').read_bytes()),benchmark_report_sha256=sha((P/'run-v1/report.json').read_bytes()),dataset_readback=True,dataset_count=len(r['dataset_files']),dataset_bytes=sum(x['bytes']for x in r['dataset_files'].values()),all20_calls_exact_output_agreement=True,summary=summary,owner_session=28130,owner_reported_exit_code=0,owner_reported_reaped=True,owner_reap_tool='41e03a',pid=r['pid'],pgid=r['pgid'],live_external_row=(P/'run-v1/live-identity.txt').read_text(),pid_group_absent=True,scope='clear CPU decoder microbenchmark only',performance_gate=None,statistical_speed_claim=False,new_optimization=False)
(OUT/'review.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:result[k]for k in ['passed','source_count','dataset_count','dataset_bytes','pid','pgid','summary']},sort_keys=True))
