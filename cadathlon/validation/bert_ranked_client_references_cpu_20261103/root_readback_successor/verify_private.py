from pathlib import Path
import hashlib,json,math,os,stat,sys,tarfile,datetime
LAB=Path('/tmp/bert-ranked-client-reference-20261103');sys.path.insert(0,str(LAB))
import ranked_collect
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report_path=LAB/'actual-v1/report.json';r=json.loads(report_path.read_bytes())
assert sha(report_path)=='77f93c90e03609bdaf12441cc8a8d07e08b5e1a475c1480f283b4f592668ae74'
assert r['passed'] and r['exact_reference_parity_passed'] and r['approximation_validation_passed']
assert all(r[n] is True for n in r['reference_acceptance_requires'])
assert r['source_hashes_before']==r['source_hashes_after']==ranked_collect.hashes(ranked_collect.source_paths())
assert len(r['source_hashes_before'])==41 and sha(LAB/'actual-v1/sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(LAB/'actual-v1/sources.tar.gz') as t:
 m=t.getmembers();assert len(m)==len({x.name for x in m})==41 and all(x.isfile() for x in m)
 assert {x.name:hashlib.sha256(t.extractfile(x).read()).hexdigest() for x in m}==r['source_hashes_before']
assert r['asset_hashes_before']==r['asset_hashes_after']=={n:sha(n) for n in r['asset_hashes_after']}
assert not Path('/proc/'+str(r['pid'])).exists() and not Path('/proc/1879300').exists()
def private(path,record):
 s=path.lstat();assert stat.S_ISREG(s.st_mode) and stat.S_IMODE(s.st_mode)==0o600 and s.st_size==record['bytes']
 b=path.read_bytes();assert hashlib.sha256(b).hexdigest()==record['sha256']
 end=path.lstat();assert (s.st_dev,s.st_ino,s.st_mode,s.st_size)==(end.st_dev,end.st_ino,end.st_mode,end.st_size)
 return b
p=Path(r['private_workspace']);s=p.lstat();assert stat.S_ISDIR(s.st_mode) and stat.S_IMODE(s.st_mode)==0o700
assert dict(device=s.st_dev,inode=s.st_ino)==r['private_workspace_identity']
assert set(x.name for x in p.iterdir())==set(r['private_reference_files']) and len(r['private_reference_files'])==11
raw={n:private(p/n,v) for n,v in r['private_reference_files'].items()}
manifest=json.loads(raw['private_manifest.json']);assert manifest['bindings']['source_archive_sha256']==r['source_archive_sha256']
assert manifest['bindings']['final_acceptance_requires_public_report'] is True
assert manifest['files']=={n:v for n,v in r['private_reference_files'].items() if n!='private_manifest.json'}
master=[json.loads(x) for x in raw['master_references.jsonl'].splitlines()];assert len(master)==408
index={row['target_idx']:row for row in master};assert len(index)==408
max_exact=max_approx=0.;domain_count=0
for row in master:
 assert row['approximation_complete'] and row['exact_reference_parity_passed']
 vals=[row[n] for n in ['native_fp32_logits','native_fp64_logits','independent_exact_fp64_logits','approximate_fp64_logits']]
 assert all(len(v)==2 and all(type(x)is float and math.isfinite(x) for x in v) for v in vals)
 official,native,exact,approx=vals
 assert max(abs(a-b) for a,b in zip(native,exact))<=1e-12
 assert all(abs(a-b)<=.02+.01*abs(b) for a,b in zip(approx,official))
 predict=lambda xs:int(xs[1]>=xs[0])
 assert predict(approx)==predict(official)==predict(native)
 assert row['label'] in (0,1)
 parity=row['exact_parity'];assert parity['embedding']['passed'] and parity['embedding']['max_abs']==0.
 checks=[parity['pooler'],parity['logits']]+[layer[k] for layer in parity['layers'] for k in ['attention','block']]
 assert len(parity['layers'])==12 and all(x['passed'] and x['max_abs']<=1e-12 for x in checks)
 max_exact=max(max_exact,*(x['max_abs'] for x in checks));max_approx=max(max_approx,max(abs(a-b) for a,b in zip(approx,official)))
 domains=row['domain_observations'];assert len(domains)==156
 assert len({x['label'] for x in domains})==156
 assert all(x['below']==x['above']==x['nonfinite']==0 and x['count']>0 and x['domain'][0]<=x['minimum']<=x['maximum']<=x['domain'][1] for x in domains)
 domain_count+=len(domains)
fixture=json.loads(Path('/tmp/bert-ranked-input-fixtures-20261103/actual-v1/report.json').read_bytes());input_root=Path(fixture['private_workspace'])
verified_inputs=0;cases=[]
for public,actual in zip(fixture['cases'],r['cases']):
 size,seed=actual['size'],actual['seed'];case_root=input_root/f'size{size}-seed{seed}'
 for n,h in public['input_file_sha256'].items():
  assert sha(case_root/n)==h;assert stat.S_IMODE((case_root/n).stat().st_mode)==0o600;verified_inputs+=1
 selection=json.loads((case_root/'private_selection.json').read_bytes())
 rows=[json.loads(x) for x in raw[f'size{size}-seed{seed}-references.jsonl'].splitlines()]
 assert rows==[index[i] for i in selection['target_ids']]
 assert [x['label'] for x in rows]==selection['labels']
 name={1:'small',2:'medium',3:'large'}[size]
 files=[n for n in public['input_file_sha256'] if n.endswith('client_preprocessed_input')];assert len(files)==1
 inputs=[json.loads(x) for x in (case_root/files[0]).read_bytes().splitlines()]
 assert len(inputs)==len(rows)==actual['count']
 for row,original in zip(rows,inputs):
  assert row['original_positions']==[i for i,x in enumerate(original['attention_mask']) if x]
  assert row['target_idx']==original['target_idx']
 official=sum(int(x['native_fp32_logits'][1]>=x['native_fp32_logits'][0])==x['label'] for x in rows)
 approximate=sum(int(x['approximate_fp64_logits'][1]>=x['approximate_fp64_logits'][0])==x['label'] for x in rows)
 assert official==actual['official_correct'] and approximate==actual['approximate_correct']
 assert 4*approximate>=3*len(rows) and 100*(official-approximate)<=2*len(rows)
 cases.append(dict(count=len(rows),seed=seed,official_correct=official,approximate_correct=approximate))
assert verified_inputs==54 and sum(x['count'] for x in cases)==2124
for n,v in r['private_reference_files'].items():assert private(p/n,v)==raw[n]
assert sha(report_path)=='77f93c90e03609bdaf12441cc8a8d07e08b5e1a475c1480f283b4f592668ae74'
result=dict(passed=True,scope='Independent private readback and aggregate recomputation; no repeated model inference or HE.',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_sha256=sha(report_path),source_count=41,private_reference_files=11,original_fixture_files=54,unique_rows=408,selected_rows=2124,domain_observations=domain_count,max_exact_native_diagnostic_error=max_exact,max_approximate_official_logit_error=max_approx,cases=cases,private_payloads_archived=False,pids_absent=True)
(LAB/'root-readback').mkdir(exist_ok=True)
(LAB/'root-readback/root_private_readback.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
