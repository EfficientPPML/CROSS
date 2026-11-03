from pathlib import Path
from fractions import Fraction as F
import hashlib,json,tarfile
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();canonical=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
v=json.loads((P/'verification.json').read_text());protocol=json.loads((P/'protocol.json').read_text());m=json.loads((P/'candidate-v2/manifest.json').read_text());base=json.loads((P/'inputs/base_manifest.json').read_text());bounds=json.loads((P/'inputs/public_gate_bounds.json').read_text());cert=json.loads((P/'inputs/uniform_polynomial_certificates.json').read_text());multipliers=json.loads((P/'norm_multipliers.json').read_text())
assert all(sha(P/n)==h for n,h in v['artifact_sha256'].items())
for name,count in [('candidate-v2',20),('selected-preflight-v1',24),('selection-preflight-v2',20)]:
 r=json.loads((P/name/'report.json').read_text());assert r['passed'] and r['source_files_unchanged']
 with tarfile.open(P/name/'sources.tar.gz') as t:
  members=t.getmembers();assert len(members)==len({x.name for x in members})==count and all(x.isfile() for x in members)
  assert {x.name:hashlib.sha256(t.extractfile(x).read()).hexdigest() for x in members}==r['source_hashes_before']==r['source_hashes_after']
 assert sha(P/name/'sources.tar.gz')==r['source_archive_sha256']
 assert all(sha(P/n)==h for n,h in r['source_hashes_before'].items())
assert sha(P/'protocol.json')==v['protocol_sha256'] and sha(P/'candidate-v2/manifest.json')==v['manifest_sha256']
assert protocol['all28_actual_uniform_certificates'] and not protocol['model_execution_authorized']
old_norms={k:r for k,r in base['nonlinear'].items() if not k.endswith('.silu')};new_norms={k:r for k,r in m['nonlinear'].items() if not k.endswith('.silu')}
assert len(new_norms)==113 and old_norms==new_norms and canonical(new_norms)==protocol['unchanged_norms_sha256']
fields=('softmax','exp','initial_reciprocal','reciprocal_by_keys');assert {k:m[k] for k in fields}=={k:base[k] for k in fields}
assert canonical({k:m[k] for k in fields})==protocol['unchanged_softmax_sha256']
assert m['all_silu_sites_supported'] and set(m['nonlinear'])==set(base['nonlinear'])
for layer in range(28):
 row=multipliers[str(layer)];rule=base['nonlinear'][f'{layer}.post.inverse_sqrt'];saved=row['delta_fraction'];delta=F(int(saved['numerator_hex'],16),int(saved['denominator_hex'],16))
 if rule['method']=='newton':assert delta==F(rule['final_relative_error_bound'])
 else:
  c=cert['results'][f'{layer}.post.inverse_sqrt'];assert c['certified'];d=c['exact_relative_bound'];assert delta==F(int(d['numerator_hex'],16),int(d['denominator_hex'],16))
 assert 0<=delta<F(1,10**6) and F(row['relative_upper_binary64'])>=delta
 B=bounds['layers'][str(layer)]['maximum_row_upper'];exact=F(B)*(1+delta)*F(9,8);D=(exact.numerator+exact.denominator-1)//exact.denominator
 assert row['public_silu_domain']==[-D,D]==m['nonlinear'][f'{layer}.silu']['domain']==protocol['final_public_silu_domains'][str(layer)]
 assert canonical(rule)==row['unchanged_norm_rule_sha256']
assessments=json.loads((P/'candidate-v2/assessments.json').read_text());assert len(assessments)==28
for label,a in assessments.items():
 assert a['supported'] and a['attempts'][-1]['passed'] and all(not x['passed'] for x in a['attempts'][:-1])
 for grid in a['attempts'][-1]['grids'].values():
  assert grid['finite'] and grid['passed'] and grid['max_abs_error']<=1e-4
  assert all(x['finite'] and x['inside_domain'] and 1<=x['minimum']<=x['maximum']<=2.1 for x in grid['denominator_checks'])
difference=sum(m['nonlinear'][f'{i}.silu']['depth']-base['nonlinear'][f'{i}.silu']['depth'] for i in range(28))
for S,r in m['depth_by_sequence'].items():assert r['final_depth']==base['depth_by_sequence'][S]['final_depth']+difference
assert protocol['fresh_validation']['sha256']=='af93ae2a580bc4ff51d1463e8ab50ab054e295d0e4932340ae81d18185696988'
assert protocol['original_published_regression']['required_after_fresh_validation']
print(json.dumps({'passed':True,'all28_supported':True,'unchanged_norm_rules':113,'softmax_unchanged':True,'model_evaluated':False,'manifest_sha256':v['manifest_sha256']}))
