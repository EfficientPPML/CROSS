from pathlib import Path
import hashlib,json
ROOT=Path('/home/jianming_gatech/CROSS');BASE=ROOT/'cadathlon/validation';LAB=Path(__file__).resolve().parent
changed={'cadathlon/common/encrypted_streaming_linear.py','cadathlon/common/package.py','cadathlon/tests/test_streaming_linear.py','cadathlon/tests/test_streaming_linear_contracts.py','cadathlon/tests/test_package.py','cadathlon/tests/test_bert_staging.py'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def resolve_p1(name):
 if name.startswith('CROSS/'):return ROOT/name.removeprefix('CROSS/')
 if name.startswith('common/'):return ROOT/'cadathlon'/name
 if name.startswith('problem/'):return ROOT/'cadathlon/non_agent_pset/p1_benchmark/src'/name.removeprefix('problem/')
 raise ValueError(name)
records={}
for label,path in [('p1',BASE/'p1_student_post_promotion_20261103/source_inventory.json'),('p3',BASE/'p3_compact_post_promotion_20261103/source-start.json')]:
 manifest=json.loads(path.read_text());rows={}
 for name,entry in manifest.items():
  p=resolve_p1(name) if label=='p1' else ROOT/name
  expected=entry['sha256'] if isinstance(entry,dict) else entry;actual=sha(p)
  rows[str(p.relative_to(ROOT))]={'expected':expected,'actual':actual,'matches':actual==expected}
 overlap=sorted(changed.intersection(rows));records[label]={'inventory_path':str(path),'inventory_sha256':sha(path),'count':len(rows),'all_live_sources_match':all(x['matches'] for x in rows.values()),'changed_path_overlap':overlap,'sources':rows}
r={'scope':'Read-only source identity/disjointness check for proposed six-file streaming helper and packaging-test changes. No fresh timing or calibration claim.','passed':all(x['all_live_sources_match'] and not x['changed_path_overlap'] for x in records.values()),'proposed_changed_paths':sorted(changed),'calibrations':records}
(LAB/'calibration_source_check.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:{a:b for a,b in v.items() if a!='sources'} for k,v in records.items()},indent=2));raise SystemExit(0 if r['passed'] else 1)
