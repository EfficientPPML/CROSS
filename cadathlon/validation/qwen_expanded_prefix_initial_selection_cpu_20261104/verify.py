"""Public integrity/count verification, not a certificate rerun or model evaluation."""
from pathlib import Path, PurePosixPath
import argparse, collections, hashlib, io, json, os, tarfile
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
SHA=lambda raw:hashlib.sha256(raw).hexdigest()
J=lambda path:json.loads(path.read_bytes())
REPORT='7ba0ffae120c7b68b51d2ec1fc040e51e3fd12d989f1ab73316f231091917ad9'
CANDIDATE='344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
TAR='821ba2035e4c0a1efbcdc9f3ee718c6ecadb2f013cd63ddaec3911d632494b6c'
MANIFEST='f3d0bf68c503c2baba2cc34b7806c152efae153bbc180a4ebb85aaeae2d833f0'
GRANT='99d3cca99b00231110b77ead984a47624296a05e15b875cc03c99e93a76bc4b9'
REVIEW='6c98b64cd506cddeb5f4bd64767e5b7e1483ba5d28ca04a442cd3d7b4fa960de'
OUTER='cbff16bf203e311b503fb3c2f3a37af97f9d4d9b83dc6e4db5ae9153ca3ad352'
BASE='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
NORMS=tuple(f'{i}.{s}.inverse_sqrt'for i in range(28)for s in('input','post','q','k'))+('final.inverse_sqrt',)

def integer(value,expected):assert type(value)is int and value==expected

def tar_contents(path):
 result={};total=0
 with tarfile.open(fileobj=io.BytesIO(path.read_bytes()),mode='r:gz')as handle:
  for m in handle:
   name=PurePosixPath(m.name);total+=m.size
   assert m.isfile()and not name.is_absolute()and '..'not in name.parts and str(name)==m.name and m.name not in result
   assert 0<=m.size<=16<<20 and total<=32<<20
   result[m.name]=handle.extractfile(m).read()
 return result

def absent(pid,group=False):
 try:(os.killpg if group else os.kill)(pid,0)
 except ProcessLookupError:return True
 return False

def protected(m):
 return dict(nonlinear={n:r for n,r in m['nonlinear'].items()if n not in NORMS},**{n:m[n]for n in('softmax','exp','initial_reciprocal','reciprocal_by_keys')})

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--current',action='store_true');args=parser.parse_args()
 assert __debug__,'Run without -O.'
 files={}
 for p in HERE.rglob('*'):
  assert not p.is_symlink()
  if p.is_file()and p.name!='files.sha256.json':files[p.relative_to(HERE).as_posix()]=SHA(p.read_bytes())
 assert files==J(HERE/'files.sha256.json')
 whitelist=J(HERE/'public_whitelist.json');assert len(whitelist['files'])==28
 assert whitelist['private_snapshot_or_journals_read_or_copied']is False and whitelist['new_selection_or_model_execution_by_archiver']is False
 for n,d in whitelist['files'].items():
  raw=(HERE/n).read_bytes();assert d['public_only']is True and len(raw)==d['bytes']and SHA(raw)==d['sha256']
 p=HERE/'actual';r=J(p/'report.json');m=J(p/'source_manifest.json')
 assert SHA((p/'report.json').read_bytes())==REPORT and SHA((p/'source_manifest.json').read_bytes())==MANIFEST
 assert SHA((p/'sources.tar.gz').read_bytes())==TAR==r['source_archive']['sha256']
 source=tar_contents(p/'sources.tar.gz');assert len(source)==36 and {n:SHA(raw)for n,raw in source.items()}==m
 assert r['source_hashes_start']==r['source_hashes_end']==m and r['source_manifest_sha256']==MANIFEST
 for k in('passed','complete','publication_complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','actual_coefficient_selection','candidate_supported','all_norm_sites_supported','protected_rules_supported','private_inputs_never_removed','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'):
  assert r[k]is True,k
 for k in('model_execution','token_or_calibration_body_reads','validation_body_reads','encrypted_execution','synthetic_only'):assert r[k]is False,k
 assert r['audit_denials']==[] and 'error_type'not in r
 integer(r['candidate_ordinal'],1)
 assert len(r['loaded_local_origins'])==16 and all(m[d['path']]==d['sha256']for d in r['loaded_local_origins'].values())
 assert set(r['outputs'])=={'candidate_manifest.json','assessments.json','protected_checks.json'}
 for n,d in r['outputs'].items():
  raw=(p/n).read_bytes();assert SHA(raw)==d['sha256']and len(raw)==d['bytes']
 assert SHA((p/'candidate_manifest.json').read_bytes())==CANDIDATE
 integer(r['outputs']['assessments.json']['bytes'],37283359);integer(r['outputs']['candidate_manifest.json']['bytes'],1110816)
 c=J(p/'candidate_manifest.json');a=J(p/'assessments.json');checks=J(p/'protected_checks.json')
 assert c['schema']=='qwen-expanded-initial-selection-v1'and c['synthetic_only']is False and c['actual_coefficient_selection']is True
 integer(c['candidate_index'],1);integer(c['candidate_ordinal'],1)
 assert c['selection_grant_sha256']==GRANT==r['grant_sha256']and c['selector_source_manifest_sha256']==MANIFEST
 assert set(a)==set(NORMS)and len(a)==113
 assert c['selection_policy']=={'degree_ladder':[16,32,64,128,256,512],'epsilon':1e-6,'grid_geometric':2049,'grid_linear':2049,'max_newton_iterations':64,'uniform_relative_target':1e-6}
 assert SHA(source['inputs/candidate3_manifest.json'])==BASE==c['parent_manifest_sha256']
 assert protected(c)==protected(json.loads(source['inputs/candidate3_manifest.json']))
 methods=collections.Counter();attempts=0
 for name in NORMS:
  row=a[name];assert row['rule']==c['nonlinear'][name]
  method=row['rule']['method'];methods[method]+=1;history=row['attempts'];attempts+=len(history)
  assert [h['degree']for h in history]==[16,32,64,128,256,512][:len(history)]and 1<=len(history)<=6
  for index,item in enumerate(history):
   selected=item['certificate']['certified']is True and item['binary64_diagnostic']['passed']is True
   assert item['selected']is selected
   if index<len(history)-1:assert selected is False
  assert row['binary64_diagnostic']['passed']is True
  if method=='polynomial':
   assert history[-1]['selected']is True and row['certificate']['certified']is True
   for key in('rule','certificate','binary64_diagnostic'):assert row[key]==history[-1][key]
  else:
   assert method=='newton'and len(history)==6 and all(h['selected']is False for h in history)
   assert row['certificate']['verified']is True and type(row['rule']['iterations'])is int and 1<=row['rule']['iterations']<=64
 assert methods=={'polynomial':36,'newton':77}and attempts==648
 assert set(checks)==set(map(str,range(28)))
 assert all(all(row[k]is True for k in('radius_covered','observed_silu_covered','observed_centered_scores_covered'))for row in checks.values())
 assert set(c['depth_by_sequence'])==set(map(str,range(1,33)))
 root=HERE/'terminal-review';review=J(root/'review.json');outer=J(root/'outer_receipt.json');detail=J(root/'detail.json')
 assert SHA((root/'review.json').read_bytes())==REVIEW and SHA((root/'outer_receipt.json').read_bytes())==OUTER
 assert SHA((root/'review.py').read_bytes())==review['review_script_sha256']==detail['script_sha256']=='f1295f2074166508e53d113e9e19d478fd18d8691f578650388baa29ba75f50e'
 assert review['passed']is True and review['candidate_calibration_performed']is False
 for k,pin in(('report_sha256',REPORT),('candidate_manifest_sha256',CANDIDATE),('source_archive_sha256',TAR),('source_manifest_sha256',MANIFEST),('grant_sha256',GRANT)):
  assert review[k]==outer[k]==pin
 assert outer['schema']=='qwen-initial-selection-root-terminal-v1'
 integer(outer['exit_code'],0);assert outer['reaped']is True and outer['process_group_absent']is True
 for k,n in(('actual_session',48044),('actual_exit_code',0),('review_session',96555),('review_exit_code',0)):integer(detail[k],n)
 assert detail['actual_reaped']is True and detail['review_reaped']is True and detail['review_group_absent']is True
 assert detail['review_sha256']==REVIEW and detail['producer_pid']==r['pid']==review['producer_pid_independently_absent']
 assert detail['review_pid']==review['review_pid']==review['review_pgid']
 integer(review['complete_norm_sites'],113);integer(review['polynomial_attempts_replayed'],648)
 assert review['selected_methods']==dict(methods)and review['outputs']==r['outputs']
 for k in('newton_iterations_newly_replayed','uniform_domains_independently_recomputed','protected_rules_exact','all32_depths_recomputed'):assert review[k]is True
 integer(review['protected_checks_replayed'],28)
 assert review['abstract_depths']=={str(s):c['depth_by_sequence'][str(s)]['final_depth']for s in(1,8,32)}=={'1':6992,'8':13522,'32':14398}
 launch=HERE/'launch-review';g=J(launch/'launch.json');lr=J(launch/'review.json')
 assert SHA((launch/'launch.json').read_bytes())==GRANT
 assert SHA((launch/'review.json').read_bytes())=='50601486fdbf17b2a685316b2fb92d8ebccb9d9a3c95c2c5be36ab6e94fb13b1'
 assert SHA((launch/'review.py').read_bytes())==lr['review_script_sha256']
 assert g['ready']is True and g['execute_authorized']is True and lr['passed']is True
 integer(g['candidate_ordinal'],1);assert g['source_manifest_sha256']==MANIFEST
 for label,name in(('parent','parent.json'),('worker','worker.json'),('source_archive','sources.tar.gz'),('root_review','root_review.json'),('reconstruction_script','reconstruction_script.py.txt'),('root_reconstruction_outer_receipt','root_outer.json')):
  raw=(HERE/'initial-proof'/name).read_bytes();d=g['initial_evidence'][label]
  assert SHA(raw)==d['sha256']and len(raw)==d['bytes']
 assert lr['initial_evidence']==g['initial_evidence']
 initial=J(HERE/'initial-proof/worker.json');ir=J(HERE/'initial-proof/root_review.json');io_=J(HERE/'initial-proof/root_outer.json');po=J(HERE/'initial-proof/observer_outer.json')
 assert initial['passed']is True and ir['passed']is True and io_['exit_code']==po['exit_code']==0 and io_['reaped']is po['reaped']is True
 integer(initial['semantic_passes'],3200);integer(initial['native_forward_completed'],3232);integer(ir['private_files'],9666)
 assert r['initial_snapshot_sha256']==ir['snapshot_descriptor']['sha256']==review['initial_snapshot_sha256']
 assert c['initial_snapshot_sha256']==r['initial_snapshot_sha256']
 assert r['initial_parent_report_sha256']==g['initial_evidence']['parent']['sha256']==c['initial_parent_report_sha256']
 assert r['initial_worker_report_sha256']==g['initial_evidence']['worker']['sha256']
 assert r['initial_root_review_sha256']==g['initial_evidence']['root_review']['sha256']
 assert r['root_reconstruction_outer_sha256']==g['initial_evidence']['root_reconstruction_outer_receipt']['sha256']
 # Existing public-output comparison only; no new coefficient selection or private snapshot read.
 union=initial['input_ranges']['native_exact_union']
 for name,row in a.items():
  observed=union[name];domain=[max(1e-6,.8*observed['minimum']),1.2*observed['maximum']]
  assert row['base_domain']==domain and row['domain']==[max(1e-6,domain[0]/4),domain[1]*4]
 prep=HERE/'preparation';pr=J(prep/'report.json');pm=J(prep/'source_manifest.json')
 assert SHA((prep/'report.json').read_bytes())==r['preflight_report_sha256']==g['preflight_report_sha256']=='804fea6b0fbacaaaa226d50962e3e535ce6bdff62e0348c9febc896bfa120c3d'
 assert pr['passed']is True and pm==m and pr['source_hashes_before']==pr['source_hashes_after']==pr['execution_hashes']==m
 pb=tar_contents(prep/'sources.tar.gz');assert len(pb)==37 and {n:v for n,v in pb.items()if n!='external_preflight.py'}==source
 assert SHA((prep/'sources.tar.gz').read_bytes())==pr['source_archive_sha256']=='4e76d85e6a3ce0f8f2570a225f86a63f22a1b101e7039b2fb900988f248e0a13'
 cases=ET.parse(prep/'tests.xml').findall('.//testcase');assert len(cases)==150 and not any(x.find(k)is not None for x in cases for k in('failure','error','skipped'))
 assert J(prep/'audit.json')['denials']==[]
 assert SHA((prep/'root-review/review.json').read_bytes())==r['independent_preflight_review_sha256']=='8b227233d673dfc41d5fbb997125e6a6c564250e8f501da519c805184168740f'
 if args.current:
  for path in(Path('/tmp/qwen-expanded-prefix-selector-admission-v2-20261104'),Path('/tmp/qwen-expanded-prefix-selector-admission-preflight-v2-20261104/execution_source')):
   assert {q.relative_to(path).as_posix():SHA(q.read_bytes())for q in path.rglob('*')if q.is_file()}==m
  actual=Path('/tmp/qwen-expanded-prefix-selector-actual-v1-20261104')
  for n in('report.json','candidate_manifest.json','assessments.json','protected_checks.json','sources.tar.gz','source_manifest.json'):
   assert SHA((actual/n).read_bytes())==SHA((p/n).read_bytes())
  assert absent(r['pid'])and absent(r['pid'],True)and absent(detail['review_pid'])and absent(detail['review_pid'],True)
 print(json.dumps({'passed':True,'indexed_public_files':len(files),'actual_sources':36,'preflight_proof_sources':37,'candidate_ordinal':1,'normalization_sites':113,'methods':dict(methods),'polynomial_attempts':attempts,'current':args.current,'certificate_replay_by_this_verifier':False,'private_body_reads':False,'candidate_model_evaluation':False}))

if __name__=='__main__':main()
