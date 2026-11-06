from pathlib import Path
import hashlib,json,shutil
P=Path('/tmp/qwen-expanded-full-model-preflight-v3-20261104')
A=Path('/home/jianming_gatech/CROSS/cadathlon/validation/qwen_expanded_full28_runtime_preparation_cpu_20261104')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):
 src=Path(src);dst=A/dst;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
def flat(src,dst,names):
 for name in names:copy(Path(src)/name,Path(dst)/name)
r=json.loads((P/'worker/report.json').read_bytes());o=json.loads((P/'owner-terminal.json').read_bytes())
assert r['passed']is True and r['tests']==dict(tests=180,failures=0,errors=0,skipped=0)
assert o['reaped']is True and o['exit_code']==0 and o['recorded_group_absent']is True
A.mkdir(exist_ok=False)
flat(P,'proof',['sources.json','sources.tar.gz','launch-draft.json','launch.json','live-observation.json','live-observation-second.json','run.log','owner-terminal.json','cnn-source-review-message.json'])
flat(P/'worker','proof/worker',['report.json','tests.xml','pytest.log','public_plan.json'])
source=json.loads((P/'sources.json').read_bytes());root=Path(r['original_root'])
for name,pin in source.items():
 assert sha(root/name)==pin;copy(root/name,Path('source')/name)
flat('/tmp/qwen-expanded-full-model-runtime-development-capture-20261104','history/runtime-v1',['development-derive-v1.log','development-focused-v1.log','development-focused-v1.xml','observation.json','source-map-after.json','source-map-during-run.json','sources-during-run.tar.gz','terminal.json'])
flat('/tmp/qwen-expanded-full-model-v2-development-proof-20261104','history/runtime-v2',['sources-after.json','sources-before.json','sources.tar.gz','terminal.json','tests.log','tests.xml'])
old=Path('/tmp/qwen-expanded-full-model-preflight-20261104')
flat(old,'history/startup198',['sources.json','sources.tar.gz','launch-draft.json','owner-terminal.json','run.log'])
copy(old/'worker/report.json','history/startup198/report.json')
failed=Path('/tmp/qwen-expanded-full-model-preflight-v2-20261104')
flat(failed,'history/audit199',['sources.json','sources.tar.gz','launch-draft.json','launch.json','live-observation.json','live-observation-second.json','run.log','owner-terminal.json'])
flat(failed/'worker','history/audit199/worker',['report.json','tests.xml','pytest.log','public_plan.json'])
smoke=Path('/tmp/qwen-expanded-preparation-startup-smoke-20261104')
flat(smoke,'history/cache-startup-smoke',['sources.json','sources.tar.gz','scope.json','owner-terminal.json','run.log','wrong-digest-pre-admission.json','wrong-digest-pre-admission.log'])
flat(smoke/'worker','history/cache-startup-smoke/worker',['report.json','tests.xml','pytest.log','public_plan.json'])
flat('/tmp/qwen-expanded-preparation-entry-root-probe-20261104/old-v1','history/entry-old',['driver.py.txt','probe.py','report.json','sources-after.json','sources-before.json','sources.tar.gz'])
flat('/tmp/qwen-expanded-preparation-entry-root-replay-20261104','history/entry-root-replay',['probe.py','review-artifacts.tar.gz','review.json','review.py','selected-after.json','selected-before.json','corrected-stderr.log','corrected-stdout.log','old-stderr.log','old-stdout.log'])
flat('/tmp/qwen-expanded-preparation-entry-all-boundaries-20261104','history/entry15',['driver.py.txt','owner.json','probe.py','report.json','sources-after.json','sources-before.json','sources.tar.gz'])
for label,src in [('site-old','/tmp/qwen-expanded-preparation-site-negative-20261104'),('site-fixed','/tmp/qwen-expanded-preparation-site-successor-v2-20261104')]:
 flat(src,'history/'+label,['driver.py.txt','probe.py','report.json'])
 for name in json.loads((Path(src)/'report.json').read_bytes())['source_after']:copy(Path(src)/name,Path('history')/label/name)
copy('/tmp/qwen-expanded-preparation-site-successor-20261104.log','history/site-draft-failure/trace.log')
flat('/tmp/qwen-expanded-full-preparation-driver-v4-20261104','history/site-draft-failure',['prepare_expanded.py','probe_site_roots.py'])
for label in ('negative-v1','negative-v2','negative-v3','successor-v1'):
 src=Path('/tmp/qwen-unbatched-softmax-wrapper-review-20261104')/label
 for name in ['audit.json','check-observation.json','refresh-observation.json','report.json','run.py','source_manifest_after.json','source_manifest_before.json','sources.tar.gz','test_wrapper_callback.py','tests.log','tests.xml']:
  if(src/name).is_file():copy(src/name,Path('history/softmax-wrapper')/label/name)
reviews=Path('/tmp/qwen-expanded-full-successor-security-review-20261104')
for src in sorted(reviews.iterdir()):
 if src.is_file()and src.suffix in ('.py','.json'):copy(src,Path('reviews/security-runtime')/src.name)
flat('/tmp/qwen-expanded-full-contract-security-review-20261104','reviews/security-bounds',['derive.py','review.json'])
copy('/tmp/build_qwen_expanded_preparation_capsule_20261104.py','archive_builder.py')
print(json.dumps(dict(artifact=str(A),source_count=len(source),report_sha256=sha(P/'worker/report.json'),archive_sha256=sha(P/'sources.tar.gz'))))
