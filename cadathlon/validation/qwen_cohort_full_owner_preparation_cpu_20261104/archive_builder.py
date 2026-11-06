from pathlib import Path
import hashlib,json,tarfile,difflib
BASE=Path('/home/jianming_gatech/CROSS/cadathlon/validation')
OUT=BASE/'qwen_cohort_full_owner_preparation_cpu_20261104'
PROOF=Path('/tmp/qwen-cohort-full-owner-combined-preflight-20261104/preflight-v1')
SOURCE=Path('/tmp/qwen-softmax-cohort-full-owner-integration-v2-20261104')
PARENT=Path('/tmp/qwen-expanded-full-model-runtime-final-v3-20261104')
HISTORY=Path('/tmp/qwen-cohort-full-integration-history-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
assert not OUT.exists();OUT.mkdir()
copied={};generated=[]
def put(name,raw,origin=None):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
 if origin is not None:copied[name]=dict(origin=str(origin),sha256=sha(raw),bytes=len(raw))
 else:generated.append(name)
def copy(src,name):
 src=Path(src);assert src.is_file()and not src.is_symlink();put(name,src.read_bytes(),src)
def dump(name,data):put(name,(json.dumps(data,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())
for n in ['sources.json','sources.tar.gz','assembly.json','launch.json','executed_outer_driver.py','live-observation.json','live-observation-2.json','run.log','outer-report.json','owner-terminal.json','original-sources-after.json','execution-sources-after.json']:
 copy(PROOF/n,'proof/'+n)
for n in ['report.json','public_plan.json','pytest.log','tests.xml']:copy(PROOF/'worker'/n,'proof/worker/'+n)
for n in ['review.json','review.py','collection_readback.json']:copy(PROOF/'peer-review'/n,'peer-review/'+n)
root=Path('/tmp/qwen-cohort-full-owner-root-review-20261104')
for n in ['review.json','verify.py','live-observation.json','terminal-admission.json']:copy(root/n,'root-review/'+n)
coll=PROOF.parent/'collection-v1'
for n in ['report.json','stdout.log','stderr.log']:copy(coll/n,'collection/'+n)
copy('/tmp/collect_qwen_cohort_combined_20261104.py','collection/executed_driver.py')
copy(PROOF.parent/'collection-startup-history/review.json','history/collection-startup/review.json')
m=json.loads((PROOF/'sources.json').read_bytes());origin=json.loads((SOURCE/'integration_origin.json').read_bytes())
parent=json.loads((SOURCE/'cohort_parent199_sources.json').read_bytes())
changes=origin['changed_parent_files'];adds=sorted(set(m)-set(parent))
for n in list(changes)+adds:copy(SOURCE/n,'selected-source/'+n)
for n in changes:
 copy(PARENT/n,'provenance/parent199-overlays/'+n)
 old=(PARENT/n).read_text().splitlines(keepends=True);new=(SOURCE/n).read_text().splitlines(keepends=True)
 patch=''.join(difflib.unified_diff(old,new,fromfile='parent199/'+n,tofile='current208/'+n))
 put('patches/'+n+'.patch',patch.encode())
copy(SOURCE/'cohort_parent199_sources.json','provenance/parent199/sources.json')
parent_arc=BASE/'qwen_expanded_full28_runtime_preparation_cpu_20261104'
for a,b in [('proof/worker/report.json','report.json'),('proof/owner-terminal.json','owner-terminal.json')]:copy(parent_arc/a,'provenance/parent199/'+b)
leaf=BASE/'qwen_softmax_cohort_synthetic_cpu_20261104'
for a,b in [('proof/sources-before.json','sources.json'),('proof/report.json','report.json'),('proof/terminal_observation.json','owner-terminal.json')]:copy(leaf/a,'provenance/cohort197/'+b)
for d in ['development-v1','development-v2','plan-development-v1']:
 for n in ['terminal.json','tests.log','tests.xml']:copy(HISTORY/d/n,'history/'+d+'/'+n)
copy(HISTORY/'development-v2/test_cohort_full_integration.py','history/development-v2/test_cohort_full_integration.py')
copy(HISTORY/'development-v1/post_terminal_sources.json','history/development-v1/post_terminal_sources.json')
oldmap=json.loads((HISTORY/'development-v1/post_terminal_sources.json').read_bytes());sparse={}
with tarfile.open(HISTORY/'development-v1/sources.tar.gz')as t:
 ms=t.getmembers();assert len(ms)==205 and all(x.isfile()for x in ms)
 blobs={x.name:t.extractfile(x).read()for x in ms};assert {n:sha(b)for n,b in blobs.items()}==oldmap
 for n,h in oldmap.items():
  if m.get(n)!=h:
   put('history/development-v1/sparse-source/'+n,blobs[n],str(HISTORY/'development-v1/sources.tar.gz')+'::'+n);sparse[n]=h
assert len(sparse)==8
reconstruction=dict(schema='qwen-cohort-development-source-reconstruction-v1',source_count=205,base='proof/sources.tar.gz',base_source_manifest_sha256=sha((PROOF/'sources.json').read_bytes()),historical_full_map='post_terminal_sources.json',sparse_files=sparse,retained_original_tar_sha256=sha((HISTORY/'development-v1/sources.tar.gz').read_bytes()),captured_after_terminal=True,startup_source_snapshot_available=False,scope='Exact retained post-terminal205 bytes reconstructed from current208 archive plus eight sparse historical files; not a claim of startup or execution-time source capture.')
dump('history/development-v1/reconstruction.json',reconstruction)
dump('provenance/parent-links.json',dict(parent199=dict(readme='../qwen_expanded_full28_runtime_preparation_cpu_20261104/README.md',source_manifest_sha256=origin['parent_manifest_sha256'],report_sha256=origin['parent_preparation_report_sha256'],source_archive_sha256='99de0e64165e26b9bd51352a1cbcd982187fdb3d093643f2a47afe3ddae26caa'),cohort197=dict(readme='../qwen_softmax_cohort_synthetic_cpu_20261104/README.md',source_archive_sha256=origin['cohort_leaf_source_archive_sha256'],report_sha256=origin['cohort_leaf_proof_report_sha256'])))
put('README.md',b'''# Qwen cohort full-owner synthetic preparation

Completed public CPU preparation: **330 tests passed**, no failures, errors, skips or audit denials. Owner session90010 exited0 and was reaped (tool6e7331); observed collector3397120 and worker3397122, including their actual process groups, were absent. The shared infrastructure wrapper16918 is outside retirement claims.

This integrates the reviewed cohort softmax into the full-owner attention constructor. The full-owner tests use **synthetic parameter/authentication, normalization, head, and linear seams**, with the real cohort softmax and PV owners. They verify actual V23 to PV20 (S>1) / PV22 (S1), K1 bypass, original positions, callback retention and failure quarantine. The original learned whole-model and HE execution remain unexecuted. This is not model accuracy, calibration, heldout, peak-memory, latency, or acceleration evidence, and grants no actual execution.

The exact208-source archive has193 unchanged parent files, six explicitly pinned overlays and nine additions. Full owner/contract, protected coefficients, final RMS3 to vocabulary2, all38 vocabulary blocks and the166-key plan remain bound. The selected330 are the predecessor180, cohort136 and integration14; final XML is authoritative. Public plans cover S1 through32 and all28 layers. At S32, planned softmax refresh calls are3760 versus11656 literal, and total planned refresh calls5035 versus12931. These are operation counts, not measured acceleration.

`proof/` retains the source manifest/archive, worker report/XML/public plan/log, launch environment, direct-wait outer report, authoritative owner receipt and before/after maps. It records five fresh public derivations and45 loaded local origins. `selected-source/` and `patches/` expose only the changed/new public files. `root-review/` and `peer-review/` contain independent source/terminal/count receipts. Source metadata recording preparation as pending is preserved byte-exact; the terminal reports establish its later completion.

The reviewed [unbatched199 parent](../qwen_expanded_full28_runtime_preparation_cpu_20261104/README.md) and [cohort197 leaf](../qwen_softmax_cohort_synthetic_cpu_20261104/README.md) remain separate. Their public manifests/reports/terminal receipts are included for portable provenance; historical full trees/tars are not duplicated. The six old parent files let the verifier reconstruct the complete199 source map against the208 archive.

Development history is not aggregated into330. The first12-case run had11 passes and one wrong exception expectation; its four joined numerical checks passed. The later focused run passed8 selected cases (four deselected), and the plan run passed2. The first205-source snapshot was captured only after terminal, with no startup source map. Its exact full map can be reconstructed from the208 archive plus eight sparse historical files; this does not upgrade its execution provenance. The collection startup SyntaxError occurred before collection/source execution and is retained separately. Final collection found330 unique nodes and executed zero tests.

Only explicitly whitelisted public source, metadata, aggregate plan, logs, XML and review artifacts are copied. No model tensors, tokens, private calibration/validation outputs, key bodies or ciphertexts are included or followed by the verifier. No expanded source tree, symlink or bytecode cache is included. This capsule was prepared without staging or changing VALIDATION.md; root owns those steps.

Run `python -I -B verify.py` for portable public-byte validation, or add `--current` to compare the frozen208 source tree. Neither mode imports the measured runtime, reruns numerical tests, follows private descriptors or performs model/HE work.
''')
copy(Path(__file__),'archive_builder.py')
put('.gitattributes',b'*.json -filter -diff -merge text\n*.py -filter -diff -merge text\n*.md -filter -diff -merge text\n*.xml -filter -diff -merge text\n*.log -filter -diff -merge text\n*.patch -filter -diff -merge text\n*.gz -filter -diff -merge binary\n')
# Verifier is prepared separately and included before index generation.
dump('public_whitelist.json',dict(schema='public-only-whitelist-v1',copied_files=copied,generated_files=sorted(generated+['public_whitelist.json','verify.py','SHA256SUMS.json']),excluded=['model/token/private observation bodies','keys/ciphertexts','expanded duplicate source trees','symlinks/bytecode caches'],scope='Public CPU synthetic preparation only; no runtime execution by archive verification.'))
print(json.dumps(dict(path=str(OUT),copied=len(copied),generated=len(generated),sparse_history=len(sparse))))
