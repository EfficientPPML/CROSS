from pathlib import Path
import hashlib,json,shutil,tarfile,io
ROOT=Path('/home/jianming_gatech/CROSS')
DEST=ROOT/'cadathlon/validation/bert_prepared_bsgs_server_preflight_cpu_20261104'
LAB=Path('/tmp/bert-prepared-bsgs-server-component-20261104')
MEASURED=Path('/tmp/bert-prepared-bsgs-server-preflight-v2-20261104')
FAILED=Path('/tmp/bert-prepared-bsgs-server-preflight-v1-20261104')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def copy(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
assert sha(MEASURED/'report.json')=='31f9120534e1d23843b3f7a5ec7e68c77963ac7456e5e0b492557aedd26b4315'
assert sha(MEASURED/'sources.tar.gz')=='fd46274e075565bcec545cca13f03470d4b9506b6327ffd7f403a756acbcaa94'
DEST.mkdir(exist_ok=False)
for source,target in ((MEASURED,DEST/'measured'),(FAILED,DEST/'history/failed-installed-v1')):
 for name in ('report.json','sources.tar.gz','source_manifest.json','tests.log','tests.xml','installed.log','installed.json','terminal_observation.json'):
  copy(source/name,target/name)
 copy(Path(str(source)+'.log'),target/'collector.log')
for name in ('assembly_schema_note.json','combined-v1.log','combined-v1.xml',
             'pending-dependencies-v1/snapshot.json','pending-dependencies-v1/sources.tar.gz',
             'worker-collector-v1-sources.json','worker-collector-v1-sources.tar.gz','worker-collector-v1.log','worker-collector-v1.xml',
             'worker-collector-v2-sources.json','worker-collector-v2-sources.tar.gz','worker-collector-v2.log','worker-collector-v2.xml'):
 copy(LAB/'development'/name,DEST/'history/development'/name)
for version in (1,2):copy(Path(f'/tmp/combined-dependency-development-v{version}.xml'),DEST/f'history/dependency-development-v{version}.xml')
(DEST/'history/dependency-development-terminal.json').write_text(json.dumps({'scope':'Sibling-reported authoritative development tool terminals and XML only; no separate before/after source capsule was collected for these runs. Final286 proof supplies exact-source coverage.','runs':[{'session_id':84832,'observed_exit_code':0,'session_reaped':True,'tests':58,'elapsed_s':1.47},{'session_id':58591,'observed_exit_code':0,'session_reaped':True,'tests':89,'elapsed_s':2.05}]},indent=2)+'\n')
copy(Path(__file__),DEST/'archive.py')
(DEST/'README.md').write_text('''# Prepared-cache and BSGS server preparation

The isolated server successor passed **157 public/synthetic tests and 25 real installed-entrypoint denial/restoration checks** in 6.7214 seconds on CPUs108–111. Session69570 exited0 and was reaped; the collector and both direct children were absent. All286 current/start/end/archive sources matched. The installed probe checked39 loaded local modules plus the explicit worker/helper origins.

This is preparation evidence. It performed no context/key construction, retained ciphertext read, server computation, native model operation or decryption. No combined encrypted accuracy, latency, peak-memory or ranked-workload result is established. Actual launch still requires a separate reviewed decision.

The candidate preserves all265 parent files and adds only named source/provenance files. Before loading runtime or gate modules, the fresh worker checks the external source-manifest digest, complete frozen bytes, static graph resolution and preloaded helper origins. It authenticates stage1→stage2→stage3, uses the unchanged eight-input handler with prepared_rotation_cache=True, and borrows only the upload subtree. Graph/linear each have two prepared entries capped at2GiB logical bytes; refresh remains ordinary. The4GiB aggregate excludes public master/mmap backing, compiler/conversion/arithmetic temporaries, ciphertexts and physical allocation overhead.

Outputs are bound to a separate fresh0700 response root and inode. The active ordinary server's original download, original private request ledger and external key/model/input workspaces are untouched. Failed or uncertain attempts retain created response artifacts. A future client successor must bind this separate response contract and completed proof before decoding. The expanded286 source inventory is18,545,970 uncompressed bytes; its readers use32MiB total/16MiB member bounds with exact hashes. Historical16MiB-only stage5 source gates cannot be reused silently.

Technical readiness now authenticates the exact combined265 construction proof, authenticated all74-matrix118⊆130 key audit, and completed full-BSGS150-source/82-test ideal proof plus root's2171 terminal attestation. execution_authorized remains false. Genuine prepared rotation parity is prior component evidence; the still-pending ordinary-group result is not used.

History is separate. Initial preflight29997 passed157 tests but failed its installed probe because the new probe imported flat key_gen before jaxite_word registered that alias. The complete286 failure source/report/log/terminal capsule is retained. Only that new probe import order changed; final69570 reran both phases. The earlier283-source development snapshot preserves explicitly pending dependency readiness and124 tests. Initial test syntax and error-message-matcher failures are retained with their exact six-source snapshots. Dependency58/89 development checks retain XML and reported tool terminals only, not an invented source-bound wrapper.

Run `python verify.py` for portable archive verification. Add `--current /tmp/bert-prepared-bsgs-server-component-20261104` to check current frozen files. The verifier does not import runtime code or read external keys, model payloads, inputs or ciphertexts. Nested archived predecessor bundles contain only public source/provenance; no key/CT/private arrays are included.
''')
print(DEST)
