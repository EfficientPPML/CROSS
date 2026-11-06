#!/usr/bin/env python3
"""Copy only enumerated public artifacts. No test or archived-code execution."""
import difflib
import hashlib
import json
from pathlib import Path
import stat

BUILD = Path('/tmp/qwen-proc-self-capsule-build-20261104')
OUT = Path('/tmp/qwen_proc_self_audit_preparation_cpu_20261104')
OUT.mkdir()
copies = []
generated = []
bindings = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def stable(path):
    before = path.lstat()
    assert stat.S_ISREG(before.st_mode) and not path.is_symlink(), path
    raw = path.read_bytes()
    after = path.lstat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    assert len(raw) == after.st_size
    return raw


def put(name, raw):
    target = OUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(raw)


def cp(source, dest, expected=None):
    source = Path(source)
    raw = stable(source)
    if expected is not None:
        assert sha(raw) == expected, str(source)
    put(dest, raw)
    copies.append(dict(source=str(source), destination=dest, sha256=sha(raw), bytes=len(raw)))


def group(source, dest, names):
    for name in names:
        cp(Path(source) / name, dest + '/' + name)


def gen(name, raw):
    if isinstance(raw, str):
        raw = raw.encode()
    elif not isinstance(raw, bytes):
        raw = (json.dumps(raw, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    put(name, raw)
    generated.append(name)


proofs = {
    'client': dict(source='/tmp/qwen-client-kernel-allowlist-tests-v2-20261104', proof='/tmp/qwen-client-kernel-focused-proof-20261104/preparation-v2', count=472, cases=24, map='208049d311bc26bb8db8ce71c610c6c484dfbec064e1b26bbbe5f51483a1ec40', tar='9ed75e3c56eaf743a34f591d6978af69394de93a8ea806a26bb5fb9b98d4a617', label='client24', affinity=[108,109,110,111]),
    'keygen': dict(source='/tmp/qwen-keygen-proc-self-tests-v2-20261104', proof='/tmp/qwen-keygen-proc-self-focused-proof-20261104/preparation-v2', count=92, cases=32, map='bb9bee8a343f45e65fdb344240bdf3b5f7fc66b9ef5167c67909b11e9d81d53d', tar='f6368d99e879e5e9e3629b05a12fc09c4748709ff3c393f7811e606303b9bc56', label='keygen32', affinity=[112,113,114,115]),
}
common = ['executed_driver.py','worker_driver.py','run_config.json','driver_provenance.json','executed_driver.py.patch','handoff.json','launch-proposal.json','sources-before.json','sources-after.json','execution-after.json','sources.tar.gz','tests.log','tests.xml','worker-audit.json','live-observation.json','report.json']
for label, info in proofs.items():
    group(info['proof'], label + '/proof', common)
    assert sha(stable(Path(info['proof'])/'sources-before.json')) == info['map']
    assert sha(stable(Path(info['proof'])/'sources.tar.gz')) == info['tar']
group(proofs['client']['proof'], 'client/proof', ['README.md','source-correction.json'])
group(proofs['keygen']['proof'], 'keygen/proof', ['worker_driver.py.patch','source-handoff.json'])

# Small readable selections; the two exact full source archives remain authoritative.
for label, names in {
    'client':['TEST_SCOPE.md','before_qwen_client_dependencies.py','conftest.py','kernel_probe.py','parent464-sources.json','runtime-sources.json','selected_tests.json','test_kernel_allowlist.py','runtime/qwen_client_dependencies.py'],
    'keygen':['SCOPE.md','conftest.py','proc_self_probe.py','runtime_sources.json','test_origin.json','test_selection.json','test_proc_self_integration.py','runtime/keygen_dependencies.py'],
}.items():
    for name in names:
        dest = label + '/selected-source/' + name
        cp(Path(proofs[label]['source']) / name, dest)
        bindings[dest] = dict(archive=label, member=name)

# Exact root admissions, historical direct tool ownership and late observation.
for label in ('client24','keygen32'):
    group('/tmp/qwen-proc-self-root-review-20261104','root-review',[label+n for n in ['-preexecution-review.json','-run-grant.json','-root-live-observation.json','-direct-owner.json','-terminal-review.json']])
for name in ('preexecution','grants','terminal'):
    filename = 'root_proc_self_' + name + '_20261104.py'
    cp('/tmp/' + filename, 'root-review/' + filename)

# Closed tiny-artifact maps authorize no recursive traversal or alias following.
for label, source, expected_count, extras in [
    ('client','/tmp/qwen-client-kernel-terminal-child-retention-20261104',96,['files.json','handoff.json','origins.json','retain_children.py','retention-review.json']),
    ('keygen','/tmp/qwen-proc-self-child-observations-20261104',36,['files.json','handoff.json']),
]:
    source = Path(source)
    mapping = json.loads(stable(source/'files.json'))
    assert len(mapping) == expected_count
    for name, expected in sorted(mapping.items()):
        assert not name.startswith('/') and '..' not in Path(name).parts
        cp(source/name, label+'/children/'+name, expected)
    group(source,label+'/children',extras)

# Runtime correction provenance, without duplicate predecessor source archives.
group('/tmp/qwen-client-kernel-allowlist-successor-handoff-20261104','client/runtime-correction',[
    'author_successor.py','before_qwen_client_dependencies.py','correction.json','handoff.json','hold_source.py','parent464-sources.json','qwen_client_dependencies.patch','shared_keygen_rule.json','sources.json'])
group('/tmp/qwen-actual-client-source-peer-review-20261104','client/source-review',['review.json','review_source.py'])
group('/tmp/qwen-keygen-actual-runtime-v3-handoff-20261104','keygen/runtime-correction',[
    'README.md','before_keygen_dependencies.py','capture_source.py','correction.json','handoff.json','keygen_dependencies.patch','parent85-sources.json','sources.json'])

# Unexecuted test-parent ownership correction and original source map.
group('/tmp/qwen-client-kernel-test-ownership-successor-handoff-20261104','client/test-ownership-correction',[
    'author_successor.py','before_test_kernel_allowlist.py','correction.json','handoff.json','parent472-sources.json','sources.json','test_parent_ownership.patch'])
old_harness = ['executed_driver.py','worker_driver.py','run_config.json','driver_provenance.json','executed_driver.py.patch','handoff.json','launch-proposal.json']
group('/tmp/qwen-client-kernel-focused-proof-20261104/preparation-v1','client/unrun-harness-v1',old_harness+['README.md'])
group('/tmp/qwen-keygen-proc-self-focused-proof-20261104/preparation-v1','keygen/unrun-harness-v1',old_harness+['worker_driver.py.patch','source-handoff.json'])

# Unexecuted keygen negative-read scope correction: retain only three changed files.
for name in ('SCOPE.md','proc_self_probe.py','test_origin.json'):
    before = Path('/tmp/qwen-keygen-proc-self-tests-20261104')/name
    after = Path(proofs['keygen']['source'])/name
    cp(before, 'keygen/negative-read-correction/before/'+name)
    patch = ''.join(difflib.unified_diff(stable(before).decode().splitlines(True), stable(after).decode().splitlines(True), fromfile='before/'+name, tofile='after/'+name))
    gen('keygen/negative-read-correction/'+name+'.patch.txt',patch)
group('/tmp/qwen-keygen-proc-self-test-handoff-20261104','keygen/negative-read-correction/unrun-source-handoff',['capture_source.py','handoff.json','sources.json','static-review.json'])
group('/tmp/qwen-keygen-proc-self-test-handoff-v2-20261104','keygen/final-source-handoff',['capture_source.py','handoff.json','sources.json','static-review.json'])

lineages = [
    dict(label='client464v2',archive='client',prefix='runtime/',replace={},map='client/runtime-correction/sources.json'),
    dict(label='client464 predecessor',archive='client',prefix='runtime/',replace={'qwen_client_dependencies.py':'client/runtime-correction/before_qwen_client_dependencies.py'},map='client/runtime-correction/parent464-sources.json'),
    dict(label='client472 unrun ownership predecessor',archive='client',prefix='',replace={'test_kernel_allowlist.py':'client/test-ownership-correction/before_test_kernel_allowlist.py'},map='client/test-ownership-correction/parent472-sources.json'),
    dict(label='keygen85v3',archive='keygen',prefix='runtime/',replace={},map='keygen/runtime-correction/sources.json'),
    dict(label='keygen85v2 predecessor',archive='keygen',prefix='runtime/',replace={'keygen_dependencies.py':'keygen/runtime-correction/before_keygen_dependencies.py'},map='keygen/runtime-correction/parent85-sources.json'),
    dict(label='keygen92 unrun negative-read predecessor',archive='keygen',prefix='',replace={n:'keygen/negative-read-correction/before/'+n for n in ('SCOPE.md','proc_self_probe.py','test_origin.json')},map='keygen/negative-read-correction/unrun-source-handoff/sources.json'),
]
summary = dict(schema='qwen-proc-self-public-capsule-v1', passed=True, tests=56, client_cases=24, keygen_cases=32,
    scope='Real installed audit hooks and os.open wrappers, own-process proc metadata only, with explicit synthetic admissions.',
    independent_live_observation=False, actual_role_execution=False, backend_imports=False, borrowed_body_reads=False, capacity_measurement=False, rng=False, key_generation=False, model_execution=False, encrypted_execution=False,
    runtime_scope={'client':'464v2: one dependency overlay; 463 unchanged','keygen':'85v3: one dependency overlay; 84 unchanged'},
    prior61_and95_retested=False, proc_contents_retained=False, source_only_history_not_observed_failures=True,
    proof_owner={'client':{'session':68374,'terminal_chunk':'170b97','exit_code':0,'reaped':True},'keygen':{'session':67367,'terminal_chunk':'74d265','exit_code':0,'reaped':True}},
    proofs={label:dict(source_count=v['count'],tests=v['cases'],source_manifest_sha256=v['map'],source_archive_sha256=v['tar'],current_source=v['source'],root_label=v['label'],affinity=v['affinity']) for label,v in proofs.items()},
    readable_source_bindings=bindings,lineages=lineages)
gen('summary.json',summary)
gen('HISTORY.md', '''The two final preparation-v2 proofs are the only runs represented here: client24 and keygen32.

The client runtime correction preserves its 464-file predecessor, with one dependency overlay. The keygen correction preserves its 85-file predecessor, also with one dependency overlay. Both replace the general-alias-first rejection of three literal own-process proc paths with the same narrow readonly rule. Source-only finding and correction receipts retain their original statuses.

The original client472 test parent could skip direct wait if killpg itself raised during cleanup. That was found in source review, before any run; final472v2 nests direct wait in finally and validates the child's reported process group even when a parent observation races. The old test and unrun harness are retained. No cleanup failure from an execution is claimed.

The original keygen92 negative operations reused a read helper. Source review identified that an unexpectedly admitted operation could read bytes. Final92v2 negative operations only open and close; the three changed predecessor files, exact maps, derived diffs and unrun harness preserve the correction. No observed negative-read failure is claimed.

Full predecessor source maps can be reconstructed from the two final archives and explicit old files. The portable verifier checks all six lineage maps. Frozen documents saying unexecuted/source-only record their authoring time; final reports and root grants/terminal reviews separately record the later authorized runs.

The prior 61 actual-keygen scaffold recording checks and 95 pure client-authority checks were separate preparations. This capsule makes no retest or actual authority claim for either. Runtime source archives include inherited public source/provenance, not generated private payloads or backend execution evidence.
''')
gen('README.md', '''# Qwen own-process audit preparation — 56 focused checks

The client **24** and keygen **32** focused cases passed under their installed audit hooks and `os.open` wrappers. Each test uses an isolated child and explicit synthetic admission. Positive cases read only the child's `/proc/self/status`, `/proc/self/cgroup` or `/proc/self/mountinfo`; receipts retain booleans and denial metadata, not proc contents. This establishes the narrow proc-path boundary under those fixtures. It does not establish capacity, actual role readiness, key generation, model execution or encrypted execution.

Client coverage comprises six allowed reads, fifteen flag/spelling denials and three unchanged464 controls. Keygen coverage includes readonly flags, builtin literals, forbidden flags/aliases/protected paths, sticky denial, body-error and restoration controls. Argument-type rejection applies at the `os.open` wrapper. Builtin audit tests authenticate the event's observed literal string; they do not infer a pre-normalized high-level argument type.

`client/proof` and `keygen/proof` retain the exact final472v2/92v2 source archives and maps, harnesses, ordered XML, logs and backend-denial audits. `client/children` retains 96 tiny files for 24 children; `keygen/children` retains 32 observations plus their exact review/origin files. No original pytest tree, alias, generated owned payload or proc content is copied. Two source archives are included; thin old files reconstruct predecessor maps without duplicate full archives.

Root owned sessions68374 and67367 and directly observed normal exit0/reap (tool170b97 and74d265). Root's separate attempted live observation arrived after exit. **No independent live observation is claimed.** Owner and terminal receipts preserve that distinction. Source and test correction histories were unexecuted drafts, not failed runs; see [HISTORY.md](HISTORY.md).

The actual client/keygen entry paths, body readers, backend imports, RNG, capacity evaluation and any execution grant remain outside this proof. Existing 61/95 preparations are separate historical evidence. No private key, ciphertext, model, calibration or token body was read to build or verify this capsule.

`copy_origins.json` is the explicit source-to-destination/hash whitelist for copied artifacts. `WHITELIST.json` is the complete closed file list; `SHA256SUMS` indexes every file except itself. Generated metadata and exact source bindings are declared separately. Verify public bytes, tar closure, source lineage, ordered test evidence and child receipt links with:

```sh
python -I -B verify.py
python -I -B verify.py --current
```

The optional current mode reads only the 472 and 92 declared public source files. Neither mode imports candidate modules, reruns tests or probes, revisits original pytest trees, stats borrowed stores, evaluates capacity, or executes any preserved reviewer script. Root controls later repository publication and admission.
''')
cp(BUILD/'verify.py','verify.py')
cp(Path(__file__),'assembly_script.py')
generated += ['copy_origins.json','WHITELIST.json','SHA256SUMS']
put('copy_origins.json',(json.dumps(dict(schema='explicit-public-copy-origins-v1',copies=sorted(copies,key=lambda x:x['destination']),generated=sorted(generated)),sort_keys=True,indent=2)+'\n').encode())
names = sorted([x['destination'] for x in copies] + generated)
assert len(names) == len(set(names))
put('WHITELIST.json',(json.dumps(names,indent=2)+'\n').encode())
put('SHA256SUMS',''.join(sha(stable(OUT/name))+'  '+name+'\n' for name in names if name!='SHA256SUMS').encode())
print(json.dumps(dict(capsule=str(OUT),files=len(names),indexed_files=len(names)-1,copied_files=len(copies),generated_files=len(generated),index_sha256=sha(stable(OUT/'SHA256SUMS')),whitelist_sha256=sha(stable(OUT/'WHITELIST.json')),origins_sha256=sha(stable(OUT/'copy_origins.json'))),sort_keys=True))
