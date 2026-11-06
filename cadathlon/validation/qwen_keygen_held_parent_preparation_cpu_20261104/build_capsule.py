"""Copy an explicit public whitelist into a fresh /tmp evidence capsule."""
from pathlib import Path
import hashlib
import json
import shutil

OUT = Path('/tmp/qwen-keygen-held-parent-public-capsule-20261104')
assert not OUT.exists()
OUT.mkdir()
sha = lambda raw: hashlib.sha256(raw).hexdigest()
pretty = lambda value: (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
rows = []


def copy(source, destination):
    source = Path(source)
    assert source.is_file() and not source.is_symlink()
    target = OUT/destination
    target.parent.mkdir(parents=True, exist_ok=True)
    assert not target.exists()
    raw = source.read_bytes()
    target.write_bytes(raw)
    rows.append(dict(source=str(source), destination=destination, sha256=sha(raw), bytes=len(raw)))


def group(source, destination, names):
    for name in names:
        copy(Path(source)/name, destination+'/'+name)


group('/tmp/qwen-keygen-held-parent-source-handoff-20261104', 'source74',
    ['sources.json','sources.tar.gz','handoff.json','capture.py','static-review.json','writer.patch'])
copy('/tmp/qwen-key-generation-producer-v2-20261104/qwen_key_store.py', 'source74/before_qwen_key_store.py')
group('/tmp/qwen-keygen-held-parent-test-handoff-20261104', 'test-authoring', ['capture.py','handoff.json'])
group('/tmp/qwen-keygen-held-parent-focused-proof-20261104/preparation-v1', 'focused83',
    ['sources.tar.gz','sources-after.json','execution-after.json','report.json','worker-audit.json',
     'tests.xml','tests.log','live-observation.json','executed_driver.py','worker_driver.py','run_config.json',
     'executed_driver.py.patch','worker_driver.py.patch','launch-proposal.json','driver_provenance.json',
     'source-review.json','handoff.json','owner_terminal_review.py','owner-terminal.json','terminal-handoff.json'])
copy('/tmp/qwen-keygen-held-parent-focused-proof-20261104/preparation-v1/sources-before.json', 'focused83/sources.json')
group('/tmp/qwen-actual-keygen-resource-source-review-20261104', 'resource',
    ['REVIEW.md','HELD_PARENT_PROPOSAL.md','resource_allowances.json','source_pin_impact.json','source_refs.json','index.json'])
group('/tmp/qwen-key-generation-actual-root-review-20261104', 'root',
    ['held-parent-source-authorization.json','held-parent74-source-review.json','held-parent83-test-source-review.json',
     'held-parent83-harness-review.json','held-parent83-one-run-authorization.json','held-parent83-terminal-review.json',
     'authority-writer-pin-successor-review.json'])
group('/tmp/qwen-held-parent-tests-transformers-review-20261104', 'peer',
    ['review.json','review_source.py','harness-review.json','review_harness.py'])
group('/tmp/qwen-keygen-authority-boundary-v2-handoff-20261104', 'authority28',
    ['sources.json','sources.tar.gz','parent28_sources.json','writer-prerequisite.patch','handoff.json','README.md'])
copy('/tmp/qwen-keygen-authority-boundary-20261104/keygen_boundary_pins.py', 'authority28/before_keygen_boundary_pins.py')
copy('/tmp/qwen-held-parent-capsule-builder-20261104/verify.py', 'verify.py')
copy(Path(__file__), 'build_capsule.py')
(OUT/'README.md').write_text('''# Held-parent Qwen key-store preparation\n\nThe focused source-bound run passed34/34 tests:18 admission/identity cases,5 disclosed UID/sticky metadata-seam cases and11 tiny NPY/manifest and lifetime cases. The83-file closure contains the exact74-file runtime successor; its writer alone overlays the old71 source. The required constructor now consumes externally pinned canonical parent bytes and checks held/named parent identity, trusted UID/sticky policy before mkdir and at later binding checks. It preserves the unbuffered writer, formats, core34 and recording producer.\n\nThe single authorized run used CPU112–115 and genuine three-module NumPy pins, deterministic arrays of at most16 uint32 values and owned temporary files. The direct tool0a780d returned0 without a persistent session. Driver81341 and child81342 were captured by the driver before direct wait; the owner later verified both PID/group absences. There was no independent external live observation. Root independently admitted the terminal result without claiming live observation or direct reaping. Exact collection/XML order, clean backend audit and all source/current/execution/start/end/archive checks passed.\n\n`source74/` preserves source-only history, patch and original writer. `focused83/` preserves raw34-test evidence and the executed owner readback. `resource/` preserves conditional source-derived RAM/disk/FD estimates, not acquired capacity or a grant. `root/` and `peer/` contain unchanged review receipts. Their earlier pending/authorization-false statements describe their original checkpoints; the later one-run authorization and terminal receipt supersede preparation state only for this synthetic run. Inherited71 tests were not selected, because the constructor API intentionally changed. Signal guards use a disclosed nullcontext seam. Import guards are not an OS sandbox.\n\n`authority28/` is a separately root-reviewed static-only one-literal successor: the future minimum writer pin changes2e60908c to96a20067. All27 other files,58 prerequisite entries and authority/denial semantics are unchanged. The prior39/48 proof was not rerun against this successor. Its inherited historical documentation remains intact; the external delta README records current scope. This pin update neither builds an actual closure nor silently activates recording generation.\n\nNo actual keys, model/private/ciphertext bodies, backend/context, RNG, capacity inspection, key-generation grant, runtime promotion or performance claim are included. Borrowed stores were untouched. All original sources and proofs remain immutable. This capsule is under/tmp only; repository staging and VALIDATION are unchanged.\n\nRun `python -B verify.py` for portable public verification. Optional `--current` also hashes the original74/83 source trees and the three installed NumPy source files; it imports no candidate or NumPy module and opens no model/key bodies. The verifier checks retained terminal claims and links, not a new live process observation. `public_whitelist.json` lists every copied artifact and original path; `index.json` covers all files except itself.\n''')
(OUT/'public_whitelist.json').write_bytes(pretty(dict(schema='qwen-held-parent-public-whitelist-v1',
    copied_files=rows,generated_files=['README.md','public_whitelist.json','index.json'],
    private_or_model_bodies_included=False,actual_role_authorized=False)))
files={p.relative_to(OUT).as_posix():sha(p.read_bytes()) for p in sorted(OUT.rglob('*')) if p.is_file()}
(OUT/'index.json').write_bytes(pretty(files))
print(json.dumps(dict(root=str(OUT),indexed_files=len(files),index_sha256=sha((OUT/'index.json').read_bytes()),
    whitelist_sha256=sha((OUT/'public_whitelist.json').read_bytes())),sort_keys=True))
