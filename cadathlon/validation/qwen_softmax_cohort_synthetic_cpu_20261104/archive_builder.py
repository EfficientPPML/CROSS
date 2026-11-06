from pathlib import Path
import hashlib,json,shutil
A=Path('/home/jianming_gatech/CROSS/cadathlon/validation/qwen_softmax_cohort_synthetic_cpu_20261104')
P=Path('/tmp/qwen-causal-softmax-cohort-public-proof-20261104/development-v1')
H=Path('/tmp/qwen-softmax-cohort-binding-review-20261104')
A.mkdir();copied={}
def copy(source,name):
 source=Path(source);assert source.is_file()and not source.is_symlink();target=A/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target);copied[name]=str(source)
for name in ('report.json','sources-before.json','sources-after.json','execution-sources-after.json','sources.tar.gz','tests.xml','stdout.log','stderr.log','executed_driver.py','process-observed.json','terminal_observation.json'):copy(P/name,'proof/'+name)
copy(P/'peer-review/security-source-review.json','peer-review/security-source-review.json')
copy('/tmp/qwen-causal-softmax-cohort-successor-20261104/COHORT_DESIGN.md','COHORT_DESIGN.md')
copy('/tmp/qwen-causal-softmax-cohort-successor-20261104/cohort_parent_sources.json','parent191_sources.json')
common=('sources-before.json','sources-after.json','sources.tar.gz','tests.xml','terminal.json')
for version in ('negative-v1','successor-v1','guard-order-negative-v1','successor-v2'):
 for name in common:copy(H/version/name,'history/'+version+'/'+name)
for name in ('terminal-replay.json','tests-replay.xml','replay-stdout.log','replay-stderr.log'):copy(H/'guard-order-negative-v1'/name,'history/guard-order-negative-v1/'+name)
copy(__file__,'archive_builder.py')
(A/'README.md').write_text('''# Qwen causal-softmax cohorts: synthetic preparation

This unstaged public capsule records **136 passing synthetic tests** from **197 exact source files**, including **191 unchanged expanded-v3 parent files**. The isolated owner groups disjoint query rows only when their original Newton iteration count and inverse-refresh bound agree. Original per-K rounded coefficients, multipliers, positions, domains and the 166-key plan remain bound. The full-model join still selects the literal-K owner.

The numerical checks cover every slot for sequence lengths 1–32 with both five- and six-squaring schedules, both full tiles against the literal predecessor, all 28 public rule descriptors, mixed coefficients, inactive dummy-one lanes and refresh perturbations, actual-scale metadata, PV layout and real K1 V bypass. Callback regressions cover primary exceptions and retained original payloads. One NumPy deprecation warning comes from the deliberate shape-mutation test; no test failed, errored or skipped.

At S32 across 28 layers the source/synthetic count is **3,760 softmax refresh calls**, versus **11,656 literal calls**, plus 20,200 rotations and 32,196 ciphertext products. Eight active plaintext arrays occupy 4 MiB of logical data per cohort, excluding transient, ciphertext, compiler/device and key allocations. These are operation/storage counts, **not measured acceleration, peak memory, or encrypted accuracy**. No parameter, token, model, private, key or ciphertext body was read. No whole learned model, HE or TPU run, source promotion, path switch or execution grant is represented.

`proof/report.json` is pinned to `7c94effe3dc5490d3499e66ecc7ce0bae8bb5ec72b4e6cb0a9c109a5016d5a61`; the 197-member archive is `023cb45bfa7e966846c54e6e1244d53be0094699185a14ce1e7db55988e7d212`. The source map is `69bbd4561761a400fc676dee63ab31b50499096d100a6752cee1567d6819b1fd`. All current, execution, start/end and tar hashes matched after the run. Owner session31548 exited0 and was reaped; observed collector3175372 and test child3175385 and their recorded groups were absent. Shared tool wrapper16918 is explicitly outside owned-process retirement claims. Root independent review is pending.

`peer-review/security-source-review.json` is a scoped source-only algebra/layout/lifetime review, not an independent test run. It was written while the 136-test run was pending; the later terminal proof is separate.

Preserved history:

- `negative-v1`: six expected failures and two passing controls across wrapper-accessor and new vector-factory substitution checks.
- `successor-v1`: the same eight tests passed. It does not cover the later check-invocation ordering issue.
- `guard-order-negative-v1`: two expected failures and one passing practical refresh-substitution control show that a replaced check was invoked before rejection. The original tool-terminal response was unavailable after truncation; that uncertainty remains in `terminal.json`. An identical bounded replay on the same frozen sources exited1, retained separately in `terminal-replay.json`, logs and `tests-replay.xml`.
- `successor-v2`: all eleven cases passed after binding wrapper identity/accessors before the inherited guard invokes `check`. The current owner is identical to this repaired runtime. The global vector-factory check is local hardening, not a general arbitrary-Python tamper guarantee.

Only explicit public files were copied. Source trees are stored as exact archives, without expanded duplicates, bytecode, symlinks, model arrays or private bodies. `public_whitelist.json` records every source path. Run `python -I -B verify.py` for portable public-byte verification, or add `--current` to rehash the known 197 current source files. Neither mode follows private descriptors, imports runtime/model/backend code, reruns tests, or replays arithmetic.
''')
(A/'public_whitelist.json').write_text(json.dumps(dict(copied_files=copied,generated_files=['README.md','verify.py','public_whitelist.json','SHA256SUMS.json'],exclusions=['expanded source directories','symlinks and bytecode','private/token/model/input/key/ciphertext bodies'],scope='public source and synthetic proof only'),sort_keys=True,indent=2)+'\n')
print(A)
