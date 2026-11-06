from pathlib import Path
import hashlib,json,shutil
L=Path('/tmp/bert-initial-softmax-corrected-component-20261103');R=L/'run-v1';T=Path('/tmp/bert-initial-softmax-corrected-terminal-20261104');A=Path('/home/jianming_gatech/CROSS/cadathlon/validation/bert_initial_softmax_corrected_component_cpu_20261104');A.mkdir(exist_ok=False);(A/'run-v1').mkdir()
for n in ('report.json','sources.tar.gz'):shutil.copyfile(R/n,A/'run-v1'/n)
for n in ('run-v1.log','run-v1.time.txt'):shutil.copyfile(L/n,A/n)
for n in ('owner-terminal.json','sources.json'):shutil.copyfile(T/n,A/n)
r=json.loads((R/'report.json').read_bytes());summary={'scope':'Recorded metric readback only; no repeated HE or private payload read.','checks':{n:{k:v[k]for k in ('passed','shape_and_finite','error_arithmetic_finite','max_abs_error','slots_checked','level','scale','expected_scale','nsd','metadata_ok')}for n,v in r['checks'].items()},'final_diagnostics':r['final_diagnostics'],'elapsed_s':r['elapsed_s'],'peak_host_rss_kib':r['peak_host_rss_kib']};(A/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
(A/'README.md').write_text('''# Genuine corrected initial-softmax component

The frozen corrected component passed all14 checkpoints over917504 checked slots
at the unchanged1e-6 absolute gate. Worst checkpoint error is
2.8573543531251744e-10. Final probability errors are2.23472802762803e-12 and
5.370348610256315e-11. Every checkpoint finite/metadata check passed. Final mass
and padding observations are retained as diagnostics, not extra acceptance gates.

Two fixed public synthetic full-slot score fixtures were encrypted and evaluated
on CPU with N131072/CD2/depth49/p60/d8, Q100/P14 and inputL38. The actual QP
bit-length is3403 and the source-bound security/P-coverage checks passed. Expected
checkpoint scales follow the real binary64 recurrence, including noncanonical
intermediates; ciphertext scales were not relabeled. All14 server ciphertexts
were completed before any client decode. Seven codec APIs and server keygen
entrypoints were guarded. Client and server are separate objects in one dedicated
process, not a three-process file boundary or an OS sandbox.

This covers mean/centering, exp and initial reciprocal/probability stages only.
The five normalized-squaring rounds, refresh, complete attention, a block and a
model were not evaluated. No transfer of refresh precision or full-model
correctness follows. The original historical metadata-assertion failure and
separate normalized-round numerical failure remain unchanged.

Session44272 exited0 and was authoritatively reaped (tool e5927b). Worker1222982
and observed parent/process-group1222978 are absent; the owned key directory is
removed and both cleanup error lists are empty. Elapsed86974.363554368s and peak
80360904KiB RSS are uncalibrated component observations, not benchmark claims.
The exact113 current/start/end/archive source hashes agree. This public capsule
contains source/metadata and reports only, no key, ciphertext or array payloads.

`python -B verify.py` performs portable public readback; `--current` additionally
checks the preserved source tree and current process/key-directory absence.
No computation or decryption is repeated. Independent root/security reviews are
separate from the original actual run and will retain their stated scopes.
''')
shutil.copyfile('/tmp/archive_initial_softmax_corrected_20261104.py',A/'archive_builder.py')
print(A)
