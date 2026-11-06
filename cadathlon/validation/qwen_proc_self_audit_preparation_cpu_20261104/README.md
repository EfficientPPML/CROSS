# Qwen own-process audit preparation — 56 focused checks

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
