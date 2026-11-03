# Qwen held-out wrapper preflight

The prepared validation wrapper requires the successful frozen 416-context calibration before opening the original public validation input manifest or model assets. It checks the pinned report, all 31 runtime sources, exact unique tar members, all nine candidate inputs, complete token histories, numerical/domain gates and published aggregate gates. The selected candidate and arithmetic remain byte-identical to calibration.

The source-bound CPU preflight passed **51 tests and 19 subtests** (pytest21.06s, wrapper22.34s), authoritative session27553 exit0/reaped. All36 archived source files were unchanged. Tests exercised altered prerequisite reports/hashes, duplicate archive names/counts, changed runtime, prerequisite ordering, stopped trajectories and existing strict circuit/capture contracts. Tiny native test models were instantiated; no held-out input file, pinned checkpoint weights, JAX, HE or TPU was evaluated.

The proposed workload is the original48 prefills (16 each atS8/S16/S32),32 teacher-forced contexts and32 free-running contexts. Every valid approximate prefix is compared with native operators on those same actual tokens. Uncomputable free suffixes remain failed/unmeasured. Fixed operator/block nRMSE0.05 and full-logit nRMSE0.10 apply to every context, with at least44/48 aggregate prefill matches,29/32 teacher matches and6/8 whole free-sequence matches. Per-prefill-suite results remain separate.

This archive is preparation evidence; validation was not launched. The historical initial15-test log is retained separately and does not have a source-bound or recovered tool-exit claim. The full calibration evidence is in `../qwen_approximate_calibration_cpu_20261103`.

Run `python verify_archive.py` to check archived bytes and closure without opening validation inputs or loading a model. Review and explicit root scheduling precede the proposed native-only CPU140–143 execution.
