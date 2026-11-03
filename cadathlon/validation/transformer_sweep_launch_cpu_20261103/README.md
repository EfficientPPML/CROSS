The validation-only launch wrapper passed **11 metadata/process tests in
1.95 seconds** on CPU152–155. Exact wrapper, test and collector hashes stayed
unchanged. Tests use fake ciphertext/device classes and real short subprocesses;
they execute no encrypted computation or TPU work. The initial two empty-folder
fixture failures and their exact sources are retained in `initial_v1/`.

The approved wrapper is `launch.py`, SHA256
`357a9f35b22bf8b78a84f10ef5dd0365ae51f12fbbb24ee934397896d36fe16a`.
The tested collector bytes match the promoted live collector. No runtime source
was modified by this task. Parent-agent read-only review accepted the wrapper;
the final requested environment guards are covered by a regression.

Preparation copies the selected core, demos, common helpers, P2/P4 sources and
public fixtures, corrected collector, and this instrumentation into a separate
snapshot. It compares original/copy hashes and seals files/directories against
accidental writes. The run executes the pinned copy, archives all selected exact
source bytes, and verifies them between cases and at completion. Installed
package versions, machine information and effective environment are recorded.
This is a runtime/public-file closure, not an arbitrary third-party import audit
or protection against an adversarial same-user process changing permissions.

The wrapper reuses the corrected collector's child shim, validation, process
group cleanup and owned-directory identity checks. Its only in-memory runner
instrumentation calls the original security gate, then records the returned
live Q/P manifests and ciphertext shape/scale/noise-degree/device metadata.
It does not serialize ciphertext payloads, plaintexts or keys. Actual payloads
must reside on TPU; this placement evidence does not prove where every internal
operation ran. Capture and publication overhead remain in end-to-end timings;
no time is subtracted and no calibrated comparison is claimed.

An external thread samples the observed owned process group and descendant
tree's RSS. Shared pages may be counted more than once and short peaks may be
missed. Device `memory_stats()` is captured at existing input/output security
boundaries; these checkpoints are not a continuous peak-HBM measurement.
The complete child process/resource records remain alongside raw stdout,
stderr and live-ring/device evidence for every attempted case.

Each attempt requires fresh evidence and public-key-parent paths. Unknown child
cleanup retains the parent. Launch failure with a completed no-child cleanup
record can remove it. Existing unrelated parents are never removed. Source
drift or incomplete cleanup stops further launches; ordinary numerical failures
remain recorded while the remaining cases are attempted. Each model remains
incomplete until all six cases pass with unchanged sources and successful
cleanup. No partial sweep can become calibration.

The default correctness schedule is Qwen1, BERT1, BERT2, Qwen2, BERT3, Qwen3,
Qwen4, BERT4, BERT5, Qwen5, Qwen6, BERT6. It starts with leveled graphs, then
known/smaller-period refresh, and leaves the unvalidated full-slot refresh last.
The current ideal plans have approximately63 Qwen refreshes versus26 BERT
refreshes across six cases; smaller key volume alone does not make Qwen-first
faster. `--order qwen-first` provides that alternate ordering without dropping
cases. Cold zero-warmup/one-repeat coverage precedes costly warm repetitions.
Full pretrained/file-stage implementations and scoring calibration remain
separate, unresolved requirements, as detailed in the
[contract review](../transformer_sweep_plan_20261103/PLAN.md).

These are prepared commands, to run only on the parent's assigned TPU lease.
They were **not executed** by this task:

```sh
/home/jianming_gatech/miniconda3/envs/jaxite/bin/python \
  /tmp/transformer-sweep-launch-20261103/launch.py prepare \
  --source-root /home/jianming_gatech/CROSS \
  --snapshot /tmp/cross-optimized-twelve-20261103/checkout

taskset -c 160-167 env PYTHONDONTWRITEBYTECODE=1 \
  /home/jianming_gatech/miniconda3/envs/jaxite/bin/python \
  /tmp/cross-optimized-twelve-20261103/checkout/validation_sweep.py run \
  --root /tmp/cross-optimized-twelve-20261103/checkout \
  --output /tmp/cross-optimized-twelve-20261103/evidence \
  --key-parent /dev/shm/cross-optimized-twelve-20261103 \
  --order interleaved --warmups 0 --repeats 1 \
  --cleanup-grace-s 30 --sample-interval-s 1
```

The run explicitly sets the snapshot CROSS/PYTHONPATH, TPU backend, one
OMP/BLAS thread, level-cache capacity1 and NTT-cache capacity2. It clears
inherited `CROSS_EAGER_LEVEL_PARAMS`, `TPU_VISIBLE_CHIPS`, `XLA_FLAGS` and
`CROSS_SKIP_TOPLEVEL_ROTKEYS`. All changes are scoped and restored. It checks
117610135552 bytes of free key storage before launching; every actual session
additionally checks its exact allocation. Tmpfs pages consume host memory.
No healthy-work deadline is imposed; 30 seconds is cancellation grace only.

Reports permanently label the workload as public toy fixtures and leave
`scoring_calibrated=false` and `full_model_evaluated=false`. A later
`--warmups 1 --repeats 3` run needs fresh output/key paths and remains toy timing
evidence. It cannot publish P2 full-BERT baseline values or implement P4's
distinct-prompt/full-model calibration schedule.
