# Preparation commands and future phase handoff

Preparation only (no `--execute`):

```
env -u TPU_VISIBLE_CHIPS -u XLA_FLAGS -u CROSS_EAGER_LEVEL_PARAMS PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_PLATFORMS=cpu JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 120-123 /tmp/cross-pretrained-validation-env/bin/python -I -B /tmp/bert-normalized-round-gain4-validator-v2-20261104/collect_validator.py --output /tmp/bert-normalized-round-gain4-validator-v2-preflight-v1-20261104 --source-manifest /tmp/gain4-validator-v2-sources.json --source-manifest-sha256 EXTERNALLY_PINNED_MANIFEST_SHA256 --key-evidence /tmp/bert-full-key-generation-component-20261103/actual-v1 --key-workspace /dev/shm/cross-full-bert-keys-143b3f5c1573427d841760a3c9d5d33d
```

The original163 fixture is already complete at `/tmp/bert-normalized-round-gain4-ready-20261104/actual-fixture-v1`, externally pinned by aca358a292ca8cc867241e09abec18889b20b31ccc4111b4fa66b2830ff271ad. Its server is still live and unpinned here. The existing IO is `/tmp/cross-normalized-gain4-1379133c462e6929737308e3195bf44d` with inode identity[66305,3370369]. The key store and IO are borrowed read-only and never retired by this validator.

A future independently reviewed successor must bind the persisted final old163 server report and root review, include the root receipt in its explicit consumer closure, and enable only validator_authorization.json. The legacy execution_authorization.json must remain false. Only then may root grant the validator phase with `--stage validate --execute`, the exact fixture/server evidence paths and externally pinned hashes, CPU124–127/four threads, and a newly owned output directory outside all retained/evidence/source trees. This file is not a launch grant.

Validation reads both existing inputs and all54 existing outputs before private key access, then decodes in order and applies unchanged gates. It writes no ciphertexts and retains all old IO. The inherited conservative storage check reserves3,244,294,144 bytes; this is an IO margin rather than measured memory or execution time. All raw decoded arrays remain process-private. There is no healthy deadline. The owner of any future actual process must retain its authoritative session handle, wait/reap it, verify exact source/payload/terminal evidence, and report numerical failures without retries or gate changes.

The public preflight source manifest is generated from the explicit source inventory before freezing and kept outside the source tree. Its SHA must be passed explicitly to both collector and worker; the parent passes the exact copied source_manifest.json digest to its worker. The placeholder above is replaced only with that independently recorded hash. This external digest is an invocation trust input, never a self-authentication claim.
