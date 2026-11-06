# Proposed phase commands — require independent launch authorization

Run from `/tmp/bert-normalized-round-gain4-ready-20261104`. Use the exact frozen163-source successor. The earlier DESIGN command retained its disabled-lab output example; the commands below use distinct ready-lab outputs. No phase has been launched.

```sh
taskset -c 124-127 env JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /tmp/cross-pretrained-validation-env/bin/python collect_component.py --execute --stage fixture --output /tmp/bert-normalized-round-gain4-ready-20261104/actual-fixture-v1 --key-evidence /tmp/bert-full-key-generation-component-20261103/actual-v1 --key-workspace /dev/shm/cross-full-bert-keys-143b3f5c1573427d841760a3c9d5d33d
```

Only after the fixture report/source/archive/cleanup/owned-group/readback gates pass and its SHA is independently pinned, invoke the same prefix and key arguments with:

```text
--execute --stage server
--output /tmp/bert-normalized-round-gain4-ready-20261104/actual-server-v1
--fixture-evidence /tmp/bert-normalized-round-gain4-ready-20261104/actual-fixture-v1
--fixture-report-sha256 <independently pinned completed fixture report SHA256>
```

Only after all54 server outputs and terminal evidence pass, invoke the same prefix and key arguments with:

```text
--execute --stage validate
--output /tmp/bert-normalized-round-gain4-ready-20261104/actual-validate-v1
--fixture-evidence /tmp/bert-normalized-round-gain4-ready-20261104/actual-fixture-v1
--fixture-report-sha256 <independently pinned completed fixture report SHA256>
--server-evidence /tmp/bert-normalized-round-gain4-ready-20261104/actual-server-v1
--server-report-sha256 <independently pinned completed server report SHA256>
```

The parent supplies and authenticates these hashes; no command derives its own launch authority. Each phase runs in a fresh process, and no healthy deadline is imposed. The raw two-input+54-output IO is2065694720bytes. Each collector requires3244294144bytes free reserve (rawIO+100MiB temporary+1GiB margin); retained authenticated130-key storage is borrowed in place. This is a disk reserve, not a RAM-peak estimate. Exact unknown runtime/NTT/device/compiler transient memory remains unmeasured. No refresh or new key generation occurs.
