# Genuine full-profile BERT key generation

The actual client key-generation handler completed successfully and was reaped with exit zero. It produced the complete public-key, relinearization-key and 130-rotation-key inventory: **263 public NPY arrays, 125,380,363,136 bytes including headers**. The retained workspace is available to later stages; its payloads are not included here.

The fixed profile is N131072, CD2, depth49, dnum8, Q100/P14. The actual QP size is 3402.080255733439 bits; the 128-bit classical-security and P-coverage checks passed before key generation. Worker time was 3272.128 seconds and collector time 3273.428 seconds on the recorded CPU affinity. This is one key-generation component measurement, not an inference or throughput benchmark.

All **58 source files** matched startup, terminal, immutable execution copy and exact archive. Archive-time readback also verified current source bytes, worker/report/public-manifest bindings, the public context digest, the reported terminal cleanup and absence of the worker/collector PIDs. The parent agent authoritatively reaped session63746; this archiver independently checked the terminal evidence and PID absence.

Context ID: `ba28d6a81704eba60c51ad59fbfee3901516b783a1144d973e8199a014d87fb9`.

The retained `/dev/shm` directory identity was checked using metadata only. **No key array or private manifest was read for archival.** Generation-time evidence records a valid private binding, restrictive permissions and two private files. Later legitimate request/receipt files may increase that count; neither this archive nor its verifier requires the private directory to remain at two files.

`measured/` contains public manifests/descriptors, raw reports/logs and exact sources. `verify_archive.py` checks those public artifacts without importing cryptographic code or opening the retained workspace. `source/archive_and_current_verify.py.txt` records the original archive-time source/metadata checks. README, summary and verifiers are archive-time additions.

No secret-key payload, private manifest, secret descriptor, public-key payload, private input, ciphertext or model tensor is included. This successful first handler does not establish input encryption, bootstrap, full-model inference, MRPC accuracy or completion of the other four file stages.
