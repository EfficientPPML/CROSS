# Ordinary grouped-rotation representation parity

The completed CPU component run matched all **66 pairs**, including 58 nonzero pairs and eight zero-rotation aliases, at levels 49 and 25. All **1,314,914,304 compared ciphertext words** and their metadata matched exactly. The run made 116 genuine rotation calls and six shared decompositions; the genuine 49→25 reduction preserved the ordered Q prefix and scale.

This establishes same-input encrypted representation parity between the selected ordinary and grouped rotation paths. Both paths share implementation dependencies. It does not establish decoded numerical accuracy, an independent mathematical reference, full-model correctness, calibrated performance, or a speedup. No ciphertext, private key, model tensor, or input body is included here.

The fixed ring is N=131072, composite degree 2, Q100/P14, d=8. The public integer check gives 3403 QP bits against the 3523 bound, and the P product covers every alpha=13 Q partition. The actual run's source and report include its runtime security checks; the portable verifier performs only the public integer and evidence checks.

Owner session **42804** ended with tool chunk **711c01**, exit 0, and was reaped. The owner receipt records that shell 1878080, collector 1878082, worker/group 1878109 and the applicable process groups were absent. The temporary process-owned directory was removed. Borrowed key/model/I/O stores were retained. The collector and worker elapsed times were 71828.319163451 s and 71823.369352592 s on CPUs 168–171; these are component wall-clock observations, not a calibrated full-evaluation timing result.

The generic nested process record deliberately remains byte-exact with `passed: false`: its inherited helper initializes that field and never updates it. The selected collector instead authenticates typed return code 0, process cleanup, `result.passed: true`, the worker-report digest, and every parity/cleanup field. The owner readback preserves and explains this reporting detail.

All **259** original, execution, start/end report, and archived source entries matched. Module-origin records also matched the frozen map. The raw reports, process record, public progress log, exact source archive/map, and executed owner readback are retained. The owner readback script is inert `.txt` evidence; the portable verifier does not execute it or follow its external paths. No new experiment was run to package this result.

Run `python -B verify.py` for portable public verification. Optional `--current /tmp/bert-ordinary-group-parity-ready-20261103` compares only the source manifest's 259 entries. Offline verification authenticates the recorded process observation; it does not assert that historical PIDs can never be reused.

[Ready preparation](../bert_ordinary_group_parity_ready_cpu_20261103/) remains a separate historical capsule. `public_references.json` records prerequisite hashes without copying or opening borrowed stores. Independent root review passed (`b1380a`); its exact receipt and executed script are included. It also checked the selected collector gate directly and rechecked borrowed directory identities. The capsule has 15 explicitly indexed public files. No production promotion, retry, or store retirement is authorized by this capsule.
