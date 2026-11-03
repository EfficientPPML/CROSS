This directory preserves completed-case checkpoints from the ongoing sealed
twelve-case optimized TPU sweep. It is **not a completed sweep or calibration**.
The independently archived checkpoints are Qwen `test1.in` and BERT
`test1.in`/`test2.in`. Three of twelve cases have completed; neither model has
completed its six cases.

Qwen1 passed the unchanged toy gates with max absolute error
**0.030491700345**, nRMSE **0.01361692956** and top1 agreement **1.0**.
One cold encrypted request was measured: setup253.23s, server1158.80s,
end-to-end1166.89s, runner total1431.57s and external case1438.63s. There were
zero warmups. Instrumentation overhead remains included; these numbers are
not a controlled speedup or full-model timing claim.

The input and output security-gate records contain both actual N131072/Q78/P37
contexts and TPU payload placement. Actual log2(QP) is3283.2516965, within the
archived classical128-bit ceiling. All **199 exact archived/current source
files** matched the sealed manifest at the checkpoint. The child exit0 was
observed by the collector, its PID is absent, and its owned key directory and
directory handle were cleaned up successfully.

The sampled aggregate process-tree RSS maximum was **22869868544 bytes**.
This can double-count shared pages and miss short peaks; it is not physical
unique memory or a continuous peak. Device-memory statistics are gate
checkpoints, not proof of continuous peak HBM or every internal execution
device. The raw resource samples and instrumented metadata are preserved.

`checkpoints/qwen_test1/` contains unchanged raw records/logs and an independent
summary. Its `parent_report_at_checkpoint.json` records the still-incomplete
overall sweep while BERT1 was active. The top-level source archive, snapshot
manifest and root preflight preserve exact code identity. `MONITORING.md`
records scheduling, ownership and cleanup responsibilities.

`checkpoints/bert_test1/` independently verifies the second completed case:
max absolute error **0.004302472197**, nRMSE **0.011767177989**, top1 **1.0**,
with zero warmups and one fresh encrypted request. Setup was592.84s,
server1704.19s, end-to-end1714.34s, runner total2322.02s and external2329.45s.
Both actual N131072/Q102/P20 rings have log2(QP)3424.7298497 and TPU payload
placement. The same199 exact archived/current sources match; collector exit0,
child absence and owned directory/handle cleanup are verified. Its sampled
aggregate RSS maximum40368857088bytes has the same limitations described
above. The parent remained live in BERT2 when this checkpoint was captured.

`checkpoints/bert_test2/` verifies the third completed case: max absolute error
**0.000857503427**, nRMSE **0.003188205659**, top1 **1.0**, with zero warmups
and one fresh encrypted request. Setup was755.60s, server1746.76s,
end-to-end1757.08s, runner total2528.87s and external2536.06s. Both actual
N131072/Q102/P20 ring records and input/output TPU placement are present;
all199 source hashes, child exit0/absence and key/handle cleanup pass. The
sampled aggregate RSS maximum46827896832bytes retains the same caveats.
The parent was live in Qwen2 when this checkpoint was captured.

Root owns exec session12657 and its eventual reaping/exit code; the monitoring
agent can observe PIDs but cannot access that parent-owned tool session. No
parent exit code is inferred from a child result or process absence. The
remaining nine cases, final source/cleanup checks and parent terminal state
are not established by these checkpoints. Full pretrained/file-stage behavior
and scoring calibration remain separate unresolved requirements.
