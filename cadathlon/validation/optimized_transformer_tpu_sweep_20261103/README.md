This archive preserves **all twelve completed cases** from the sealed
optimized TPU toy sweep: seven passes and five failures. BERT test4 failed during
client decoding after five refreshes; BERT test5 reports an `OverflowError` after
eight refreshes; Qwen tests5 and6 failed strict approximation validation after19
refreshes. BERT test6 failed strict approximation validation after eight
refreshes. None of these failures produced decoded-logit metrics.
The original parent exited1 on November5. All six Qwen cases completed with
four passes and two failures; BERT completed with three passes and three failures.

| Case | Max absolute error | nRMSE | Top1 agreement | Cold runner total (s) | External case (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| qwen test1 | 0.0304917003449 | 0.0136169295599 | 1.0 | 1431.57 | 1438.63 |
| bert test1 | 0.00430247219694 | 0.0117671779891 | 1.0 | 2322.02 | 2329.45 |
| bert test2 | 0.000857503427194 | 0.00318820565898 | 1.0 | 2528.87 | 2536.06 |
| qwen test2 | 0.0266696542003 | 0.0185270428426 | 1.0 | 6870.74 | 6877.97 |
| bert test3 | 0.000773446206586 | 0.00445166578068 | 1.0 | 11106.24 | 11113.48 |
| qwen test3 | 0.0473918855789 | 0.0322078518748 | 1.0 | 17698.76 | 17705.95 |
| qwen test4 | 0.0258476940157 | 0.0160142303793 | 1.0 | 21146.86 | 21154.16 |
| bert test4 (failed decode) | unavailable | unavailable | unavailable | unavailable | 13866.62 |
| bert test5 (OverflowError) | unavailable | unavailable | unavailable | unavailable | 19834.48 |
| qwen test5 (failed decode) | unavailable | unavailable | unavailable | unavailable | 30241.93 |
| qwen test6 (failed decode) | unavailable | unavailable | unavailable | unavailable | 31490.24 |
| bert test6 (failed decode) | unavailable | unavailable | unavailable | unavailable | 41889.49 |

The [BERT4 failure diagnosis](checkpoints/bert_test4/DIAGNOSIS.md) preserves the
decoder's rejected `stddev=inf` estimate, original failed flags, unchanged
sources and process cleanup. The estimate is not an observed logit error.
The [BERT5 failure evidence](checkpoints/bert_test5/README.md) preserves the
recorded `int too large to convert to float` exception and eight refresh pairs;
the saved stderr does not localize its source line.
The [Qwen5 failure evidence](checkpoints/qwen_test5/README.md) preserves its
`stddev=4.099e48` approximation rejection against the unchanged0.03125 limit.
Its19 completed refreshes and secure output metadata do not identify the first
corrupting operation or establish numerical correctness.
The [Qwen6 failure evidence](checkpoints/qwen_test6/README.md) preserves the
separate `stddev=6.314e81` rejection against the same0.03125 limit. Its final
12-limb Q prefix fits within binary64's finite range; that fact does not locate
the numerical failure or support weakening the strict decoder.
The [BERT6 failure evidence](checkpoints/bert_test6/README.md) preserves the
`stddev=1.036e26` rejection against the unchanged0.03125 limit after eight
refreshes and six layer milestones. The progress logs do not locate the first
encrypted numerical divergence.

Each case used zero warmups and one fresh encrypted request. Instrumentation
overhead remains included. These are toy correctness and cold-run measurements;
they establish neither full pretrained execution, calibrated scoring, repeated
benchmark timing nor a controlled speedup. Qwen test4 took about5.87hours. Its
nRMSE0.01601423 passes the existing toy gate but exceeds the separately required
full-Qwen HE nRMSE ceiling0.01. Qwen tests1–4 all exceed that stricter error gate
and the contest timing target.

All **199 exact archived, start, end and current snapshot files** matched at
terminal admission. Each completed child has collector-observed reaping, absent
PID and successful owned-key-directory and handle cleanup. The seven passing
children exited0; BERT4, BERT5, BERT6, Qwen5 and Qwen6 exited1. Input/output security
records contain actual Q/P rings and TPU payload placement. Qwen test4 uses
N131072/Q100/P24, with actual log2(QP)3490.1615810 and recorded classical128-bit
security. Full raw ring inventories remain in each security_device.json.

Qwen test4 sampled aggregate process-tree RSS peaked at91,975,016,448bytes.
RSS sums can count shared pages twice and miss short peaks. Device-memory
statistics are input/output checkpoints, not continuous peak HBM or proof of
every internal operation. These limitations apply to every case.

Each checkpoints/MODEL_testN directory preserves six unchanged raw evidence
files, the parent report captured at that checkpoint, an independent
summary and hashes. The source archive, snapshot manifest and root preflight
preserve code identity. MONITORING.md records the sequential schedule and
ownership; its earlier phase observations are historical.

The [final report](final/report.json) records200,478.90s for the complete sweep.
The [root terminal receipt](final/root-terminal-admission.json) records the
original session12657 exit1, all twelve original child processes and process
groups absent, both original parent processes absent, all owned case/parent
key paths removed, and no cleanup errors. The seven passing cases do not make
the sweep a passing baseline. Full pretrained/file-stage behavior, ranked HE
accuracy and scoring calibration remain separate unresolved requirements.
