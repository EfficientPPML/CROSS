# Fixed-Q public control construction

The unchanged real HERot/BConv public control constructors passed all 50 CD2 prefixes for d8/aux30, d6/aux30 and d6/aux31: 150 constructions and 11,092,910 independently recomputed constant entries. All 37 installed forbidden-entry probes denied; no forbidden call occurred during construction. CPU backend initialization is explicit. Polynomial allocation was replaced only by immutable metadata descriptors. This is public control evidence, not CKKS/compiled execution, key generation, precision, latency, or memory-performance evidence.

All three profiles remain rejected by the unchanged registered validator. Its explicit logical OpenFHE grouping estimates P counts 15/19/18; physical bit lower bounds are 13/17/17; generated exact-coverage counts are 14/18/17. Actual QP security and physical partition coverage pass. Adding P alone cannot satisfy all exact-policy checks. A future physical-partition profile would require a separate coherent contract and new proofs; this evidence changes no profile or validator.

Success: session 21330, exit 0, authoritatively reaped; 49 current/start/end/archive sources match, 150 prefixes in 10.6077 seconds. The 16 tests were run afterward against the same exact source inventory and checked unchanged again. Test timing is separate from the public control run. The report and source archive are authoritative for exact values.

The earlier session 6589 exited 1 and was reaped. It constructed initial public controls then failed in audit-only recursive readiness traversal due to a misspelled Python attribute. Its exact49 archive, report, log and terminal evidence are retained under historical-failure. A fresh successor fixes that typo and adds an actual-function-AST regression; no measured failure bytes were edited.

Run `python verify.py` for portable source/report/XML verification, or add `--current /tmp/bert-fixed-q-registration-diagnosis-v2-20261104`. No keys, private data or ciphertext bodies are archived.
