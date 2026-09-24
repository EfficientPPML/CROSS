# Upstream reference measurements (context, not the scoring baseline)

These nine files are the **reference implementation's own** measurements,
checked into `fhe-benchmarking/ml-inference` at the pinned commit
`739378698fc1f668aff7246724b45cff76f04a98` under `measurements/`. The upstream
`measurements/README.md` states they were generated in February 2026 for the
single, small and medium instances. They are copied here verbatim for two
reasons.

First, as context. Median `Encrypted computation` over the three runs:

| instance | images | compute (s) | ms/image | accuracy |
|---|---|---|---|---|
| single | 1 | 7.6145 | 7614.50 | not reported for this instance |
| small | 100 | 647.0703 | 6470.70 | 0.9600 |
| medium | 1000 | 6505.6093 | 6505.61 | 0.9740 |

Second, as a hazard. `run_submission.py` writes into the same
`measurements/<instance>/` directory, so a fresh checkout already contains
results files that look exactly like a submission's. A scorer that reads that
directory without checking will report the reference implementation's numbers
as somebody's submission. `score.py` reads the per-case archives in the parent
directory first and labels any fallback read in its `read_from` field.

**These are not a like-for-like hardware comparison.** The upstream numbers
were produced on the reference implementation's own machine, which is not
described in the repository and is not the TPU host the CROSS baseline runs on.
Treat the ratio between them as a rough order-of-magnitude figure, not a
measured speedup. The ranked score in this problem is computed against the
CROSS baseline in `baseline_manifest.json`, measured on the contest hardware,
never against this table.
