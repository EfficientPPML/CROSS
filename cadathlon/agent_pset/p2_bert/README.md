# Problem 4: Enable and optimize BERT on TPUs with CROSS

Read the [HTML assignment](doc/problem.html) or [PDF](doc/problem.pdf).

**Use or improve CROSS to enable encrypted BERT inference on TPUs, then
optimize it for better benchmark results and a higher assignment score.**

1. **Use the [CROSS library](https://github.com/EfficientPPML/CROSS) and improve it as needed**
   to implement encrypted BERT and execute its server computation on TPUs.
   Submit changes to the library in `cross.patch`.
2. **Evaluate your results with [fhe-benchmarking/BERT](https://github.com/fhe-benchmarking/BERT).**
   Run your CROSS implementation through its pinned evaluation harness.
   First pass the correctness and accuracy gates; then reduce latency and
   increase throughput. The handout's scoring rubric converts these validated
   benchmark measurements into assignment points.

**Students are explicitly allowed to use AI agents** for this workload,
including implementation, debugging, testing, profiling and optimization.
Disclose the tools used and verify their work in `REPORT.md`.

The target is `google-bert/bert-base-cased-finetuned-mrpc`, all 12 encoder layers,
the CLS pooler and two-class MRPC head, with sequence length 128.

Implement `src/bert_he.py` using CROSS and extend the library where needed.
The package includes seven upstream-compatible
stage entrypoints, working tokenization and output adapters, a strict result
checker, six synthetic reference fixtures, and a staging tool that preserves
the pinned upstream harness. The upstream DESILO submission is background
material; it is not this track's implementation or performance baseline.

```bash
cd src
make all
make reference-test           # six small plaintext checks, no downloads
make test                     # intentionally fails until bert_he.py is implemented
make submit                   # creates a local submission.tar.gz
```

For actual benchmark staging and commands, follow section 3 of the handout.
`make benchmark-check BERT_ROOT=/path/to/BERT` verifies the pinned checkout;
`make benchmark-stage BERT_ROOT=/path/to/BERT BERT_RUN=/new/run/directory`
copies its unchanged harness and your current source into a new run directory.
Restage after changing your source. Set `CROSS_ROOT` to your patched checkout.

The `.golden` is explicitly a **plaintext numerical oracle**. This starter
does not include a completed encrypted BERT solution or claim MRPC accuracy
or TPU performance. Organizers must freeze assets and calibrate the proposed
100-point, eight-hour rubric before an event.
