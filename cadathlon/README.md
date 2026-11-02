# CROSS CADathlon problem set

Four problems in two sets, each following the supplied CADathlon template:
introduction, problem statement, compile/run instructions, scoring, references,
`doc/`, `src/`, `test/`, `oneliner`, and `EXCLUDE`.

## `non_agent_pset` — no AI assistance

Both problems here are solved without coding agents. A team's total is
**essential + bonus**, so **120 points** are available.

| | Problem | Points | Judged by | Handout |
|---|---|---|---|---|
| **Essential** | [P1 · MNIST throughput](non_agent_pset/p1_benchmark/doc/problem.md) — raise encrypted MNIST throughput across eight TPU chips on the FHE benchmarking suite workload | **100** | rule-based rubric | [HTML](non_agent_pset/p1_benchmark/doc/problem.html) · [PDF](non_agent_pset/p1_benchmark/doc/problem.pdf) |
| **Bonus** | [P3 · LeNet & AlexNetTiny](non_agent_pset/p3_cnn/doc/problem.md) — accelerate both encrypted CNNs while preserving their fixed models | **20** | open-ended demonstration | [HTML](non_agent_pset/p3_cnn/doc/problem.html) · [PDF](non_agent_pset/p3_cnn/doc/problem.pdf) |

**P1 is the essential problem.** It is scored by the rule-based rubric in its
section 4, case by case, against measurements the benchmarking suite's own
harness writes.

**P3 is a bonus problem** — attempt it only after P1, for up to 20 extra
points. It is evaluated openly rather than by a per-case rubric: a team
demonstrates feasibility. The 20 points are **all or nothing**, and are awarded
only when a team shows **both** of the following:

1. **Correctness** — the CNNs run end to end under HE on the TPU, and the
   decrypted predictions match the same model in cleartext.
2. **Performance** — measured against the best open-source implementation the
   organizers name, the submission is **no more than 50% slower**, or faster.
   A run taking `T` against a state-of-the-art `T_sota` qualifies when
   `T <= 1.5 * T_sota`; equivalently, throughput must reach at least two
   thirds of the state of the art.

Showing only one of the two earns nothing. A correct but slow implementation
scores zero bonus points, as does a fast one that cannot be shown correct.

## `agent_pset` — AI agents allowed

| Problem | Points | Handout |
|---|---|---|
| [P2 · BERT / MRPC](agent_pset/p2_bert/doc/problem.md) — use or improve CROSS to enable encrypted BERT on TPUs, then optimize performance and score using the [FHE BERT benchmark](https://github.com/fhe-benchmarking/BERT) | 100 | [HTML](agent_pset/p2_bert/doc/problem.html) · [PDF](agent_pset/p2_bert/doc/problem.pdf) |
| [P4 · Qwen3-0.6B](agent_pset/p4_qwen/doc/problem.md) — enable full Qwen3-0.6B inference under CKKS, with staged credit for operators and decoder blocks | 100 | [HTML](agent_pset/p4_qwen/doc/problem.html) · [PDF](agent_pset/p4_qwen/doc/problem.pdf) |

The proposed format is an eight-hour contest. “AI agents allowed” means
students may use coding agents to solve the task; it does not mean that several
agents share an HE secret key. The two sets have separate leaderboards, and so
do the problems within `agent_pset`. Duration, hardware, and resource limits
below are proposed contest rules, not historical results.

The package contains editable Markdown, rendered handouts, TODO entrypoints,
six public input/output pairs per problem, numerical references, build/test/
submission targets, a CNN baseline, and [organizer instructions](ORGANIZER.md).
The HTML and PDF design follows the CROSS tutorial: Inter/Outfit typography,
purple/pink accents and a dark palette. HTML also supports a light theme,
responsive navigation and a section index. Fonts and page assets are embedded
for offline viewing. The shared styles live in `common/tutorial.css`, with
navigation behavior in `common/tutorial.js`; `make docs` rebuilds both formats.
The small public Qwen fixtures test decoder semantics; they do not replace the
full 28-layer, 0.6B target. Its `.golden` file is an explicitly labelled
**plaintext numerical oracle**, not a completed encrypted Qwen implementation.
CNN fixtures use synthetic weights for portability; ranked accuracy runs require
the frozen trained checkpoints and data described in the handout.
Problem 3's ranked cases run inside the FHE benchmarking suite's own harness at
a pinned commit, so correctness is decided against the suite's ground-truth
labels and timing comes from its `Timing` block; `make -C non_agent_pset/p1_benchmark/src
suite-check` refuses to proceed if `harness/` differs from that commit.
BERT includes a pinned upstream harness contract, seven stage adapters and a
strict output checker. Its six small numerical fixtures and plaintext golden
do not replace full-checkpoint MRPC evaluation. Students may use AI agents
throughout the BERT workload, with usage disclosed in the team report.

From this directory:

```bash
make all                 # syntax and package checks; does not solve the TODOs
make reference-test      # 24 public numerical-reference checks; CPU
make docs                # rebuild HTML/PDF (requirements-docs.txt)
make release             # create contestant archive, applying EXCLUDE
```

Each problem's `make -C pN_name/src test` runs its contestant entrypoint and
intentionally fails until the TODOs are implemented. The CNN
`make baseline-plan` target checks the actual CROSS packing path without key
generation. `make baseline-test` runs actual encrypted inference and can take
minutes and substantial memory. No public smoke check certifies security,
full-model correctness, or measured contest performance.

Encrypted smoke runners enforce the 128-bit classical Q*P ceiling on every
live `CKKSContext` exposed by the submission, then check that input and output
ciphertexts use those rings. P1 suite stage 3 applies the same gate after warm-up
and records the observed execution rings. Standard `ctx`, `_ctx`, `mapping`,
and `_mapping` wrappers work directly; custom service/session wrappers must
provide `security_contexts()` returning all live client and server contexts,
including contexts created lazily during evaluation. A declared profile alone
does not satisfy the gate. These checks catch accidental insecure parameters;
they do not make contestant Python trusted, and organizer evaluator audits
remain necessary.

The problem set is authored against CROSS commit
`e652a56f897d8ed814978359e5a2b0c78f46641d`. Source precedence is current checked-out
code/API reference, then tutorial guidance, then historical paper measurements.
See [provenance](SOURCES.md) and [validation](VALIDATION.md). This is a complete
problem proposal and starter package; organizer asset preparation and hardware
calibration are required before conducting a scored event.
