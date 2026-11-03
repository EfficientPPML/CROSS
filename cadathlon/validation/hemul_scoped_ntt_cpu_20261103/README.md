# Optional HEMul NTT reuse: isolated CPU candidate

This candidate extends the explicit, owner-scoped NTT provider from rotation controls to multiplication controls. It is **unapplied**. It does not establish a TPU speedup, complete refresh, or encrypted transformer pass.

Apply `hemul_scoped_ntt_followup.patch` **after** the unchanged [eleven-file combined candidate](../combined_runtime_candidate_cpu_20261103/README.md). The incremental patch changes four runtime files and adds one test file. Its base/candidate file hashes are in `patch_sources.json`; the combined patch SHA256 is in `summary.json`. `git apply --check` passed against that combined candidate. The original combined patch remains byte-identical to its archived copy.

`_HEMulKernel` receives an optional Polynomial factory. Both ordinary multiplication and the no-rescale tensor/relinearization accessor pass the context-owned callback into it. All six control-construction sites use that callback. Explicitly supplied `ntt_ctx` objects retain their existing behavior, including the finite-field-only tensor output wrapper. Existing scoped-provider guards are unchanged. Default callbacks remain `None`, and the session's `ntt_subsets` option remains `False`. Opting in installs the same provider for future HERot and HEMul controls. Closing it detaches only callbacks still owned by that provider. Previously constructed caller-owned controls remain valid.

Nested rescale construction is unchanged. `CompiledTensorOnly` continues to avoid constructing HEMul/key-switch controls entirely. No new shared module or release dependency is introduced; `encrypted_ntt.py` is already in the combined candidate's package allowlist.

The final CPU regression process passed **123 tests and 6 subtests**, with one optional second-device skip, in **159.86 seconds** including the wrapper. Its 137 captured Python/configuration sources were unchanged, and every archived source byte was checked against the recorded startup hash. A separate two-logical-CPU-device process passed the skipped device guard. A fixture-focused follow-up passed **24 tests** with 187 sources/fixtures captured, including all twelve inputs, expected outputs and reference modules. The twelve packed graphs and twelve bounded graphs are conditional NumPy/ideal-refresh checks, not encrypted passes. Their symbolic queue uses one pending item because it has no physical ciphertext payload.

The new secure component test uses N=8192, Q6/P2, dnum=3, batch=2, levels 2 and 1, and actual log2(QP)=184.6768058. It passes the library's 128-bit security gate and uses a public-key-only evaluation context. All six public NTT fields match ordinary construction exactly, including discontiguous key-switch subsets. Real ciphertext payloads and scale/noise metadata match for core multiplication, split tensor/relinearization, `CompiledTensorOnly`, `CompiledRelinearization`, `CompiledTensorProduct`, and the actual `Bootstrap._he_op_mul` dispatch. The Bootstrap checks include genuinely formed nsd=2 inputs. Maximum decrypted error in the new final component checks was **5.0420e-8**, below the fixed 2e-6 gate.

The tests also cover CD1/CD2 construction, both rescale modes, unchanged Montgomery/64-bit/alternate-layout fallbacks, occupied callback rejection, reset/close ownership, cache clear and rebuild, executable reuse, and collection of owned contexts and control arrays while machine code remains retained. The broader suite includes existing HEMul, double-rescale, CKKS level, NTT, primitive, tensor, compiler and lifecycle tests.

Evidence folders preserve raw output, JUnit, aggregate numerical records, exact commands/environment/versions, source hashes and `tested_sources.tar.gz`:

- `regressions/`: final 123-test checkpoint.
- `second_device/`: explicit two-device guard.
- `graph_fixture_followup/`: twelve packed plus twelve bounded fixture contracts with complete fixture capture.
- `initial_proof/`: earlier 32-test component checkpoint before the additional fallback tests.
- `mistyped_test_path/`: preserved command-selection failure (pytest exit 4, zero tests); corrected in the final checkpoint.

Each source archive contains the exact collector/plugin version used for that run. Extract an archive into a fresh directory and use its recorded command/environment, adjusting only the extraction root. `sources/` also exposes the five final changed files as exact `.py.txt` snapshots for review. No generated keys, ciphertext arrays or private input bytes are retained. Existing upstream test source includes its fixed public arithmetic test vectors.
