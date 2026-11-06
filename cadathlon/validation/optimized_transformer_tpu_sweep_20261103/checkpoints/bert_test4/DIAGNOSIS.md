# Optimized toy BERT test4 failure

This eighth completed case is the live twelve-case sweep's first failure. The
seven earlier case passes remain preserved. The parent sweep continues under
root session12657; this archive neither reaps it nor launches a retry.

The child exited1 after `ApproximationErrorTooHigh`: estimated decode standard
deviation `inf`, unchanged limit0.03125. Its stdout contains no decoded logits,
maximum logit error, nRMSE, top1 agreement or successful benchmark timing fields.
The external process duration was13866.616927938012 seconds. Do not describe
`inf` as an observed logit error; it is the decoder's rejected error estimate.

## What the saved evidence establishes

The case is synthetic BERT: batch2, sequence8, hidden24, intermediate48, four
heads, two layers, two labels, seeded public weights9504. Valid-key counts are6
and5. It is not full-width pretrained BERT, MRPC accuracy, or ranked performance.
The public packed layout derives feature width64 and period8192, repeated eight
times across65536 slots; this is not a proof over independent full-slot inputs.

The actual input/output records show degree131072, Q100/P24, chain depth49,
dnum4 on the evaluator, log2(QP)=3490.161581036245 under the recorded128-bit
classical security limit. Both ciphertext boundaries reside on TPU v6 lite.
Output has24 Q limbs (CD2 level11), scale6.8994011570896904e16 and NSD1. Thus the
record does not show a zero-level or nonfinite-scale metadata failure. Remaining
levels and secure parameters do not certify numerical correctness.

The stderr progress log reaches both layer outputs and contains five matching
refresh-start/refresh-complete pairs. It then reports overflow in the squared
terms of `jaxite_word/ckks_ctx.py` lines1658,1659,1661,1663. In that source,
`_decode_approximation_stddev` forms a conjugate-symmetry error estimate;
`_check_decode_stddev` rejects nonfinite or over-limit estimates. `_fast_decode`
performs this check before averaging conjugates and the slot FFT. The original
`transformer_runner.benchmark_session` calls client decrypt before `metrics`
and BERT `check_output`, so neither output gate was reached.

The security probe invokes the original ring/security gate before recording
each boundary. Both boundary events exist. Its overall report is nevertheless
`passed:false`, with `SystemExit:1` from the failed runner. This archive preserves
that false flag; successful metadata capture is not an overall case pass.

## Limits of the diagnosis

No intermediate decrypted payload, measured polynomial input range, per-refresh
numerical residual, or final coefficient payload was saved. Consequently the
first corrupt operation cannot be localized from these records. The source's
public polynomial/refresh bounds are explicitly conditional ideal-value
contracts; they do not bound accumulated CKKS or refresh error. The BERT toy
contracts are inverse-sqrt degree2 on[0.5,1.5], exp degree8 on[-4,4], reciprocal
degree8 on[0.5,8], GELU degree8 on[-3,3], and tanh degree8 on[-2,2]. No new claim
that these conditions held for the failed ciphertext trajectory is made.

The records do not distinguish violated input domains, refresh error growth,
scale/arithmetic behavior or another numerical defect. They do show a genuine
client decode failure after encrypted computation. No threshold, ring profile,
coefficient, source or case fixture was changed. No numerical rerun occurred.

A future separately authorized diagnostic would need source-frozen intermediate
ciphertext checkpoints, all server work before client decode, independent fixed
polynomial references and unchanged output gates to localize the first error.
That is a proposed follow-up only; it is not part of this failed run.

## Preservation and resource scope

All199 frozen current source files match the original tar member bytes. All six
case files were copied unchanged and checked before/after. Child2366294 is
recorded reaped with exit1; PID/group and its owned public-key directory are
absent. The owned-directory handle is recorded closed. Parent944985 was present
at checkpoint capture, with eight completed cases: seven passes and this failure.

There are13251 resource samples at nominal1-second intervals. Sampled process
tree RSS peaks at135146196992 bytes; shared pages may be counted twice and peaks
may be missed. Output-device checkpoint peak_bytes_in_use is29276874240 bytes,
not a continuous independent resource guarantee. No complete-sweep or performance
pass is claimed.

Run `python -B verify.py` from this checkpoint. It uses the enclosing archived
199-source manifest/tar and only public evidence files; no live runtime import,
payload read, key generation, TPU or model execution occurs.
