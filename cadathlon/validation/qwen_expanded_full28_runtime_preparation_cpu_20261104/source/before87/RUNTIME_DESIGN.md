# Full28 runtime successor, pending wholegraph numerical review

The new `encrypted_qwen_full_model.py` borrows the accepted same-context
synchronous arithmetic, chained linear and fullperiod bounded refresher. The
68-source public predecessor remains immutable. No new nonlinear arithmetic is
implemented here: actual `QwenAttentionJoin` and `QwenDecoderTail` objects execute
each layer, their outputs are used directly as the next layer's inputs, and each
tail receives that layer's exact input tuple as its residual source.

`QwenParameterSource(contract, matrix_loader=..., gamma_loader=...)` accepts only
public parameter callbacks. Each requested array is checked against the pinned
QKV/decoder manifest or exact RMS gamma digest and copied into immutable bytes.
It never accepts intermediate activations. Callback identities, contract fields
and loader accessors are construction-bound. Trusted file authentication and
lifecycle belong to the eventual caller; this is not a Python sandbox. Layers
are constructed serially, so retired layer matrices can be released. The tied
vocabulary is authenticated once and held in immutable storage while 38 slices
are processed; its authentication temporarily materializes its full FP64 bytes.

`QwenFullModel(arithmetic, linear, contract, parameters, refresher=...,
progress=...).evaluate(original_tiles)` returns38 logit blocks with exact public
slices, scales, original position identity and continuation event records. The
input is a tuple of fresh L25 packed embedding tiles. The 27 interlayer events
explicitly refresh actual L4 outputs to L25. Final normalization sees only the
selected last valid hidden row: actual mask4→3, row rotation, B2048 refresh→25,
actual frozen final RMS→7, then full vocabulary7→6. Original positions are never
renumbered. All expected scales use the actual returned refresh metadata; no
scale field is assigned or relabeled.

Complete source/result payload references, metadata and compiler/key identities
are retained through callbacks. Borrowed original annotations remain untouched;
owned outputs lose inherited public interval annotations. Any entered failure
retains original payloads and borrowed runtime owners until dedicated-process
exit. Failed readiness also requires outer-loader quarantine. No context,
compiled helper or key store is closed by this owner.

Focused `full_synthetic.py` tests explicitly replace child attention/tail/norm
and parameter/linear payloads with stand-ins. They check only parent connections,
conditional event order, actual-scale bookkeeping, final masks/rotation, all38
slices, fail-before-next-work and retention. They are not a full numerical proof.
`full_reference.py` is an independent dense NumPy oracle using Clenshaw or the
actual rounded power coefficients, original Newton recurrences and all domain
checks; its tiny tests compare individual arithmetic with the frozen evaluator.
No model/parameter/input bodies or full reference trajectory have run yet.

The unchanged wholegraph fixtures remain S1 checkpoint embedding row0 and the
first authenticated original calibration S8 prefix. Full numerical execution
awaits root review. No fresh or held-out outcomes select rules or fixtures.
The unproved domain/refresh/error and resource limitations in FULL_DESIGN remain,
including 12,931 S32 refreshes, 166 keys and the 147.84375 GiB raw rotation store.

The v3 successor also pins exact ordered Q100/P14, p60 canonical target and dnum8 before work/after callbacks, and places activation/owner-map setup inside protected try/finally. Both pre-repair negative paths are preserved separately; no arithmetic or coefficient changed.
