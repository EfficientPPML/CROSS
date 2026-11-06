# Public Qwen Q/K/V parameter export

The authorized public-only export passed for all84 pinned BF16 tensors:28 full2048×1024 Q matrices and56 full1024×1024 K/V matrices. Exact bit promotion tofloat64, every array/file digest/readback, pinned checkpoint/config/candidate and all4 source bytes passed. Elapsed6.014s is export preprocessing time, not encrypted timing. No model construction/forward, backend, keys or encrypted work occurred. Tensor bodies remain only in the local experimental workspace and are absent from this archive.

The completed export is bound to result-v1/sources.tar.gz, not to the later cleanup repair. Independent review found an fdopen-failure descriptor leak; fd-negative-v1 preserves its failing synthetic regression. The separate fd-successor-v2 repair tracks ownership until stream handoff;14 focused tests pass with3 startup/end source bytes identical. No changed-export rerun is claimed.

The84 tensor metadata/descriptors and checkpoint identities are public. This is parameter preparation for an isolated joined ideal Qwen attention component, not model inference or HE readiness.
