# Guarded dispatch-cache opt-in proof

The isolated policy passed **35 tests,4 skipped**, plus a separate secure six-cycle lifetime proof. The default continues to clear JAX caches. With explicit `preserve_dispatch_cache=True` and `low_memory=True`, it preserves ordinary dispatch caches only when the Polynomial registry is empty; otherwise it performs the previous full JAX clear before registry eviction.

Actual cached Polynomial functions returning different levels or scales now retrace safely after cleanup. Array-only JIT functions retain their cache only under opt-in. The secure N8192 Q8/P2 component exercised rotations, plaintext/rescale and ciphertext multiplication plus the compiled bootstrap tensor/relinearization adapter; all retired binding arrays and owner contexts were released while code remained cached.

This is for a dedicated serial evaluator with all other cleanup and synchronization unchanged. Original Bootstrap stage clears and session shutdown are preserved. It is not a total-HBM, TPU performance or full-model validation. The measured flat-removal probe and its general-Polynomial KeyError counterexample remain immutable separate evidence.

[candidate.patch](candidate.patch) is unapplied and passes `git apply --check` against the live base. [summary.json](summary.json), raw logs, before/after hashes and sources.tar.gz preserve the exact tested implementation.
