# Dispatch-cache lifetime audit

The isolated explicit-array component passed six secure CPU cycles at levels3,2,1, repeated. It exercised CompiledArithmetic, the client codec, and the tensor/relinearization adapter used by compiled bootstrap. No Polynomial crossed a JAX transformation. All retired binding arrays were released at checkpoints. After owners were dropped, all2293 tracked binding arrays and1111 tracked facade objects were released while compiled executable caches remained alive; a final global JAX clear released no additional tracked objects. A changed ambient CPU default device preserved the operand device and numerical result.

**Flat removal is unsafe for the general Polynomial API.** A separate tiny arithmetic-only test reproduces KeyError when a cached JIT function returns a different Polynomial output key after the registry alone is cleared. The counterexample and its exact sources are preserved.

An opt-in could be investigated only for a dedicated explicit-array execution graph, retaining all other synchronization and control eviction, and falling back to a full JAX clear whenever Polynomial registry metadata exists. Bootstrap still performs its original stage-level global clears. This audit does not measure total HBM, compiled-code memory, accelerator speed or a full transformer.

[summary.json](summary.json) states the limits; exact source hashes, raw logs and sources.tar.gz preserve both tests. The earlier audit is included separately and is not additive.
