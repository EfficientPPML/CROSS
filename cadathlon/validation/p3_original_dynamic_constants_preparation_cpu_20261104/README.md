# Original P3 dynamic-constant preparation

The isolated alternative adds only `dynamic_constants=True` to the original baseline's `Mapping` call. It keeps `nn.vectorize`, the exact `PackingPolicy(lazy_constants=True)`, original model/input shape, batch and device arguments. `baseline_original.py`, `baseline_dynamic_constants.py` and `candidate.patch` show the complete change. The candidate SHA256 is `b93c89318086138b82c7b2d30cb525390a431b17253a0dfcc301c6f88caafeb1`.

Ten CPU checks passed in 2.51 seconds on cores 120–123:

- Six original public cases run the real synthetic PyTorch frontend and packing for both baseline functions. Mapping construction is a recording stub. Frontend digests, packed fingerprints, complete N32768/Q16/P4/dnum4/depth7 rings, weights, batch and device arguments agree. The generated clear outputs also match the original public fixture answers.
- Three unchanged `DynamicConstantsCompilationTest` cases exercise a small clear uint32 resident-constant function across two inputs, plus cache deferral during concurrent cleanup and after an exception. These compile and run on the CPU; they do not compile or execute an encrypted CNN.
- One AST check removes the single `dynamic_constants=True` keyword and verifies equality with the entire original baseline AST.

Session 35135 exited zero and was reaped in tool chunk `140f65`. The worker PID 3331910 and process group 3331903 were absent at terminal readback. All 100 source files match before/after/archive; the 96 copied original repository files also remained unchanged. The raw log retains eight harmless unknown-mark warnings from the unchanged test module; XML records ten passes with no failures, errors or skips. `owner-terminal.json` records the owner observation, not a new independently measured execution.

No CKKS context, Polynomial or key was constructed, and no encrypted request or TPU computation ran. The guarded cryptographic entrypoints recorded zero attempts. JAX reported CPU devices; the optional `libtpu` metadata import was denied. The public fixture arrays are generated deterministically in memory; no learned weights, user input, ciphertext or private keys are included. There is no encrypted numerical result, setup/memory comparison, calibrated timing or speed claim. The 2.51-second test duration is not evaluation latency.

This preparation leaves the original six-case TPU queue, packaged profiles, existing calibrations and all production baselines unchanged. It does not replace the pending complete final-source original P3 sweep. A later separately admitted comparison must preserve all original cases/rings/gates and include setup through cleanup. The existing runner selects `baseline.py`, so any eventual alternative must be placed in its own comparison checkout; no production baseline swap or new runner is supplied here. The current Mapping option is restricted to whole-program, bootstrap-free execution on one device, matching the original P3 runner's explicit single-device selection.

`sources.tar.gz` contains the frozen 100 public source/fixture files. The two source maps, original-copy origins, raw log, XML, report and owner receipt are preserved without modification. The candidate and original baseline copies are also checked against their archive members. The portable verifier reads only these public files and parses source as AST; it does not import the runtime, execute tests or follow provenance paths.

```sh
python -B verify.py
# Optional current source-byte comparison; still no runtime import:
python -B verify.py --current /tmp/p3-original-dynamic-constants-candidate-20261104/source
```

Root independently reviewed the source delta, all 10 CPU checks, source/archive identity and process retirement; all assertions passed. The unchanged receipt is included as `root-review.json`. Root owns staging. This preparation carries no launch, promotion or calibration authority.
