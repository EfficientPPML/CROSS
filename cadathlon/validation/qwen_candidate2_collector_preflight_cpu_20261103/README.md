Candidate2 calibration collector preflight only. No checkpoint model execution, held-out evaluation, encrypted execution, or calibration pass is claimed.

The frozen candidate manifest is `0af45a8fc6d13b9d429ee1af2311b2cb6ac0aa4cde02eca56478472762482489`. Before model/assets are read, the pure-stdlib prerequisite verifies all 20 selection inputs, successful selection report and exact unique-member source archive, the original separate five calibration files, and actual arithmetic/native dependencies. Selection and calibration protocols remain distinct. The 416-context workload and fixed numeric/aggregate gates remain unchanged; teacher and free trajectories compare native operators on the actual same tokens. A stopped free trajectory never substitutes native tokens or receives suffix success.

The collector carries the previously reviewed cleanup-primary preservation and reports finite metric failure on subtraction/cast/nRMSE overflow. Its scaled RMS implementation avoids spurious overflow on large matching finite arrays; thresholds are unchanged. Source tests include all input substitutions, mutually changed corpus/tokenization/pins, failed selection, duplicate archive members, current runtime substitution, cancellation/publication/cleanup, unchanged public arithmetic/depth, and tiny native model hook invariance.

Authoritative preflight session 19379 exited 0 and was reaped. The separate checkpoint has 66 tests plus 19 subtests, 53 unchanged source files, 5.64 s pytest time and 7.86 s outer time. These are not model performance measurements. `sources.tar.gz` contains the exact inputs, helper/test sources, and installed Transformers Python source closure; no checkpoint tensors, activations, keys, or ciphertexts are included. It is a source audit, not a complete binary environment image.

Offline verification: `python verify_archive.py`. Proposed smoke command (requires a separate grant):

```sh
taskset -c 140-143 env JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 /tmp/cross-pretrained-validation-env/bin/python /tmp/qwen-candidate2-calibration-20261103/check_approximate.py --output /tmp/qwen-candidate2-calibration-20261103/smoke-v1 --smoke
```

The full calibration is the same command without `--smoke` and with a fresh output directory. No model command was run during this preflight.
