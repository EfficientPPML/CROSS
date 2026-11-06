# Candidate3 original published regression preflight

This source-bound preparation passed 110 tests plus 19 subtests (129 XML cases), with 70 exact before/archive/after source files. It did not read the original published token body or run a model. The original 55 calibration/runtime dependencies and candidate manifest `7f6478a6…` remain unchanged. The wrapper requires the completed original416 calibration evidence before authenticating the separately pinned original published inputs.

The actual preflight affinity was CPU168–171; the 16.8423-second elapsed value is test-wrapper time, not a model or HE benchmark. The authorized subsequent actual regression has a separate 71-source closure and result. These preparation tests do not establish that result. Fixed numeric/domain/semantic gates and 44/29/6 aggregate top1 thresholds are unchanged.

`preflight-v1/sources.tar.gz` preserves every tested source, including the calibration prerequisite archive/report and copied original calibrated inputs already authorized during calibration. It excludes the external published token body. Public source and hash evidence is archived; no model weight, key, ciphertext, or private activation array is included.

Run `python verify.py` for offline source and prerequisite readback. Signal tests cover caught install/cleanup/restoration failures; main-thread signal masking does not make arbitrary native-thread process signal delivery atomic.
