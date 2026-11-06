# Historical P1 student calibration before student-v2

The report, calibration manifest and aggregate C3/C4 archives were copied
byte-for-byte before the new evaluator can publish a replacement. They describe
the historical 262.14-second run, not a `student-v2` calibration.

`provenance.json` preserves the historical report's source hashes. Twenty of
its 27 source versions were recovered from current files or the Git index only
when their bytes matched the recorded SHA256 exactly. Seven historical source
versions were unavailable and are explicitly listed; current code is not
substituted for them. Recovered source bytes use `.txt` suffixes to prevent test
discovery from executing archived code. The original paths and stored paths
are both recorded.

No per-image labels, pixels, predictions, ciphertexts or key arrays were copied.
These files are historical validation evidence and are not the current scoring
calibration required by the strengthened student evaluator.
