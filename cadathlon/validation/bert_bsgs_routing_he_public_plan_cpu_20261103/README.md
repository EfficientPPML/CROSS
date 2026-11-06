Public preparation for a future BSGS encrypted routing component passed25
checks on CPU120–123. All8 source files matched before/after/archive and all4
copied public inputs matched their pinned originals. No key/ciphertext payload,
model asset, private input, JAX backend or HE operation was accessed.

The two fixed cases are canonical/gain1 and genuine-L6-product/gain1/8. The
noncanonical input scale is1.3872675178787955e18 atL5; it is never relabelled.
Each case has forward, inverse-of-forward and independent hostile-input inverse
checkpoints. Their18 outputs cover1,179,648 physical slots. The independent
inverse first checkpoint checks the entire rotated dense input, including all
hostile padding; subsequent checkpoints gather the chosen global-key entries.
All future numerical gates remain absolute1e-6.

The exact public profile/key check establishes36 required keys belong to the
existing130 inventory. Route counts are72 rotations,168 public products,
156 adds and42 composite rescales; the separate prestage adds one ciphertext
product/rescale. Raw stored payload is200MiB of input plus148MiB of output.
Temporary writes, headers, operational margin, controls/compiler and actual
host/device memory remain separate obligations. This is no noise, HE accuracy,
performance or launch claim.

`sources.tar.gz` contains the exact plan/tests/README and four pinned public
dependencies; no runtime owner or mutable ciphertext/key files are included.
`input_sources.json` preserves their original paths and hashes. The independent
owner read-only review is copied unchanged; its separate negative evidence is
preserved with the BSGS owner proof. The public plan is the authoritative
derived artifact here, not a replacement for that owner proof.

The source-bound pytest call returned exit0 directly from exec_command and
provided no session identifier or PID. The OS PID and separate reaping history
were not captured. `terminal_observation.json` preserves that limited evidence;
no process history is inferred. The plan/readback call also returned exit0.

Run `python verify_archive.py --recompute` for portable archive/XML/hash checks
and pure public NumPy plan replay. No original `/tmp` paths are required.
