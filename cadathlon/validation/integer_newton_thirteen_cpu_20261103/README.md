# Thirteen-update integer-correction inverse component

The original isolated candidate **passed** thirteen genuine encrypted Newton updates on CPU. Every 32,768-slot checkpoint met the unchanged absolute `1e-6` gate against its binary64 circuit; worst error was **1.829385976e-8**. Each of four independently packed domain segments also passed against `4/(128*x)` at its predeclared update. Session30428 returned exit0 and was reaped; its shell and child were absent at audit, cleanup reported no errors, and all **99 exact source files** matched before execution, the startup archive, final execution and the preserved checkout. Input ciphertext payload and metadata remained unchanged.

| Conditional stage | Declared update | Output level | Maximum true-inverse error |
|---|---:|---:|---:|
| 1 | 9 | 9 | 1.825794183e-8 |
| 2 | 12 | 3 | 1.639959457e-8 |
| 3 | 13 | 1 | 1.559223151e-8 |
| 4 | 13 | 1 | 1.666440586e-8 |

Each segment contains 8,192 slots cycling through 258 public points: interval endpoints, linear/geometric interiors, adjacent representable boundary values and the public inactive midpoint. The intervals and iteration counts were selected analytically before encrypted execution. Earlier segments are evaluated further as part of the fixed packed circuit, but their declared results use only their prescribed checkpoint. All client decoding happens after the complete thirteen-update server circuit.

The secure profile is N65536, CD2, depth26, Q54/P7 and dnum8. Actual log2(QP) is **1726.2820288254172**, with integer bit length1727 below the library's1747-bit classical128-bit ceiling; actual P coverage is checked. The keyless server uses only public evaluation material and existing multiplication, add/subtract and plaintext-add operations. Both key-generation module aliases and server codecs are guarded. Client and server are separate objects in one dedicated process; this is not an OS sandbox.

For public C32 and s4, `z <- z*(2-C*x*z)` uses an addition-chain integer correction at unchanged level and scale. A high-precision public backward planner determines the initial scale; an actual affine encode/multiply/rescale realizes it. Every subsequent binary64 scale equals the planned recurrence exactly, without metadata relabeling. Output levels are25,23,...,1. The final scale is7.114398630806586e16, a relative1.521e-13 from the requested canonical scale; the report retains the actual value. The smallest signed message-capacity headroom is60.37bits. The largest observed excursion above the ideal upper bound8 is1.64204e-8, and no negative excursion was observed. These observations are not a uniform CKKS noise bound.

Elapsed time including final cleanup was **3189.5171s**, with process peak host RSS **13,202,848KiB**. Raw evaluation keys are244MiB and the thirteen retained ciphertext payloads total182MiB; NTT tables, controls, compiled executables and host workspaces account for additional storage. This is an uncontrolled shared-host CPU observation. The original root-owned larger-ring proof remains a separate experiment; different rings/layouts and host load preclude a controlled speed comparison.

`actual/` contains the unchanged result, log and startup source archive. `prelaunch/` retains14 passing source-bound contract tests and the actual-ring public preflight. The core/common closure was copied from the sealed optimized-sweep snapshot, including its explicit compiled arithmetic, scoped NTT and guarded dispatch options; `base_manifest.json` preserves its identity. The loose helper/driver sources match the measured archive. `audit.json` records authoritative process outcome and independent hash checks. Run `python verify_archive.py` to verify the retained evidence without loading JAX.

This proves four independently sampled conditional inverse components. It does **not** compose all five softmax normalization rounds, execute a refresh, evaluate a full attention block/model, establish accelerator behavior or calibrate a score. The prior first-round inverse remains a separate proof. The helper is unpromoted; general API hardening will be preserved as a new candidate, never attributed to this measured source. No keys, ciphertext payloads or private model/input/reference arrays are retained here.
