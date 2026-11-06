# Qwen expanded-prefix candidate1 selection

**Candidate1 selection completed and its public certificate history passed root's independent replay.** The result has 113 normalization rules: 36 polynomial and 77 newly certified Newton rules. This is selection evidence only. It is not a new-candidate calibration, heldout, row0, encrypted-execution or full-model accuracy result.

Every normalization site uses the same rule: form `[max(epsilon, 0.8*vmin), 1.2*vmax]` from the complete initial native/exact union, expand that interval by a factor of four, then choose the first passing degree in 16/32/64/128/256/512 using the exact relative-`1e-6` certificate plus the fixed linear/geometric grid diagnostics. If none passes, certify the newly computed minimal Newton count, at most 64. All 28 SiLU rules, 28 softmax rules, exp8 and reciprocal maps remain identical to frozen candidate3 (`7f6478…`). The frozen baseline's trace-generation ranges are excluded from the initial bounds.

The full **37,283,359-byte assessments**, **1,110,816-byte candidate manifest** and protected checks are preserved exactly. Root replayed all 648 polynomial attempt certificates/grids, all 77 Newton fallbacks, all 113 domain bounds, 28 protected checks and depth plans for lengths 1–32 using the frozen certificate implementation. This was an independent execution of that implementation, not a different formal certificate implementation. The relative certificates include public rounded coefficient/map interpretation; subsequent binary64 execution, CKKS and refresh errors are outside that ideal-real guarantee. Grid checks are diagnostics. Abstract depth totals are 6,992/13,522/14,398 at lengths 1/8/32; these counts do not establish a realizable ring or refresh schedule.

Root owns the authoritative terminals: actual selection session **48044**, exit 0/reaped (`4aca4e`), and independent replay session **96555**, exit 0/reaped (`12c52b`). The producer and review PIDs/process groups were absent. The exact outer receipt binds the candidate, report, source archive/manifest and grant. The report records completed publication, cleanup and signal restoration, unchanged 36 sources and zero audit denials. There is one selector process, no invented worker report. The reported 13.289-second selection component duration is not model or benchmark timing.

The external grant links the completed [initial3200 observer and private reconstruction proof](../qwen_expanded_prefix_initial_observer_cpu_20261104/README.md) to the unchanged [36-source/150-check admission preparation](../qwen_expanded_prefix_selector_admission_cpu_20261104/README.md). Public copies of those exact admission receipts are included. The source default remains disabled; the separate pinned grant authorized candidate1 only. Candidates2/3 and any subsequent model evaluation require their own reviewed provenance and authorization.

`public_whitelist.json` records all copied paths and hashes. No private snapshot, journals, calibration/token bodies or model tensors were opened or copied by this archiver. The selector itself was authorized to consume the one reconstructed private snapshot; that body's hash is retained only as a public descriptor. The archive preserves output coefficients and certificate histories, not private observations journals. Earlier preparation failures and repairs remain in the linked admission archive.

Run `python -I -B verify.py` for portable public integrity, provenance and count checks; add `--current` to compare current/execution source bytes, current published outputs and process absence. This verifier does not select coefficients, replay certificates, execute the included root scripts, import model code or follow private descriptor paths. Root's exact replay script is included as reviewable provenance.

Pins:

- Candidate: `344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a`.
- Selection report: `7ba0ffae120c7b68b51d2ec1fc040e51e3fd12d989f1ab73316f231091917ad9`.
- Actual 36-source archive: `821ba2035e4c0a1efbcdc9f3ee718c6ecadb2f013cd63ddaec3911d632494b6c`.
- Root replay review: `6c98b64cd506cddeb5f4bd64767e5b7e1483ba5d28ca04a442cd3d7b4fa960de`.
- Root outer receipt: `cbff16bf203e311b503fb3c2f3a37af97f9d4d9b83dc6e4db5ae9153ca3ad352`.

Security independently inspected the public verifier and ran portable/current readback successfully. Its unchanged receipt is retained under `peer-review/security.json`; it did not follow private descriptors or rerun reconstruction/certificates.
