# Separate expanded-prefix Qwen protocol proposal

This public design declares3200 calibration entries (the same96 original prompts at every length1..32 plus original64teacher/64free steps) and binds the old416 subset. It preserves fixed gates and proposes a uniform113-site normalization-only selection policy. Existing candidate3 and all old results remain unchanged. No input/tokenizer, selection, certificate/model or HE execution occurred.

The new authored32-prompt/all-prefix validation recipe is frozen as a proposal, not yet tokenized or certified disjoint. Before any new calibration result or selection, a separately reviewed tokenizer phase must check expanded3072+originaldecode/oldsets/row0 full-sequence disjointness and report entries versus unique contexts by length. Entries/denominators remain fixed despite duplicates. Row0 tokenID0 is a post-freeze regression only. See DESIGN.md and public-v3/plan.json for exact rules and pending gates.

Final public-v3 passed68 tests and23 exact source/current/execution/archive checks with zero audited old input/model/backend reads. Preserved public-v2 has all68 tests passing but overallfailure because inventory erroneously expected67; the sole successor change is that count. Preserved draft-v1 predates expanded-prefix disjointness wording and has development27-test evidence only, without a startup archive claim.

The archive includes public original protocols/coefficient manifests/algorithm source and newly authored validation text design. It contains no old calibration/validation token bodies, checkpoint tensors, keys or ciphertext. No claim of1088 unique validation contexts or already achieved untouched token disjointness is made. Run `python -B verify.py` for offline public readback.

The original416 subset denotes the same workload entry, prompt/seed, mode and step mapping. A newly selected candidate may generate different free-running tokens; it must be compared to native/exact outputs on its actual prefix. No equality to old approximate free prefixes is asserted.
