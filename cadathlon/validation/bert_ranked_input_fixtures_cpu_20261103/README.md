The actual upstream BERT ranked selectors and tokenizer generated all nine
100/200/408-record cases with base seeds 42/43/44: 2,124 selected records.
The 27 source files exactly match the reviewed 32-check preflight. Original
selection order, raw sentence/label exports and fixed-128 tokenizer output
match independent references byte for byte. Each private case retains six
files under directories mode 0700 and files mode 0600. The three full-408
preprocessed inputs are byte-identical, as required by the upstream exporter.

Root independently reopened all 54 private files, authenticated the workspace
inode and all assets, recomputed selection and tokenization, and verified
exact bytes and permissions. `private_readback.json` contains aggregates only.
No private sentences, target order, labels or token payloads are archived here.
The original size0 workspace remains separate.

The collector completed in 3.916 seconds with its final success, cleanup and
terminal-publication flags true. Its outer session was 61991; the historical
terminal poll output was unavailable after context compaction, so no outer
exit-code or reaping assertion is made. The later independent readback session
74483 exited zero and was reaped. This limitation does not alter the archived
collector report or invent missing process evidence.

Run `python verify_archive.py` for portable public integrity checks. Optional
`/tmp/cross-pretrained-validation-env/bin/python verify_private.py` reopens
the retained local private workspace and assets and prints only aggregate
checks. The latter requires the original paths and installed tokenizer stack.

This is actual client input preparation only: no embeddings, model forward,
key generation, encrypted execution, ranked accuracy or benchmark result.
The successful [preflight](../bert_ranked_input_fixture_preflight_cpu_20261103/README.md)
and historical development proof remain separate.
