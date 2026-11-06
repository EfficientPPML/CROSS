# Genuine encrypted BSGS score routing

All eighteen ordered checkpoints passed the fixed absolute1e-6 gate across
1,179,648 physical slots. Maximum absolute error was2.0359652806592398e-10.
Canonical and actual noncanonical scales cover forward routing, roundtrip
inverses and independent hostile-input inverses. No numerical gate was relaxed.

The three separate fixture/server/validator processes used the same147 frozen
sources and retained authenticated p60/d8 keys, N131072,Q100/P14. Actual public
input encryption took151.396s, keyless server13745.564s and private client
validation116.590s. Root observed/reaped all three outer sessions5310/89442/6747
at exit0 and independently checked every worker/group absent. All18 encrypted
outputs and terminal server evidence authenticated before the private loader
entered. Complete input/output payload readback occurred before validation.

Six routing maps used72 rotations,168 public products,156 additions and42
composite rescales; input preparation separately used three level reductions
and one ciphertext product/rescale. Input scales were1.1467305003699405e18 and
1.3872675178787955e18 atL5. Source/current/execution/archive bytes all matched.
The server OS maximum child RSS was133933540KiB; this is not aggregate memory
or a capacity guarantee. Timings reuse keys and exclude initial key generation.

This is one genuine component correctness result. Full BERT integration, ranked
accuracy, controlled speedup, full Qwen and competition latency remain separate.
Existing keys and ciphertext bodies remain in their local owned workspaces;
only public manifests, reports, logs and exact source archives are stored here.
The earlier fixture-only archive remains a historical intermediate checkpoint.

Run `python verify.py` for portable verification without private or payload reads.
