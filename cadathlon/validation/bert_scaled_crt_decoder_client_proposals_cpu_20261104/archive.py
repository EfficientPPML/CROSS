"""Copy only explicitly listed public source/provenance proposal artifacts."""
from pathlib import Path
import hashlib,json,shutil

SOURCE=Path('/tmp/bert-scaled-crt-decoder-client-proposals-20261104')
REVIEW=Path('/tmp/bert-decoder-client-proposal-root-review-20261104')
OUT=Path('/home/jianming_gatech/CROSS/cadathlon/validation/bert_scaled_crt_decoder_client_proposals_cpu_20261104')
OUT.mkdir(exist_ok=False)
whitelist={n:SOURCE/n for n in ('prepare.py','report.json','terminal_observation.json')}
whitelist['proposal_README.md']=SOURCE/'README.md'
for label in ('original122','combined125'):
 for name in ('source_manifest.json','parent_source_manifest.json','candidate.patch','sources.tar.gz'):
  whitelist[label+'/'+name]=SOURCE/label/name
for name in ('review.json','review.py'):whitelist['root-review/'+name]=REVIEW/name
whitelist['verify.py']=Path('/tmp/verify_bert_decoder_client_proposals_archive_20261104.py')
whitelist['archive.py']=Path(__file__)
for name,path in whitelist.items():
 assert path.is_file()and not path.is_symlink()
 destination=OUT/name;destination.parent.mkdir(exist_ok=True,parents=True)
 shutil.copyfile(path,destination)
(OUT/'root-review/terminal_observation.json').write_text(json.dumps(dict(
 scope='Root-reported authoritative direct tool terminal; archive agent did not own the root command',
 direct_tool_chunk='ac3034',exit_code=0,reaped=True,peer_reported=True,
 review_sha256=hashlib.sha256((REVIEW/'review.json').read_bytes()).hexdigest()),sort_keys=True,indent=2)+'\n')
(OUT/'README.md').write_text('''# Optional scaled-CRT decoder BERT client proposals

Public source-only proposals and independent root readback. **The completed
original122 and combined125 clients remain the launch defaults.** These new
bytes have no client test, installed probe, full-preflight or launch authority.
No backend, context, key, ciphertext, private result or model body was accessed.

Each proposal changes exactly four files: the decoder, its explicit source-guard
overlay pin, the existing source-guard test expectation/alias assertions, and
the README. The decoder is exact reviewed SHA
`9675b300d507fd9c6a3d03656c9df325d8da68a9331b4c92def0512784d0dc26`.
Removing its one new helper and reversing two call substitutions reproduces
the complete predecessor module AST. Successful conversion preserves the exact
normal expression `raw = float(value)` then `raw * scale_inv`. Only conversion
overflow uses the exact binary64 scale ratio; all later numerical gates remain.

The fixed final BERT response is L2 with six Q towers. Their product is181 bits
and centered coefficients are at most180 bits, so this final boundary cannot
trigger integer-to-float overflow. No BERT repair, accuracy, latency or toy5
recovery is claimed. The separate decoder proof is
[here](../ckks_scaled_crt_decoder_cpu_20261104/README.md).

| Proposal | Sources | Unchanged from clean parent | New manifest SHA256 |
|---|---:|---:|---|
| Original response |122|118|`6e3e30ad643921e9546be144e7da0dce485e9dd1be088f31f62f35cd2aa4c5f9`|
| Separate combined response |125|121|`6b2b1985c1f9e7298bf47cf8559fd8a2122b0ad854289fed8674c2c05e8475a0`|

All original producer maps/gates, profile/candidate, worker/collector/CLI
bootstrap, numerical checks and original/separate receipt ownership contracts
remain exact inherited bytes. In particular the old217 and combined286 producer
identities are not rewritten to this consumer decoder. Parent preparation is
[original122](../bert_original_stage5_io_successor_cpu_20261104/README.md) and
[combined125](../bert_prepared_bsgs_client_active_readers_cpu_20261104/README.md).

`report.json` SHA
`2251a4bc580ba2f008a638fca74803d77a48fa666e12c25a1b93f0b4794b2e97`
records static construction checks. `prepare.py` is the exact executed builder;
session97772 was reaped exit0. Root independently checked all247 current,
clean-parent and archive members, the four deltas, complete decoder AST reversal,
guard overlay-only AST change and Q6 arithmetic; its exact receipt/script and
separately labeled reported tool terminal are under `root-review/`. Root direct
chunk ac3034 exited0. These are static source checks, not runtime tests.

Public whitelist: exact two source tars, their old/new manifests and patches;
proposal/preparation reports and scripts; owner/root terminal observations;
root source-review receipt/script; and archive README/verifier/index. Expanded
source trees are not duplicated. The exact tars retain only the already declared
public source closure, including inherited public provenance and source archives.
No private/key/ciphertext/token/model body or followed report descriptor is copied.

Run `python -I -B verify.py` for portable hashes, bounded tar/member readback,
in-memory exact parent reconstruction from the patches, AST/pin/closure checks
and public Q6 arithmetic. Add `--current` to compare the two proposal and clean
parent source roots under `/tmp`; it reads only exact source-map paths. The
verifier imports no candidate module, executes no tests and follows no private
descriptor. Any optional adoption needs a separately reviewed complete client
proof, completed corresponding producer evidence and a distinct launch grant.
The active server46674, original producers and ledgers are untouched.
''')
files={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
       for p in OUT.rglob('*')if p.is_file()}
(OUT/'SHA256SUMS.json').write_text(json.dumps(files,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(path=str(OUT),indexed_files=len(files),total_files=len(files)+1)))
