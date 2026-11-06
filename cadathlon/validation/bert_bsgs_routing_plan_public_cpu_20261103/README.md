This public NumPy proof rearranges the existing factored BERT score masks into
baby-step/giant-step groups. It does not implement or run encrypted arithmetic.
All 128 key-tile/offset mappings are checked in both directions, with both
supported forward gains and two dense inputs. The 768 comparisons cover
50,331,648 physical slots, exactly matching an independent direct scatter.
All 392 checks pass over ten unchanged source/public-input files. Session
65329 exited zero and was reaped; root verified the archive, current/original
source hashes and absent producer processes.

For output-coordinate masks and rotation `R_s(x)[j] = x[j+s]`, the identity is
`sum_k d_k R_k(x) = sum_g R_g(sum_b R_-g(d_(g+b)) R_b(x))`.
The actual existing masks are in this output-coordinate convention. A second
review confirmed the identity, mask rotation direction and operation counts.

| Per routing map | Current | Proposed |
| --- | ---: | ---: |
| Internal rotations | 26 | 11 |
| Plaintext multiplication terms | 28 | 28 |
| Ciphertext additions | 26 | 26 |
| Rescale calls | 2 | 7 |
| Critical rescale depth | 2 | 2 |

The 12 head terms use four baby steps and three giant groups; the 16 local-key
terms use four of each. Outer tile rotations remain unchanged. The combined
forward/inverse routing inventory decreases from 66 to 36 keys, all already
present in the current inventory. Other model and refresh keys still apply.

Applied to the separately recorded full-K128 operation count, the proposal
removes 368,640 rotations and adds 122,880 rescale calls across 24,576 routing
maps. Total rotations excluding bootstrap internals would change from 906,829
to 538,189. These counts are conditional on implementing the proposal exactly;
they are not runtime measurements or a completed model result.

Every giant group performs a real weighted sum/rescale before the final
rotation/addition. Equal input scales imply equal group output scale under
the reviewed binary64 recurrence, but a future encrypted owner must validate
actual returned metadata. Key-switch, summation and rescale order differ from
the measured current route. Genuine encrypted numerical, memory and timing
checks remain required; fewer rotations alone do not establish a speedup.

`python verify_archive.py` checks the portable public evidence without importing
NumPy, a backend or any private data. The running BERT server and all other
frozen encrypted jobs were left unchanged.
