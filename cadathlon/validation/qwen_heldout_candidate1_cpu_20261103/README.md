# Qwen candidate1 public held-out failure

The unchanged candidate that passed all416 calibration contexts **failed the original112-context public held-out validation**. All112 required entries are retained:87 approximate trajectories completed,22 stopped at declared SiLU domains, and3 free-running suffixes are explicitly unmeasured after a failed prefix. No clipping, native-token substitution or domain/coefficients revision occurred.

Prefill matches were31/48 (S8:16/16; S16:0/16; S32:15/16), below the aggregate44/48 requirement. Teacher matches were28/32, below29/32. Seven of eight complete free sequences matched, meeting its6/8 threshold; this does not override the failed numerical/domain and prefill/teacher gates. Every completed trajectory passed its fixed native operator/block/full-logit comparisons. Stopped trajectories retain only their actually computed prefixes; no suffix accuracy is claimed.

Sixteen S16 prefills stopped at layer1 SiLU, one S32 prefill at layer3, and decode seed Q6_05's four teacher prefixes plus first free prefix at layer9. The remaining three free suffixes were not evaluated. The report preserves actual violations; they were not used to revise this candidate.

Session3412 exited1 and was reaped; shell1154841, time1154845 and Python1154846 were independently absent. The run took293.759s internally. All38 archived source/input dependencies and seven pinned model/tokenizer assets were unchanged, including all31 calibrated runtime files and the frozen candidate manifest. The archive embeds the exact successful full416 calibration report and source archive as prerequisites. No HE/JAX/TPU ran.

Source provenance passed; numerical validation failed. `python verify_archive.py` recomputes integrity, histories, counts and fixed gates offline. The earlier calibration success remains separately archived in `../qwen_approximate_calibration_cpu_20261103`.

A post-launch review identified an unprotected `gc.collect()` in final cleanup that could mask a primary failure if cleanup itself were interrupted. No such error occurred here. Separate cleanup-only wrapper hardening is outside this measured source set. No candidate revision or new validation is authorized by this evidence.
