The ranked BERT input builder passed 32 mocked checks over 27 unchanged
source files. It uses the pinned upstream selection, seed and tokenizer
function bodies for sizes 100/200/408 and base seeds 42/43/44. All nine cases
are required; the full-408 order is identical across seeds by upstream design.

The checks cover exact selection/order/tokenization bytes, private0700/0600
outputs, invalid cases, model construction/forward denial, interrupted guard
installation/restoration, owned workspace cleanup and terminal publication.
The collector rechecks the source archive before retaining successful private
outputs. A zero-code SystemExit from a failed body cannot produce CLI success.
No real dataset, tokenizer or checkpoint asset open was attempted in preflight.

Root independently verified current/snapshot/archive source bytes, test counts,
the read audit and absent producer PIDs. Session32066 exited zero and was reaped.
This is preparation evidence only; actual input generation, model inference,
encrypted execution and ranked accuracy are separate results.

Run `python verify_archive.py` for an offline check without external data reads.
