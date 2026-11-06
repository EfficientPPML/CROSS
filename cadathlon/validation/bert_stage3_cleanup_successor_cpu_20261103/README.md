Stage3 cleanup signal helper successor

This isolated helper fixes demonstrated cleanup-boundary defects without changing any measured stage3-v2 or running-job source. The new helper SHA256 is `1798455cdd88d4bbac5e1769e6b3226676ccb9901e7be6faf383e7ad063c1ef6`. The overlay changes only `cleanup.py`; parent integration and five-handler behavior require their own tests.

The final source-bound run passed 30 tests in 0.13 seconds (0.27496 seconds including wrapper), with all five source files unchanged and archive readback verified. Tests cover partially mutating signal setters, both mask/setup failure paths, attempt-all handler/mask restoration, body/incoming-primary preservation, cleanup-only KeyboardInterrupt/SystemExit precedence, actual terminal-record publication despite installation failure, and four real-signal modes in fresh single-thread Python subprocesses. Tests explicitly reject successful restoration when a setter failed before changing its handler; other restorations are still attempted and failure propagates.

`cleanup_signals()` always yields the owned cleanup/publication body even if shield installation fails. It records original handlers before mutation, defers repeated INT/TERM/HUP during cleanup, and reports setup/restoration failures after attempting the body. `termination_signals()` does not yield work on installation failure and uses a single-fire handler, recording repeat signals. Both preserve active primary exceptions and prioritize cleanup-only cancellation over ordinary cleanup errors.

The historical helper is preserved in both negative source archives. The first negative run produced one assertion failure followed by pytest's handling of an uncaught negative KeyboardInterrupt (exit 2). The second run only changed the regression to catch BaseException and assert its identity; all five historical regression cases then failed as expected (exit 1). Neither negative run is counted as a successful test.

Main-thread `pthread_sigmask` protects only that thread. Signals delivered to native worker threads can still schedule Python main-handler execution; this helper and these single-thread subprocess tests do not prove arbitrary process-wide atomicity, delivery timing, or primary identity after the scope has exited. Security's separate combined loader/wrapper stress and worker-delivery counterexample remain separate evidence.

A collector that publishes success inside this cleanup context can still need a final fail-closed publication if scope exit raises. Root is addressing that separately in stage3-v3; this artifact does not claim the full stage3 report is fail-closed, or that any actual encryption/model operation was run. Existing successful measured evidence remains tied to its original bytes.

`python verify.py` verifies exact source archives, the candidate overlay, all retained artifact bytes and the recorded test outcomes. No model, keys, ciphertexts or numerical backend are needed.
