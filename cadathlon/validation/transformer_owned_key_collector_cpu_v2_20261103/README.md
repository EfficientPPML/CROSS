# Transformer collector ownership follow-up

This isolated candidate passed **78 CPU tests in 5.53 seconds**, with five source files unchanged. It remains unapplied while Qwen uses the frozen collector.

The collector and outer wrapper pin their created directory inode with a descriptor, verify device/inode identity before deletion, and retain moved or replaced directories. The outer wrapper requires explicit successful inner process, key-directory and handle cleanup once the case collector has been entered. Missing or corrupt metadata causes retention; a recorded launch failure before any child exists can still clean both owned directories.

Cancellation remains primary if final evidence writing fails. Tests cover real ordinary-directory replacement, nested parent retention, partial setup, descriptor acquisition failure, unknown metadata and the existing process cancellation paths. The original reproducer deleted the replacement directory; its rerun against this candidate preserves it and reports failed cleanup.

[candidate.patch](candidate.patch) applies to the current live base without modifying it. [report.json](report.json) records exact source/base hashes. The earlier65-test archive is unchanged; the interim72 checkpoint is preserved separately and is not additive.

Directory identity is a concurrency guard, not an OS sandbox or atomic defense against arbitrary same-user manipulation during recursive deletion. These checks do not establish encryption, model accuracy or accelerator performance.
