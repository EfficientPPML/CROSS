# Isolated transformer collector ownership fix

The candidate passed **65 CPU tests in 5.19 seconds**. It gives every public-case subprocess its own key directory under the selected shared parent, records ownership before launch, and removes only that directory after the owned process group disappears. A failed process or key cleanup prevents the next case from launching.

Cancellation during cleanup still terminates the sweep. Existing failures survive secondary cleanup or final-evidence-write errors. Real subprocess tests cover ordinary exit, forced termination, launch failure, shared-file preservation and outer SIGTERM; focused fault injection covers retained live groups, symlink replacement and error precedence.

This candidate is **not applied** while the Qwen2 proof uses the frozen live collector. [candidate.patch](candidate.patch) records the proposed change; [report.json](report.json) binds the final test log to five unchanged exact source files. `sources/` and `sources.tar.gz` preserve those bytes. Earlier logs are superseded and not additive. This is lifecycle evidence, not encrypted or TPU validation.

An uncatchable SIGKILL of the outer collector cannot perform cleanup. The early process record identifies the child and owned directory for recovery. Remaining process-group members conservatively retain files and stop the sweep.
