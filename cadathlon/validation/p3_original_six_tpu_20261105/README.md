# Original six-case P3 TPU sweep — all six passed

The original LeNet and AlexNetTiny fixtures, each at batches 1, 2 and 4, passed in one unchanged-source TPU sweep. The original collector exited0; the complete run took **3369.48s (56.2 minutes)**. All128 measured source/fixture files matched the start, every case, end, current files and archive. The129-file preparation archive additionally includes its API_REFERENCE sentinel.

See the [complete final archive](final/README.md) for all30 case files, exact metrics, source archives and original terminal/cleanup receipts. [Root capsule readback](root-final-capsule-readback.json) independently verified56 indexed artifacts,51 origin copies and14 source definitions. The original root admission also verified completed process groups and removed only the identity-checked empty key parent.

Maximum absolute error across the six cases was9.141108009425025e-13, with top-1 agreement1.0 in every case. These are source-bound recorded synthetic encrypted results; decrypted logits were not retained for independent recomputation. The full original cold sweep is separate from the compact145-second schedule and does not establish trained accuracy or ranked performance.

Earlier partial checkpoints remain unchanged: [batch1](checkpoints/test1/README.md), [batch2](checkpoints/test2/README.md), [batch4](checkpoints/test3/README.md).
