# Verified result snapshot

This compact snapshot preserves the paper-facing five-arm aggregate without bundling data, checkpoints, or private paths. It is descriptive evidence for the accompanying paper and is not a training recipe selector.

| Arm | IoU mean | IoU sample SD | SegAcc mean | $P_d$ mean | $F_a \times 10^4$ mean |
|---|---:|---:|---:|---:|---:|
| Reference-gated | 88.1864 | 0.2277 | 92.5133 | 98.1577 | 0.0634 |
| Uniform | 86.3731 | 0.2776 | 90.7157 | 97.8169 | 0.0686 |
| Observed-only | 88.2321 | 0.4419 | 92.2164 | 97.9621 | 0.0599 |
| Global threshold (EventRefine) | 88.3235 | 0.2085 | 92.5544 | 98.1535 | 0.0626 |
| Classification-only | 84.9389 | 0.6614 | 87.7006 | 96.9378 | 0.0407 |

The rows belong to one matched five-arm evaluation family. The global-threshold row is the integrated paper presentation; the table does not establish that an edge-specific reference is necessary.

The full row-level receipts remain in the private research workspace; this repository keeps only the compact aggregate above.
