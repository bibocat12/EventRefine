# Provenance and scope

## Release boundary

This repository contains the fixed EventRefine implementation: sensor/local-attention encoding, the feature-space graph update, the global reference threshold, EV-UAV archive adaptation, training, VAL evaluation, and explicit TEST evaluation. It does not contain the EV-UAV dataset, checkpoints, logs, or private campaign outputs.

## Data contract

The adapter expects the official EV-UAV split layout:

```text
<EV_UAV_DATA_ROOT>/train/train_*.npz
<EV_UAV_DATA_ROOT>/val/val_*.npz
<EV_UAV_DATA_ROOT>/test/test_*.npz
```

Training reads TRAIN and VAL only. TEST evaluation requires both `--split test` and `--allow-test`.

The last verified campaign census was 99 TRAIN clips, 24 development clips / 1,424,330 events, and 24 final-evaluation clips / 2,074,586 events. The repository does not assume those counts without reading the supplied archives.

## Model contract

- 64-channel event representation.
- Four-head local attention in groups of 1,024 serialized events.
- Deterministic spatial-cell/local-order candidates with `k=8`; this is not exact KNN.
- 16-dimensional relational projection.
- Global reference is the mean reference score over candidate slots in the current input.
- `w_ij = max(a_ij - mean(b), 0)`.
- Message denominator is `1 + sum(abs(w_ij))`.
- Zero-initialized output projection preserves the identity at initialization.
- The motion path is computed by the checkpoint-compatible backbone, while the selected classification representation is the sensor branch.

## Results

The paper's verified reference-gate family contains five arms × three seeds. The global-threshold EventRefine presentation row is 88.3235% IoU mean with 0.2085 percentage-point sample standard deviation; the compact result artifact is stored under `results/`. This number is a result record, not a claim that TEST ordering was used to redesign the method.

## External evaluator

The official EV-UAV evaluator is not redistributed. The upstream reference repository is documented in `THIRD_PARTY.md`; its inspected snapshot did not provide a clear license file. Supply an evaluator checkout separately and follow its license. The default evaluator command reports direct event metrics and writes a receipt; it does not silently import an external evaluator.
