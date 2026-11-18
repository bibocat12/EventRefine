# EventRefine

EventRefine is a feature-space graph-refinement model for event-wise tiny-UAV
segmentation on EV-UAV. The model uses **pair-reference graph refinement**:
for each candidate edge, the observed affinity is compared with the reference
score computed for that same edge. The resulting relation weight is used by the
feature-space refinement operator before the final classifier.

The public default release uses:

- local-attention sensor encoding;
- pair-reference graph refinement;
- deterministic local candidates with `k=8`;
- full-clip processing over 8 seconds;
- final cross-entropy plus threshold-shifted Lovasz loss with weight `0.15`;
- checkpoint selection by VAL IoU at foreground probability threshold `0.90`.

## Verified official TEST result

The result below is the official EV-UAV TEST aggregate over seeds `37`, `1337`,
and `2024`. It is reported as mean +/- sample standard deviation.

| Metric | Pair-reference, `k=8`, full-clip 8 s |
|---|---:|
| IoU (%) | **88.1343 +/- 0.4182** |
| Foreground-event ACC (%) | **92.1250 +/- 0.3544** |
| $P_d$ (%) | **94.2808 +/- 0.3840** |
| $F_a \times 10^4$ | **0.0569 +/- 0.0029** |

Here, ACC is foreground-event recall, `TP / (TP + FN)`, not overall event
accuracy.

## Checkpoints

The three verified pair-reference checkpoints are available in this Google
Drive folder:

**[Download EventRefine checkpoints from Google Drive](https://drive.google.com/drive/folders/1i_YvLVaLH6knxhC8z0matg-zVRDHQ25B?usp=sharing)**

The Drive package is expected to contain:

```text
eventrefine_pair_reference_k8_fullclip8s_seed37.pt
eventrefine_pair_reference_k8_fullclip8s_seed1337.pt
eventrefine_pair_reference_k8_fullclip8s_seed2024.pt
eventrefine_pair_reference.yaml
train_normalization.json
metadata/seed37_training_summary.json
metadata/seed1337_training_summary.json
metadata/seed2024_training_summary.json
```

Checkpoint SHA-256 values:

```text
seed 37:   9864e33ef0cc7b764cf7253d176716421d05207f184ec15003708f22c0a9c9d2
seed 1337: 1066a6353eaf37a604739f656c451ab42493fa53fdea52943cf20581eb14fc8f
seed 2024: 6ffb9bf7829217fac1322893b5f25ecb7780e8933ec75c8612cc870e36b6fc3a
```

The checkpoints are not stored in this Git repository. The EV-UAV dataset,
TRAIN normalizer, private run receipts, and official evaluator are also not
redistributed as part of the source release. The Drive package contains the
normalizer needed for the released checkpoints.

## Repository layout

```text
src/eventrefine/       model, EV-UAV adapter, loss, runtime, and metrics
configs/               public fail-closed configurations
scripts/               training, evaluation, normalization, and inspection CLIs
```

## Installation

Create an environment and install the package:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

For CUDA execution, install the PyTorch build appropriate for the target
platform. The package requires PyTorch, NumPy, and PyYAML.

## EV-UAV dataset

The dataset must be provided separately. Set its root explicitly:

```bash
export EV_UAV_DATA_ROOT=/path/to/EV-UAV-dataset
```

The expected archive layout is:

```text
$EV_UAV_DATA_ROOT/
  train/train_*.npz
  val/val_*.npz
  test/test_*.npz
```

The adapter validates event alignment, monotonic timestamps, coordinates,
labels, and optional structured event fields.

Inspect a split without opening TEST:

```bash
PYTHONPATH=src python scripts/inspect_dataset.py \
  --data-root "$EV_UAV_DATA_ROOT" \
  --split val
```

## TRAIN normalization

Normalization must be fitted on TRAIN only and reused unchanged for VAL and
TEST. Compute it with:

```bash
PYTHONPATH=src python scripts/compute_normalization.py \
  --data-root "$EV_UAV_DATA_ROOT" \
  --output outputs/normalization.json
```

The normalizer records the seven input-feature statistics, split, event count,
feature channels, and `tau_scale`. This command never opens TEST.

For the released checkpoints, use the `train_normalization.json` supplied in
the Drive package unless you are reproducing training from scratch with the
same registered TRAIN data and normalization protocol.

## Train on VAL-safe development data

The trainer reads TRAIN and selects checkpoints using VAL IoU at foreground
probability `0.90`. It does not open TEST:

```bash
export EVENTREFINE_NORMALIZATION=/path/to/train_normalization.json

PYTHONPATH=src python scripts/train.py \
  --config configs/eventrefine_pair_reference.yaml \
  --data-root "$EV_UAV_DATA_ROOT" \
  --normalization "$EVENTREFINE_NORMALIZATION" \
  --output outputs/eventrefine-seed37 \
  --seed 37 \
  --device cuda:0
```

Run a one-epoch smoke test before a long campaign:

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/eventrefine_pair_reference.yaml \
  --data-root "$EV_UAV_DATA_ROOT" \
  --normalization "$EVENTREFINE_NORMALIZATION" \
  --output outputs/eventrefine-smoke \
  --seed 37 \
  --epochs 1 \
  --device cuda:0
```

Checkpoint loading is strict. Training and evaluation receipts record the
configuration, source, checkpoint, split, event counts, and metrics.

## Evaluate VAL

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eventrefine_pair_reference.yaml \
  --data-root "$EV_UAV_DATA_ROOT" \
  --normalization "$EVENTREFINE_NORMALIZATION" \
  --checkpoint outputs/eventrefine-seed37/best.pt \
  --output outputs/eventrefine-seed37/val_evaluation.json \
  --split val \
  --device cuda:0
```

## Evaluate TEST

TEST access is explicit and disabled by default. Use TEST only after freezing
the source, configuration, evaluator, normalizer, checkpoint-selection rule,
and checkpoint set:

```bash
export EVENTREFINE_OFFICIAL_EVALUATOR_ROOT=/path/to/Ev-UAV

PYTHONPATH=src python scripts/evaluate.py \
  --config configs/eventrefine_pair_reference.yaml \
  --data-root "$EV_UAV_DATA_ROOT" \
  --normalization "$EVENTREFINE_NORMALIZATION" \
  --checkpoint /path/to/eventrefine_pair_reference_k8_fullclip8s_seed37.pt \
  --output outputs/eventrefine-seed37/test_evaluation.json \
  --split test \
  --allow-test \
  --official-evaluator-root "$EVENTREFINE_OFFICIAL_EVALUATOR_ROOT" \
  --device cuda:0
```

Without `--allow-test`, the evaluator refuses to access TEST. The official
EV-UAV evaluator is external and is not vendored. Its default settings in the
public CLI are an interval of `50` ms and a correctness threshold of `0.0001`;
verify these settings against the evaluation campaign being reproduced.

The evaluator reports direct event metrics separately from official `$P_d$` and
`F_a`. `F_a \times 10^4` is the scaled value used in the result table above.

## Acknowledgment

EventRefine reads the public EV-UAV archive format. The dataset and official
evaluator are external inputs and are not redistributed in this repository.

The upstream EV-UAV repository used for format reference is:

- https://github.com/ChenYichen9527/Ev-UAV

Thank you to the authors of the original EV-UAV repository for making the
dataset format and official evaluation resources available. Please follow the
upstream repository's terms when using its dataset or evaluator. The inspected
snapshot did not contain a clearly identified `LICENSE` or `COPYING` file, so
EventRefine does not vendor upstream implementation code. If you use the
official evaluator, provide its checkout separately and follow its licensing
terms.

PyTorch, NumPy, and PyYAML are runtime dependencies installed from their
respective upstream package distributions under their own licenses.

The accompanying manuscript is **EventRefine: Pair-Reference Graph Refinement
for Tiny UAV Segmentation**.
