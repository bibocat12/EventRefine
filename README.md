# EventRefine


## Repository layout

```text
src/eventrefine/       model, EV-UAV adapter, loss, runtime, metrics
configs/               fail-closed public configuration
scripts/               train, evaluate, and dataset inspection entry points
tests/                 contract and data-adapter tests
docs/                  provenance and result notes
results/               compact, machine-readable result snapshot
```

## Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
```

PyTorch must be installed for model training/evaluation. Use the PyTorch wheel appropriate for the target CPU/CUDA platform.

## EV-UAV data

Set the dataset root explicitly or pass it to each command:

```text
$EV_UAV_DATA_ROOT/
  train/train_*.npz
  val/val_*.npz
  test/test_*.npz
```

The repository does not redistribute EV-UAV data, checkpoints, normalizers, or private run logs. The archive adapter validates event alignment, monotonic timestamps, coordinates, labels, and optional structured event fields.

Inspect a split:

```bash
PYTHONPATH=src python scripts/inspect_dataset.py \
  --data-root "$EV_UAV_DATA_ROOT" --split val
```

## Train

The bundled trainer is a minimal runner; the historical paper training recipe
and full run receipts are not included.

Training reads TRAIN and selects checkpoints using VAL IoU at foreground probability 0.90. It never opens TEST:

```bash
PYTHONPATH=src python scripts/train.py \
  --data-root "$EV_UAV_DATA_ROOT" \
  --normalization "$EVENTREFINE_NORMALIZATION" \
  --output outputs/eventrefine-seed37 \
  --seed 37 \
  --device cuda:0
```

The normalizer must be fitted on TRAIN and contain seven finite mean/std values. Run a short smoke with `--epochs 1` before a full campaign.

## Evaluate VAL

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --data-root "$EV_UAV_DATA_ROOT" \
  --normalization "$EVENTREFINE_NORMALIZATION" \
  --checkpoint outputs/eventrefine-seed37/best.pt \
  --output outputs/eventrefine-seed37/val_evaluation.json \
  --split val \
  --device cuda:0
```

Checkpoint loading is strict. The output receipt records source/config/checkpoint hashes, split, counts, and metrics.

## Evaluate TEST

TEST is a separate, explicit operation:

```bash
PYTHONPATH=src python scripts/evaluate.py \
  --data-root "$EV_UAV_DATA_ROOT" \
  --normalization "$EVENTREFINE_NORMALIZATION" \
  --checkpoint outputs/eventrefine-seed37/best.pt \
  --output outputs/eventrefine-seed37/test_evaluation.json \
  --split test --allow-test \
  --official-evaluator-root /path/to/Ev-UAV \
  --device cuda:0
```

Without `--allow-test`, the evaluator refuses to access TEST. The official EV-UAV
evaluator is an external dependency and is not vendored; see `THIRD_PARTY.md`.
Install its dependencies with `pip install -e '.[evaluator]'` and provide a trusted
checkout containing `utils/eval.py`. The adapter invokes its `roc_update` and
`cal_roc` methods on reassembled full-clip predictions. Missing or incompatible
evaluator code fails explicitly; there is no substitute detection metric.

Defaults are `--official-interval-ms 50` and
`--official-correct-threshold 0.0001`. Verify these against the campaign being
reproduced. The receipt records the evaluator source hash, settings, and effective
counters. `Pd` is a percentage, `Fa` is the unscaled value, and `Fa_x1e4` is the
value for a paper column headed `Fa (10^-4)`. Without an official evaluator,
only direct event IoU/precision/recall are reported. The same adapter can be used
on VAL by adding the evaluator argument to the VAL command.

## Tests

```bash
PYTHONPATH=src pytest -q
python -m compileall -q src scripts
```

The tests include a one-epoch synthetic CPU train/validate/checkpoint-reload
regression. To additionally compare the adapter with a direct upstream call:

```bash
EVENTREFINE_OFFICIAL_EVALUATOR_ROOT=/path/to/Ev-UAV PYTHONPATH=src pytest -q
```

## Evidence

The compact result snapshot is in `results/reference_gate_weight_loss_test_summary.json`. Full provenance, evidence boundaries, and the distinction between matched TEST results and VAL diagnostics are in `docs/provenance.md` and `docs/results.md`.

## Citation

See `CITATION.cff`. The accompanying manuscript is **EventRefine: Global Reference Graph Refinement for Tiny UAV Segmentation**.

