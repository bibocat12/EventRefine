from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventrefine.training import train_one


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the fixed EventRefine model.")
    parser.add_argument("--config", type=Path, default=Path("configs/eventrefine_pair_reference.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=(37, 1337, 2024))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = train_one(config_path=args.config, data_root=args.data_root, normalization_path=args.normalization, output_root=args.output, seed=args.seed, epochs=args.epochs, device=args.device)
    print(json.dumps({key: summary[key] for key in ("status", "arm", "seed", "best_epoch", "best_val_iou_09", "split", "uses_test")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
