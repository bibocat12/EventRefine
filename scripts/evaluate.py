#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventrefine.evaluation import evaluate


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate EventRefine on VAL or explicitly authorized TEST.")
    parser.add_argument("--config", type=Path, default=Path("configs/eventrefine_pair_reference.yaml"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--official-evaluator-root", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = evaluate(config_path=args.config, data_root=args.data_root, normalization_path=args.normalization, checkpoint=args.checkpoint, output=args.output, split=args.split, device=args.device, allow_test=args.allow_test, official_evaluator_root=args.official_evaluator_root)
    print(json.dumps({key: result[key] for key in ("status", "split", "uses_test", "clips", "events", "metrics")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
