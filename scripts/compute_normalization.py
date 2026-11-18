#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventrefine.normalization import compute_train_normalization, write_normalization


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute EventRefine normalization from TRAIN only.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = compute_train_normalization(args.data_root)
    write_normalization(args.output, payload)
    print(json.dumps({"status": "COMPLETED", "path": str(args.output.resolve()), **payload}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
