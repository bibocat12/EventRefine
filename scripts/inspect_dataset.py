#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eventrefine.dataset import EvUAV


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect one EV-UAV split without touching other splits.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    args = parser.parse_args()
    dataset = EvUAV(args.data_root, split=args.split)
    events = sum(len(dataset[index]["ts"]) for index in range(len(dataset)))
    print(json.dumps({"split": args.split, "clips": len(dataset), "events": events, "uses_test": args.split == "test"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
