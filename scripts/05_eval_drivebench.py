from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=Path("configs/eval/drivebench.yaml"), type=Path)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--out", default=Path("reports/drivebench"), type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    print("DriveBench eval entry point is scaffolded.")
    print("Next implementation step: load processed eval JSONL, run Qwen3-VL, compute metrics.")
    print(f"limit={args.limit}, out={args.out}")
    print(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
