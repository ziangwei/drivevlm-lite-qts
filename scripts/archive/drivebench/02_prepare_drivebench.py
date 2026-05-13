from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.drivebench import iter_drivebench_json
from drivevlm_lite.data.jsonl import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path("data/drivebench"), type=Path)
    parser.add_argument("--json", default=Path("data/drivebench/text/drivebench-test.json"), type=Path)
    parser.add_argument("--image-root", default=Path("data/drivebench"), type=Path)
    parser.add_argument("--out", default=Path("data/processed/drivebench_eval.jsonl"), type=Path)
    args = parser.parse_args()

    json_path = args.json if args.json.is_absolute() else Path(args.json)
    if not json_path.exists():
        fallback = args.root / "drivebench-test.json"
        json_path = fallback if fallback.exists() else json_path

    samples = list(iter_drivebench_json(json_path, args.image_root))
    count = write_jsonl(args.out, samples)
    print(f"Wrote DriveBench eval rows: {count} -> {args.out}")


if __name__ == "__main__":
    main()
