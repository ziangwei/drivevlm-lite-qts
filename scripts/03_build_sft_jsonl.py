from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or merge SFT JSONL files.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    valid = [row for row in rows if row.get("images") and row.get("messages")]
    count = write_jsonl(args.output, valid)
    print(f"Validated SFT rows: {count}/{len(rows)} -> {args.output}")


if __name__ == "__main__":
    main()
