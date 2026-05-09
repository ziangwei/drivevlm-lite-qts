from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.data.image_loading import ImageLoader
from drivevlm_lite.data.jsonl import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--image-zip", required=True, type=Path)
    parser.add_argument("--limit", default=20, type=int)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    checked = 0
    missing: list[str] = []
    with ImageLoader(args.image_zip) as loader:
        for row in rows:
            for path in row.get("images", []):
                checked += 1
                try:
                    loader.resolve(path)
                except FileNotFoundError:
                    missing.append(str(path))

    print(f"checked_rows={len(rows)}")
    print(f"checked_images={checked}")
    print(f"missing_images={len(missing)}")
    for path in missing[:20]:
        print(f"MISSING {path}")
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
