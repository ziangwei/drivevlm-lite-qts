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
    parser.add_argument("--zip-condition", default=None)
    parser.add_argument("--limit", default=20, type=int)
    parser.add_argument("--show-prefixes", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit > 0:
        rows = rows[: args.limit]

    checked = 0
    missing: list[str] = []
    ambiguous: list[str] = []
    with ImageLoader(args.image_zip, zip_condition=args.zip_condition) as loader:
        if args.show_prefixes:
            print("zip_prefixes:")
            for prefix, count in loader.top_prefixes(depth=2, limit=50):
                print(f"  {prefix}: {count}")
        for row in rows:
            for path in row.get("images", []):
                checked += 1
                try:
                    loader.resolve(path)
                except FileNotFoundError as exc:
                    candidates = loader.candidates(path)
                    if candidates:
                        ambiguous.append(str(path))
                    else:
                        missing.append(str(path))
                    if len(missing) + len(ambiguous) <= 5:
                        if candidates:
                            print(f"CANDIDATES for {path}:")
                            for candidate in candidates[:20]:
                                print(f"  {candidate}")
                        else:
                            print(f"NO_CANDIDATES for {path}")
                        print(f"ERROR {exc}")

    print(f"checked_rows={len(rows)}")
    print(f"checked_images={checked}")
    print(f"ambiguous_images={len(ambiguous)}")
    print(f"missing_images={len(missing)}")
    for path in ambiguous[:20]:
        print(f"AMBIGUOUS {path}")
    for path in missing[:20]:
        print(f"MISSING {path}")
    if missing or ambiguous:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
