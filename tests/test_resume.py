"""Tests for the crash-resume helpers in ``drivevlm_lite.eval.resume``.

Run without pytest::

    PYTHONPATH=src python tests/test_resume.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drivevlm_lite.eval.resume import (
    check_meta_compatible,
    read_jsonl_robust,
    truncate_to_last_newline,
)


def test_truncate_is_noop_when_file_ends_on_newline() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        p.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
        removed = truncate_to_last_newline(p)
        assert removed == 0
        assert p.read_text(encoding="utf-8") == '{"a":1}\n{"b":2}\n'


def test_truncate_removes_partial_trailing_line() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        p.write_text('{"a":1}\n{"b":2}\n{"c":', encoding="utf-8")
        removed = truncate_to_last_newline(p)
        assert removed == len('{"c":')
        assert p.read_text(encoding="utf-8") == '{"a":1}\n{"b":2}\n'


def test_truncate_clears_file_with_no_newlines() -> None:
    """If the entire file is one partial line, the only safe move is to wipe."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        p.write_text('{"partial', encoding="utf-8")
        removed = truncate_to_last_newline(p)
        assert removed == len('{"partial')
        assert p.read_text(encoding="utf-8") == ""


def test_truncate_handles_missing_and_empty_files() -> None:
    with tempfile.TemporaryDirectory() as d:
        missing = Path(d) / "missing.jsonl"
        assert truncate_to_last_newline(missing) == 0
        empty = Path(d) / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        assert truncate_to_last_newline(empty) == 0


def test_truncate_is_idempotent() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        p.write_text('{"a":1}\n{"b":2', encoding="utf-8")
        first = truncate_to_last_newline(p)
        second = truncate_to_last_newline(p)
        assert first > 0 and second == 0
        assert p.read_text(encoding="utf-8") == '{"a":1}\n'


def test_read_jsonl_robust_returns_rows_and_zero_skipped_on_clean_file() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        p.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
        rows, bad = read_jsonl_robust(p)
        assert rows == [{"a": 1}, {"b": 2}]
        assert bad == 0


def test_read_jsonl_robust_skips_malformed_lines() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.jsonl"
        p.write_text('{"a":1}\n{not json}\n   \n{"b":2}\n', encoding="utf-8")
        rows, bad = read_jsonl_robust(p)
        assert rows == [{"a": 1}, {"b": 2}]
        assert bad == 1


def test_read_jsonl_robust_handles_missing_and_empty() -> None:
    with tempfile.TemporaryDirectory() as d:
        missing = Path(d) / "missing.jsonl"
        assert read_jsonl_robust(missing) == ([], 0)
        empty = Path(d) / "empty.jsonl"
        empty.write_text("", encoding="utf-8")
        assert read_jsonl_robust(empty) == ([], 0)


def test_check_meta_returns_none_when_meta_missing() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "missing.json"
        assert check_meta_compatible(p, {"seed": 42}) is None


def test_check_meta_returns_none_when_meta_matches() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        p.write_text('{"seed":42,"ablation":"full"}', encoding="utf-8")
        assert check_meta_compatible(p, {"seed": 42, "ablation": "full"}) is None


def test_check_meta_returns_diff_on_mismatch() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        p.write_text('{"seed":42,"ablation":"full"}', encoding="utf-8")
        msg = check_meta_compatible(p, {"seed": 1, "ablation": "full"})
        assert msg is not None
        assert "seed" in msg
        assert "42" in msg and "1" in msg


def test_check_meta_ignores_extra_keys_in_saved_meta() -> None:
    """Schema can grow; saved meta with extra fields still resumes cleanly."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        p.write_text('{"seed":42,"ablation":"full","legacy_key":"x"}', encoding="utf-8")
        assert check_meta_compatible(p, {"seed": 42, "ablation": "full"}) is None


def test_check_meta_flags_malformed_json() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.json"
        p.write_text("{not json", encoding="utf-8")
        msg = check_meta_compatible(p, {"seed": 42})
        assert msg is not None
        assert "malformed" in msg


def _run_all_tests() -> int:
    test_fns = [
        test_truncate_is_noop_when_file_ends_on_newline,
        test_truncate_removes_partial_trailing_line,
        test_truncate_clears_file_with_no_newlines,
        test_truncate_handles_missing_and_empty_files,
        test_truncate_is_idempotent,
        test_read_jsonl_robust_returns_rows_and_zero_skipped_on_clean_file,
        test_read_jsonl_robust_skips_malformed_lines,
        test_read_jsonl_robust_handles_missing_and_empty,
        test_check_meta_returns_none_when_meta_missing,
        test_check_meta_returns_none_when_meta_matches,
        test_check_meta_returns_diff_on_mismatch,
        test_check_meta_ignores_extra_keys_in_saved_meta,
        test_check_meta_flags_malformed_json,
    ]
    failures = 0
    for fn in test_fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"\n{failures} test(s) failed")
    else:
        print(f"\nAll {len(test_fns)} tests passed.")
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all_tests() else 0)
