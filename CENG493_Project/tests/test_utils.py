"""Tests for utils.read_jsonl."""

from __future__ import annotations

import pytest
from pathlib import Path

from utils import read_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, lines: list[str], name: str = "data.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestReadJsonlHappy:
    def test_basic_yield(self, tmp_path):
        p = _write(tmp_path, ['{"a": 1}', '{"b": 2}'])
        result = list(read_jsonl(p))
        assert result == [{"a": 1}, {"b": 2}]

    def test_blank_lines_skipped(self, tmp_path):
        p = _write(tmp_path, ['{"a": 1}', "", "   ", '{"b": 2}'])
        result = list(read_jsonl(p))
        assert len(result) == 2

    def test_empty_file(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert list(read_jsonl(p)) == []

    def test_single_record(self, tmp_path):
        p = _write(tmp_path, ['{"key": "value"}'])
        assert list(read_jsonl(p)) == [{"key": "value"}]

    def test_generator_is_lazy(self, tmp_path):
        """read_jsonl returns a generator, not a list."""
        import types
        p = _write(tmp_path, ['{"x": 1}'])
        gen = read_jsonl(p)
        assert isinstance(gen, types.GeneratorType)

    def test_path_as_string(self, tmp_path):
        p = _write(tmp_path, ['{"ok": true}'])
        result = list(read_jsonl(str(p)))
        assert result == [{"ok": True}]


# ---------------------------------------------------------------------------
# Error-handling: on_error="warn" (default)
# ---------------------------------------------------------------------------

class TestReadJsonlWarn:
    def test_bad_line_skipped_by_default(self, tmp_path):
        p = _write(tmp_path, ['{"good": 1}', "NOT JSON", '{"good": 2}'])
        result = list(read_jsonl(p))
        assert result == [{"good": 1}, {"good": 2}]

    def test_warning_emitted_with_filename_and_lineno(self, tmp_path, caplog):
        import logging
        p = _write(tmp_path, ['{"ok": 1}', "BROKEN", '{"ok": 2}'])
        with caplog.at_level(logging.WARNING, logger="utils"):
            list(read_jsonl(p))
        assert any(
            str(p) in rec.message and "2" in rec.message
            for rec in caplog.records
        ), f"Expected warning with filename+lineno=2, got: {[r.message for r in caplog.records]}"

    def test_all_bad_lines_skipped(self, tmp_path):
        p = _write(tmp_path, ["not json", "also not json"])
        assert list(read_jsonl(p)) == []


# ---------------------------------------------------------------------------
# Error-handling: on_error="raise"
# ---------------------------------------------------------------------------

class TestReadJsonlRaise:
    def test_raises_value_error_on_bad_line(self, tmp_path):
        p = _write(tmp_path, ['{"ok": 1}', "BAD", '{"ok": 2}'])
        with pytest.raises(ValueError) as exc_info:
            list(read_jsonl(p, on_error="raise"))
        msg = str(exc_info.value)
        assert str(p) in msg, "Error message must include the file path"
        assert "2" in msg, "Error message must include the 1-based line number"

    def test_raises_on_first_bad_line(self, tmp_path):
        p = _write(tmp_path, ["BAD1", "BAD2"])
        with pytest.raises(ValueError) as exc_info:
            list(read_jsonl(p, on_error="raise"))
        assert "1" in str(exc_info.value)

    def test_no_raise_when_all_valid(self, tmp_path):
        p = _write(tmp_path, ['{"a": 1}', '{"b": 2}'])
        result = list(read_jsonl(p, on_error="raise"))
        assert len(result) == 2
