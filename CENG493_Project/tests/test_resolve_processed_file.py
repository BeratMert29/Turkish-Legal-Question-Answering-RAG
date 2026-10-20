"""Tests for utils.resolve_processed_file — task (3)."""
from __future__ import annotations

import sys
import pathlib

import pytest

_PROJECT = pathlib.Path(__file__).parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import config
from utils import resolve_processed_file


class TestResolveProcessedFile:
    # ------------------------------------------------------------------
    # Return value
    # ------------------------------------------------------------------

    def test_returns_processed_dir_slash_name(self, tmp_path, monkeypatch):
        """Returned path is exactly config.PROCESSED_DIR / name."""
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        (tmp_path / "qa_train.jsonl").write_text("", encoding="utf-8")
        result = resolve_processed_file("qa_train.jsonl")
        assert result == tmp_path / "qa_train.jsonl"

    def test_returns_path_object(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        (tmp_path / "qa_eval.jsonl").write_text("", encoding="utf-8")
        result = resolve_processed_file("qa_eval.jsonl")
        assert isinstance(result, pathlib.Path)

    # ------------------------------------------------------------------
    # required=True (default): file must exist
    # ------------------------------------------------------------------

    def test_required_true_existing_file_no_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        (tmp_path / "qa_train.jsonl").write_text("x", encoding="utf-8")
        path = resolve_processed_file("qa_train.jsonl", required=True)
        assert path.exists()

    def test_required_default_is_true(self, tmp_path, monkeypatch):
        """Omitting required= raises on missing file (default is True)."""
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            resolve_processed_file("missing.jsonl")

    def test_required_true_missing_raises_file_not_found(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            resolve_processed_file("missing_file.jsonl", required=True)

    def test_error_message_includes_missing_path(self, tmp_path, monkeypatch):
        """FileNotFoundError message names the exact missing path."""
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_processed_file("some_file.jsonl", required=True)
        assert str(tmp_path / "some_file.jsonl") in str(exc_info.value)

    def test_error_message_mentions_upstream_script(self, tmp_path, monkeypatch):
        """Error message advises the user to run an upstream script."""
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_processed_file("qa_train.jsonl", required=True)
        msg = str(exc_info.value).lower()
        assert "script" in msg or "run" in msg or "upstream" in msg

    # ------------------------------------------------------------------
    # required=False: path returned regardless of existence
    # ------------------------------------------------------------------

    def test_required_false_missing_returns_path_no_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        path = resolve_processed_file("nonexistent.jsonl", required=False)
        assert path == tmp_path / "nonexistent.jsonl"
        assert not path.exists()

    def test_required_false_existing_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        (tmp_path / "qa_eval.jsonl").write_text("", encoding="utf-8")
        path = resolve_processed_file("qa_eval.jsonl", required=False)
        assert path == tmp_path / "qa_eval.jsonl"

    # ------------------------------------------------------------------
    # No silent fallback to any other directory
    # ------------------------------------------------------------------

    def test_no_silent_fallback_to_other_dir(self, tmp_path, monkeypatch):
        """File present in a sibling dir but NOT in PROCESSED_DIR must raise."""
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        (other_dir / "qa_train.jsonl").write_text("", encoding="utf-8")
        # PROCESSED_DIR is tmp_path — file only exists in other_dir
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            resolve_processed_file("qa_train.jsonl", required=True)

    def test_uses_config_processed_dir_not_tlr(self, tmp_path, monkeypatch):
        """resolve_processed_file always uses config.PROCESSED_DIR, never TLR_PROCESSED_DIR."""
        tlr_dir = tmp_path / "tlr"
        tlr_dir.mkdir()
        pdir = tmp_path / "pdir"
        pdir.mkdir()
        (tlr_dir / "qa_train.jsonl").write_text("", encoding="utf-8")
        # File exists in tlr_dir but NOT in pdir (PROCESSED_DIR)
        monkeypatch.setattr(config, "PROCESSED_DIR", pdir)
        monkeypatch.setattr(config, "TLR_PROCESSED_DIR", tlr_dir)
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_processed_file("qa_train.jsonl", required=True)
        # Error must mention pdir, not tlr_dir
        assert str(pdir) in str(exc_info.value)
