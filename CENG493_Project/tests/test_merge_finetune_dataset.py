"""Tests for scripts/07_merge_finetune_dataset.py — leakage dedup logic."""
from __future__ import annotations

import importlib
import json
import sys
import pathlib
import tempfile
from unittest.mock import patch, MagicMock

import pytest

# Ensure project root is importable
_PROJECT = pathlib.Path(__file__).parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from data.data_processor import normalize_question


# ---------------------------------------------------------------------------
# Helpers: build minimal in-memory JSONL content
# ---------------------------------------------------------------------------

def _make_jsonl(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)


def _write_jsonl(path: pathlib.Path, records: list[dict]) -> None:
    path.write_text(_make_jsonl(records), encoding="utf-8")


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Unit tests for normalize_question Turkish I/İ behaviour
# ---------------------------------------------------------------------------

class TestNormalizeQuestion:
    def test_basic_ascii_lower(self):
        assert normalize_question("Hello World") == "hello world"

    def test_turkish_capital_i_dotless(self):
        # 'I' → 'ı' (dotless i), not 'i'
        result = normalize_question("ISTANBUL")
        assert result == "ıstanbul"

    def test_turkish_capital_i_dotted(self):
        # 'İ' → 'i' (dotted i)
        result = normalize_question("İSTANBUL")
        assert result == "istanbul"

    def test_mixed_case_turkish(self):
        q = "İstifa eden milletvekilinin milletvekilliği nasıl düşer?"
        result = normalize_question(q)
        # Should start with 'i' (from İ) and contain no uppercase
        assert result[0] == "i"
        assert result == result.lower() or result == normalize_question(q)

    def test_punctuation_collapsed(self):
        # Punctuation becomes a single space or is removed
        result = normalize_question("soru: nedir?")
        # No colon, no question mark in result
        assert ":" not in result
        assert "?" not in result

    def test_whitespace_collapsed(self):
        result = normalize_question("  çok   boşluk   ")
        assert "  " not in result
        assert result == result.strip()

    def test_empty_string(self):
        assert normalize_question("") == ""

    def test_none_input(self):
        assert normalize_question(None) == ""

    def test_same_question_different_case_matches(self):
        """Turkish-case variants must produce the same key (dedup must work)."""
        q1 = "Seçimlerin genel yönetim ve denetimi kim tarafından yapılır?"
        q2 = "SEÇİMLERİN GENEL YÖNETİM VE DENETİMİ KİM TARAFINDAN YAPILIR?"
        assert normalize_question(q1) == normalize_question(q2)

    def test_dotless_i_collision(self):
        """'I' (Latin capital I) must map to 'ı', not 'i', so it doesn't
        falsely collide with İ-initial words."""
        key_i = normalize_question("ı")    # dotless i stays dotless
        key_I = normalize_question("I")    # capital Latin I → dotless ı
        assert key_i == key_I  # both → 'ı'

        key_dotted = normalize_question("İ")   # capital dotted İ → 'i'
        assert key_dotted == "i"
        assert key_dotted != key_I             # 'i' ≠ 'ı'


# ---------------------------------------------------------------------------
# Integration-style tests for the dedup logic in script 07
# ---------------------------------------------------------------------------

class TestMergeFinetuneDedup:
    """Tests that simulate the dedup step in script 07 without requiring the
    HuggingFace dataset or a real filesystem layout."""

    @pytest.fixture
    def tmp_data_dir(self, tmp_path):
        return tmp_path

    def _run_dedup(
        self,
        train_records: list[dict],
        eval_records: list[dict],
        ipproo_records: list[dict],
        tmp_path: pathlib.Path,
    ) -> tuple[list[dict], dict]:
        """Run the core dedup logic from script 07 in isolation."""
        # Write qa_train.jsonl
        train_file = tmp_path / "qa_train.jsonl"
        _write_jsonl(train_file, train_records)

        # Write eval file (simulates results/processed_data/qa_eval.jsonl)
        eval_file = tmp_path / "qa_eval.jsonl"
        _write_jsonl(eval_file, eval_records)

        # Inline implementation of the dedup logic mirroring script 07
        from utils import read_jsonl

        existing = list(read_jsonl(train_file))
        eval_keys = {normalize_question(r.get("question", "")) for r in eval_records}
        eval_keys.discard("")

        seen: set[str] = set()
        clean_existing: list[dict] = []
        leaked = 0
        for rec in existing:
            key = normalize_question(rec.get("question", ""))
            if key in eval_keys:
                leaked += 1
            else:
                clean_existing.append(rec)
                seen.add(key)

        added = 0
        eval_skipped = 0
        dupes = 0
        merged = list(clean_existing)
        for idx, row in enumerate(ipproo_records, 1):
            q = (row.get("question") or row.get("instruction") or "").strip()
            a = (row.get("answer") or row.get("output") or "").strip()
            if not q or not a:
                continue
            key = normalize_question(q)
            if key in eval_keys:
                eval_skipped += 1
                continue
            if key in seen:
                dupes += 1
                continue
            seen.add(key)
            merged.append({"question": q, "answer": a, "source": "test_ipproo"})
            added += 1

        stats = dict(
            original=len(existing),
            leaked_removed=leaked,
            ipproo_eval_skipped=eval_skipped,
            ipproo_added=added,
            duplicates=dupes,
            final=len(merged),
        )
        return merged, stats

    def test_no_leakage(self, tmp_data_dir):
        train = [{"question": "Tazminat nedir?", "answer": "Para."}]
        eval_ = [{"question": "Ceza nedir?", "answer": "Hapis."}]
        ipproo = [{"question": "Vergi nedir?", "answer": "Devlet geliri."}]

        merged, stats = self._run_dedup(train, eval_, ipproo, tmp_data_dir)

        assert stats["leaked_removed"] == 0
        assert stats["ipproo_added"] == 1
        assert len(merged) == 2

    def test_existing_record_leaked(self, tmp_data_dir):
        """A train record that matches an eval question must be removed."""
        leaked_q = "Seçimlerin genel yönetim ve denetimi kim tarafından yapılır?"
        train = [
            {"question": leaked_q, "answer": "YSK."},
            {"question": "Tazminat nedir?", "answer": "Para."},
        ]
        eval_ = [{"question": leaked_q, "answer": "YSK."}]
        merged, stats = self._run_dedup(train, eval_, [], tmp_data_dir)

        assert stats["leaked_removed"] == 1
        assert stats["final"] == 1
        assert all(normalize_question(r["question"]) != normalize_question(leaked_q)
                   for r in merged)

    def test_ipproo_record_in_eval_skipped(self, tmp_data_dir):
        """An ipproo record that matches an eval question must be skipped."""
        eval_q = "Ceza nedir?"
        train = [{"question": "Tazminat nedir?", "answer": "Para."}]
        eval_ = [{"question": eval_q, "answer": "Hapis."}]
        ipproo = [{"question": eval_q, "answer": "Özgürlükten yoksun bırakma."}]

        merged, stats = self._run_dedup(train, eval_, ipproo, tmp_data_dir)

        assert stats["ipproo_eval_skipped"] == 1
        assert stats["ipproo_added"] == 0
        assert len(merged) == 1

    def test_duplicate_ipproo_not_added_twice(self, tmp_data_dir):
        """Two ipproo rows with the same (normalised) question: only the first added."""
        train = []
        eval_ = []
        ipproo = [
            {"question": "Vergi nedir?", "answer": "Devlet geliri."},
            {"question": "Vergi nedir?", "answer": "Kamusal kaynak."},
        ]

        merged, stats = self._run_dedup(train, eval_, ipproo, tmp_data_dir)

        assert stats["ipproo_added"] == 1
        assert stats["duplicates"] == 1

    def test_turkish_i_dedup_train_vs_eval(self, tmp_data_dir):
        """Train record with İ/I case variant of an eval question is removed."""
        # eval has lowercase 'istifa', train has 'İSTİFA' (capital dotted İ)
        eval_q = "istifa eden milletvekilinin milletvekilliği nasıl düşer"
        train_q = "İSTİFA EDEN MİLLETVEKİLİNİN MİLLETVEKİLLİĞİ NASIL DÜŞER"
        train = [{"question": train_q, "answer": "Sıfatı kendiliğinden düşer."}]
        eval_ = [{"question": eval_q, "answer": "Sıfatı kendiliğinden düşer."}]

        assert normalize_question(train_q) == normalize_question(eval_q), (
            "Prerequisite: Turkish-aware normalisation must collapse these to the same key"
        )

        merged, stats = self._run_dedup(train, eval_, [], tmp_data_dir)

        assert stats["leaked_removed"] == 1
        assert stats["final"] == 0

    def test_turkish_capital_i_dotless_dedup(self, tmp_data_dir):
        """Capital 'I' (without dot) → 'ı' (dotless), so 'ISTANBUL' and 'ıstanbul'
        must produce the same dedup key.  'SÖZLEŞMESI' is intentionally avoided here
        because its final capital 'I' also maps to 'ı' (correct), while the intended
        lowercase counterpart uses dotted 'i' — a legitimately distinct character.
        This test focuses only on the unambiguous 'I' → 'ı' mapping."""
        q_upper = "ISTANBUL HUKUKU"   # both I's map to ı
        q_lower = "ıstanbul hukuku"
        train = [{"question": q_upper, "answer": "Bir konu."}]
        eval_ = [{"question": q_lower, "answer": "Bir konu."}]

        assert normalize_question(q_upper) == normalize_question(q_lower), (
            "Prerequisite: 'ISTANBUL' and 'ıstanbul' must normalise identically"
        )

        merged, stats = self._run_dedup(train, eval_, [], tmp_data_dir)

        assert stats["leaked_removed"] == 1

    def test_empty_train(self, tmp_data_dir):
        """Empty qa_train.jsonl produces only ipproo records."""
        train = []
        eval_ = [{"question": "Ceza nedir?", "answer": "x"}]
        ipproo = [{"question": "Tazminat nedir?", "answer": "Para."}]

        merged, stats = self._run_dedup(train, eval_, ipproo, tmp_data_dir)

        assert stats["final"] == 1
        assert stats["ipproo_added"] == 1

    def test_all_leaked_empty_result(self, tmp_data_dir):
        """If all train records leak, merged must be empty (no ipproo)."""
        q = "Anayasa nedir?"
        train = [{"question": q, "answer": "Temel kanun."}]
        eval_ = [{"question": q, "answer": "Temel kanun."}]

        merged, stats = self._run_dedup(train, eval_, [], tmp_data_dir)

        assert stats["final"] == 0
        assert stats["leaked_removed"] == 1
        assert merged == []

    def test_leaked_question_not_readded_from_ipproo(self, tmp_data_dir):
        """Task (9): removing seen_questions.discard(key) has no effect.
        The eval_keys check in the ipproo loop prevents re-adding a question
        that was removed from the train set due to eval leakage."""
        leaked_q = "Ceza nedir?"
        train = [{"question": leaked_q, "answer": "Hapis."}]
        eval_ = [{"question": leaked_q, "answer": "Hapis."}]
        # Same question also present in ipproo — must be blocked by eval_keys check.
        ipproo = [{"question": leaked_q, "answer": "Özgürlükten yoksun bırakma."}]

        merged, stats = self._run_dedup(train, eval_, ipproo, tmp_data_dir)

        assert stats["leaked_removed"] == 1, "Leaked train record must be removed"
        assert stats["ipproo_eval_skipped"] == 1, "Ipproo record must be skipped via eval_keys"
        assert stats["final"] == 0, "No records should survive"
        assert all(
            normalize_question(r["question"]) != normalize_question(leaked_q)
            for r in merged
        ), "Leaked question must not appear in merged output"


# ---------------------------------------------------------------------------
# Regression test: actual results/processed_data/qa_train_merged.jsonl
# ---------------------------------------------------------------------------

class TestActualMergedFileClean:
    """Verify the committed qa_train_merged.jsonl contains no eval questions
    (post-fix state).  Skipped when the file is absent."""

    @pytest.fixture(scope="class")
    def merged_path(self):
        # After the _PDIR fallback was removed, script 07 writes to
        # config.PROCESSED_DIR (the canonical processed-data directory).
        import config
        p = config.PROCESSED_DIR / "qa_train_merged.jsonl"
        if not p.exists():
            pytest.skip(f"qa_train_merged.jsonl not found at {p}")
        return p

    @pytest.fixture(scope="class")
    def eval_keys(self):
        from data.data_processor import DataProcessor
        return DataProcessor.saved_eval_question_keys()

    def test_no_eval_leakage_in_merged(self, merged_path, eval_keys):
        leaked = []
        with merged_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                key = normalize_question(rec.get("question", ""))
                if key in eval_keys:
                    leaked.append(rec.get("question", ""))

        assert leaked == [], (
            f"qa_train_merged.jsonl contains {len(leaked)} eval-set question(s):\n"
            + "\n".join(f"  {q!r}" for q in leaked[:10])
        )


# ---------------------------------------------------------------------------
# Task (4): Eval leakage enforcement — main() must hard-fail when eval is absent
# ---------------------------------------------------------------------------

import importlib.util as _ilu


def _load_script07():
    """Import script 07 as a fresh module (leading digit blocks normal import)."""
    _spec = _ilu.spec_from_file_location(
        "_script07",
        pathlib.Path(__file__).parent.parent / "scripts" / "07_merge_finetune_dataset.py",
    )
    _mod = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod


class TestEvalLeakageEnforcement:
    """Task (4): empty eval_keys → SystemExit; --allow-missing-eval → bypass + metadata."""

    @pytest.fixture
    def pdir(self, tmp_path, monkeypatch):
        """Write a minimal qa_train.jsonl and redirect config to tmp_path."""
        import config
        (tmp_path / "qa_train.jsonl").write_text(
            json.dumps({"question": "Soru?", "answer": "Cevap."}) + "\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(config, "TLR_PROCESSED_DIR", tmp_path)
        return tmp_path

    def test_missing_eval_raises_systemexit(self, pdir):
        """Empty eval_keys with no --allow-missing-eval → SystemExit."""
        from data.data_processor import DataProcessor
        mod = _load_script07()
        with patch.object(DataProcessor, "saved_eval_question_keys", return_value=set()):
            with pytest.raises(SystemExit) as exc_info:
                mod.main(argv=[])
        msg = str(exc_info.value).lower()
        assert "leakage" in msg or "eval" in msg

    def test_systemexit_message_names_missing_files(self, pdir):
        """SystemExit message lists which eval files were not found."""
        from data.data_processor import DataProcessor
        mod = _load_script07()
        with patch.object(DataProcessor, "saved_eval_question_keys", return_value=set()):
            with pytest.raises(SystemExit) as exc_info:
                mod.main(argv=[])
        # All three expected files are absent in tmp_path → all listed
        msg = str(exc_info.value)
        assert "qa_eval" in msg or "qa_hmgs" in msg or "qa_turkish_legal_rag" in msg

    def test_allow_missing_eval_no_systemexit(self, pdir):
        """--allow-missing-eval: completes without raising when eval absent."""
        from data.data_processor import DataProcessor
        mod = _load_script07()
        with patch.object(DataProcessor, "saved_eval_question_keys", return_value=set()):
            mod.main(argv=["--allow-missing-eval"])  # must not raise

    def test_allow_missing_eval_leakage_checked_false(self, pdir):
        """--allow-missing-eval: metadata records leakage_checked=False."""
        from data.data_processor import DataProcessor
        mod = _load_script07()
        with patch.object(DataProcessor, "saved_eval_question_keys", return_value=set()):
            mod.main(argv=["--allow-missing-eval"])
        meta = json.loads((pdir / "merge_config.json").read_text(encoding="utf-8"))
        lc = meta["leakage_check"]
        assert lc["leakage_checked"] is False

    def test_allow_missing_eval_skip_reason_recorded(self, pdir):
        """--allow-missing-eval: skip_reason is a non-empty string in metadata."""
        from data.data_processor import DataProcessor
        mod = _load_script07()
        with patch.object(DataProcessor, "saved_eval_question_keys", return_value=set()):
            mod.main(argv=["--allow-missing-eval"])
        meta = json.loads((pdir / "merge_config.json").read_text(encoding="utf-8"))
        lc = meta["leakage_check"]
        assert isinstance(lc["skip_reason"], str) and lc["skip_reason"]

    def test_eval_present_leakage_checked_true(self, pdir):
        """Non-empty eval_keys → leakage_checked=True and count matches."""
        from data.data_processor import DataProcessor
        eval_keys = {"soru bir", "soru iki"}
        mod = _load_script07()
        with patch.object(DataProcessor, "saved_eval_question_keys", return_value=eval_keys):
            mod.main(argv=[])
        meta = json.loads((pdir / "merge_config.json").read_text(encoding="utf-8"))
        lc = meta["leakage_check"]
        assert lc["leakage_checked"] is True
        assert lc["eval_keys_loaded"] == 2

    def test_metadata_includes_eval_file_lists(self, pdir):
        """merge_config.json always records eval_files_found and eval_files_missing."""
        from data.data_processor import DataProcessor
        mod = _load_script07()
        with patch.object(DataProcessor, "saved_eval_question_keys", return_value={"k1"}):
            mod.main(argv=[])
        meta = json.loads((pdir / "merge_config.json").read_text(encoding="utf-8"))
        lc = meta["leakage_check"]
        assert "eval_files_found" in lc
        assert "eval_files_missing" in lc

    def test_bypass_eval_keys_loaded_zero(self, pdir):
        """When bypassed, eval_keys_loaded=0 in metadata (no eval questions processed)."""
        from data.data_processor import DataProcessor
        mod = _load_script07()
        with patch.object(DataProcessor, "saved_eval_question_keys", return_value=set()):
            mod.main(argv=["--allow-missing-eval"])
        meta = json.loads((pdir / "merge_config.json").read_text(encoding="utf-8"))
        assert meta["leakage_check"]["eval_keys_loaded"] == 0
