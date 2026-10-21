"""Tests for citation injection in RAG training data.

Part 1: utils.inject_citations (inference-time, must remain unchanged).
Part 2: script 10 flow with generation.train_citations.inject_citations_strict
        and generation.rag_pipeline.assemble_context_ex.
"""
import re
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import inject_citations
from generation.train_citations import inject_citations_strict, CitationStats
from generation.rag_pipeline import format_source, AssembleResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(*texts: str, source: str = "test_law.pdf") -> list[dict]:
    """Build a minimal chunk list suitable for inject_citations."""
    return [{"text": t, "source": source, "chunk_id": f"c_{i}"}
            for i, t in enumerate(texts)]


# ===========================================================================
# Part 1 — utils.inject_citations (inference-time, unchanged)
# ===========================================================================

class TestCitationInjection:
    """inject_citations adds [Kaynak N] markers via token overlap."""

    def test_single_sentence_single_chunk_match(self):
        answer = "Tapu sicili alenidir."
        chunks = _chunks("Tapu sicili alenidir ve herkes tapu sicilini inceleyebilir.")
        result = inject_citations(answer, chunks)
        assert "[Kaynak 1]" in result

    def test_kaynak_number_matches_chunk_position(self):
        """Kaynak N must equal the 1-based position of the matched chunk."""
        answer = "Miras hakki kanunla duzenlenir."
        chunks = _chunks(
            "Vergi mevzuati genel esaslar.",                    # chunk 1 -- unrelated
            "Miras hakki ve miras paylari kanunla duzenlenir.",  # chunk 2 -- match
        )
        result = inject_citations(answer, chunks)
        assert "[Kaynak 2]" in result
        assert "[Kaynak 1]" not in result

    def test_multiple_sentences_different_chunks(self):
        answer = (
            "Tapu sicili alenidir. "
            "Miras hakki kanunla duzenlenir."
        )
        chunks = _chunks(
            "Tapu sicili alenidir ve herkes bunu inceleyebilir.",
            "Miras hakki ve miras paylari kanunla duzenlenir.",
        )
        result = inject_citations(answer, chunks)
        assert "[Kaynak 1]" in result
        assert "[Kaynak 2]" in result

    def test_no_overlap_returns_unchanged(self):
        answer = "Bu tamamen farkli bir konu hakkindadir."
        chunks = _chunks("Tapu sicili alenidir ve herkes tapu sicilini inceleyebilir.")
        result = inject_citations(answer, chunks)
        assert result == answer
        assert "[Kaynak" not in result

    def test_empty_chunks_returns_unchanged(self):
        answer = "Bir cevap metni."
        result = inject_citations(answer, [])
        assert result == answer

    def test_empty_answer_returns_unchanged(self):
        chunks = _chunks("Tapu sicili alenidir.")
        result = inject_citations("", chunks)
        assert result == ""


class TestCitationEdgeCases:

    def test_answer_single_word(self):
        """Very short answer should not crash."""
        answer = "Evet."
        chunks = _chunks("Evet, kanunla duzenlenmistir.")
        result = inject_citations(answer, chunks)
        assert isinstance(result, str)

    def test_turkish_characters_handled(self):
        answer = "Turk Medeni Kanunu Madde 997 uygulanir."
        chunks = _chunks("TURK MEDENI KANUNU MADDE 997 uygulanir.")
        result = inject_citations(answer, chunks)
        assert "[Kaynak 1]" in result

    def test_multiple_chunks_same_sentence(self):
        answer = "Tapu sicili herkes tarafindan incelenebilir ve alenidir."
        chunks = _chunks(
            "Tapu sicili alenidir ve herkes tapu sicilini inceleyebilir.",
            "Tapu sicili herkes tarafindan incelenebilir resmi kayittir.",
        )
        result = inject_citations(answer, chunks)
        assert "[Kaynak" in result


# ===========================================================================
# Part 2 — script 10 flow (inject_citations_strict + assemble_context_ex)
# ===========================================================================

class TestScript10FlowStrict:
    """Simulate the data flow in 10_build_rag_train_data with the new logic."""

    def test_cited_answer_in_output_record(self):
        raw_answer = "Tapu sicili alenidir ve herkes tapu sicilini inceleyebilir."
        included = _chunks("Tapu sicili alenidir ve herkes tapu sicilini inceleyebilir.")
        cited, n_cited, _, _ = inject_citations_strict(raw_answer, included)
        out_record = {
            "question": "Tapu sicili nedir?",
            "answer": cited,
            "context_str": "[Kaynak 1] (test)\nTapu sicili alenidir.\n\n",
        }
        assert "[Kaynak 1]" in out_record["answer"]
        assert n_cited >= 1

    def test_context_numbering_consistency(self):
        """Kaynak numbers in the answer must match [Kaynak N] blocks in context."""
        chunks = _chunks(
            "Vergi hukuku genel esaslar ve verginin tarhi usulleri.",
            "Miras hukuku miras paylasimi kurallari kanunla duzenlenir.",
        )
        context_parts = [format_source(i + 1, c["source"], c["text"])
                         for i, c in enumerate(chunks)]
        context_str = "".join(context_parts)

        answer = "Miras paylasimi kurallari kanunla duzenlenir ve onemlidir."
        cited, _, _, _ = inject_citations_strict(answer, chunks)

        cited_nums = {int(m) for m in re.findall(r"\[Kaynak (\d+)\]", cited)}
        context_nums = {int(m) for m in re.findall(r"\[Kaynak (\d+)\]", context_str)}
        assert cited_nums.issubset(context_nums), (
            f"Answer cites {cited_nums} but context only has {context_nums}"
        )

    def test_assemble_result_fields(self):
        """AssembleResult dataclass has all expected fields."""
        r = AssembleResult(
            context="ctx", included=[{"text": "a"}],
            truncated_ids=["c_0"], dropped_ids=["c_1"], hit_limit=True,
        )
        assert r.hit_limit is True
        assert r.truncated_ids == ["c_0"]
        assert r.dropped_ids == ["c_1"]

    def test_strict_rejects_short_answer(self):
        """Short answer that old inject_citations might cite should be skipped."""
        answer = "Evet."
        chunks = _chunks("Evet, kanunla duzenlenmistir ve madde uygulanir.")
        cited, n_cited, _, n_short = inject_citations_strict(answer, chunks)
        assert "[Kaynak" not in cited
        assert n_short >= 1

    def test_strict_preserves_newlines(self):
        """Multi-line answer must keep its \\n structure."""
        answer = "Satir bir hukuki dayanak.\nSatir iki kanun maddesi detayi."
        chunks = _chunks(
            "Satir bir hukuki dayanak ve aciklama bilgisi.",
            "Satir iki kanun maddesi detayi ve uygulama alani.",
        )
        cited, _, _, _ = inject_citations_strict(answer, chunks)
        assert "\n" in cited
        assert len(cited.split("\n")) == 2

    def test_stats_integration(self):
        """CitationStats accumulates correctly over multiple records."""
        stats = CitationStats()
        # Record 1: cited
        _, n1, t1, s1 = inject_citations_strict(
            "Miras hakki kanunla duzenlenir ve miras paylari belirlenir.",
            _chunks("Miras hakki ve miras paylari kanunla duzenlenir ve belirlenir."),
        )
        stats.record(t1, n1, s1, has_truncation=True, has_drop=False)
        # Record 2: uncited (no overlap)
        _, n2, t2, s2 = inject_citations_strict(
            "Vergi beyannamesi her yil hazirlanir ve verilir.",
            _chunks("Tapu sicili alenidir ve herkes inceleyebilir."),
        )
        stats.record(t2, n2, s2, has_truncation=False, has_drop=True)

        d = stats.summary_dict()
        assert d["total_records"] == 2
        assert d["cited_records"] == 1
        assert d["truncated_records"] == 1
        assert d["dropped_records"] == 1
