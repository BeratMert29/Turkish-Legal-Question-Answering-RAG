"""Tests for citation injection in RAG training data (script 10 / utils.inject_citations).

Validates:
  1. [Kaynak N] markers are appended to matching answer sentences.
  2. Kaynak numbers correspond to chunk position (1-indexed).
  3. Answers with no chunk overlap remain unchanged.
  4. Empty chunks / empty answer edge cases.
"""
import sys
from pathlib import Path

import pytest

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from utils import inject_citations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(*texts: str, source: str = "test_law.pdf") -> list[dict]:
    """Build a minimal chunk list suitable for inject_citations."""
    return [{"text": t, "source": source, "chunk_id": f"c_{i}"}
            for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestCitationEdgeCases:

    def test_answer_single_word(self):
        """Very short answer should not crash."""
        answer = "Evet."
        chunks = _chunks("Evet, kanunla duzenlenmistir.")
        result = inject_citations(answer, chunks)
        # May or may not add citation depending on threshold; must not crash
        assert isinstance(result, str)

    def test_turkish_characters_handled(self):
        """Turkish-specific characters (i/I, dotted/dotless) should not break matching."""
        answer = "Turk Medeni Kanunu Madde 997 uygulanir."
        chunks = _chunks("TURK MEDENI KANUNU MADDE 997 uygulanir.")
        result = inject_citations(answer, chunks)
        # Turkish normalization should allow match
        assert "[Kaynak 1]" in result

    def test_multiple_chunks_same_sentence(self):
        """When multiple chunks overlap with the same sentence, all should cite."""
        answer = "Tapu sicili herkes tarafindan incelenebilir ve alenidir."
        chunks = _chunks(
            "Tapu sicili alenidir ve herkes tapu sicilini inceleyebilir.",
            "Tapu sicili herkes tarafindan incelenebilir resmi kayittir.",
        )
        result = inject_citations(answer, chunks)
        # At least one citation present
        assert "[Kaynak" in result


# ---------------------------------------------------------------------------
# Integration-like: simulate script 10 flow
# ---------------------------------------------------------------------------

class TestScript10Flow:
    """Simulate the data flow in 10_build_rag_train_data after the change."""

    def test_cited_answer_in_output_record(self):
        """After inject_citations, the output record must contain cited text."""
        raw_answer = "Tapu sicili alenidir."
        included_chunks = _chunks("Tapu sicili alenidir ve herkes tapu sicilini inceleyebilir.")
        cited = inject_citations(raw_answer, included_chunks)
        out_record = {
            "question": "Tapu sicili nedir?",
            "answer": cited,
            "context_str": "[Kaynak 1] (test)\nTapu sicili alenidir.\n\n",
        }
        assert "[Kaynak 1]" in out_record["answer"]

    def test_context_numbering_consistency(self):
        """Kaynak numbers in the answer must match [Kaynak N] blocks in context_str."""
        from generation.rag_pipeline import format_source

        chunks = _chunks(
            "Vergi hukuku genel esaslar ve verginin tarhi.",
            "Miras hukuku ve miras paylasimi kurallari.",
        )
        # Build context_str the same way assemble_context does
        context_parts = [format_source(i + 1, c["source"], c["text"]) for i, c in enumerate(chunks)]
        context_str = "".join(context_parts)

        answer = "Miras paylasimi kurallari onemlidir."
        cited = inject_citations(answer, chunks)

        # The citation number in the answer should appear in context_str too
        import re
        cited_nums = {int(m) for m in re.findall(r"\[Kaynak (\d+)\]", cited)}
        context_nums = {int(m) for m in re.findall(r"\[Kaynak (\d+)\]", context_str)}
        assert cited_nums.issubset(context_nums), (
            f"Answer cites {cited_nums} but context only has {context_nums}"
        )
