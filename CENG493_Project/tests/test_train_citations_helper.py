"""Tests for generation.train_citations — the strict citation helper used by script 10.

Covers:
  - Short answer -> no citation (MIN_SENTENCE_TOKENS guard)
  - No gold chunk available -> no citation (overlap below threshold)
  - Gold chunk present + sufficient overlap -> correct Kaynak number
  - Multi-line answer: newline structure preserved
  - Truncated/dropped statistics
  - Turkish I/i normalization
  - CitationStats accumulator
"""
import sys
from pathlib import Path

import pytest

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from generation.train_citations import (
    inject_citations_strict,
    CitationStats,
    _normalize_turkish,
    _content_tokens,
    _sentence_coverage,
    _split_sentences_per_line,
    _reassemble,
    OVERLAP_THRESHOLD,
    MIN_SENTENCE_TOKENS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunks(*texts: str, source: str = "test_law.pdf") -> list[dict]:
    return [{"text": t, "source": source, "chunk_id": f"c_{i}"}
            for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Turkish normalization
# ---------------------------------------------------------------------------

class TestTurkishNormalization:

    def test_dotted_I_lowered_correctly(self):
        """Turkish 'I' -> 'ı' (dotless), 'İ' -> 'i' (dotted)."""
        assert _normalize_turkish("İSTANBUL") == "istanbul"
        assert _normalize_turkish("ISIK") == "ısık"

    def test_mixed_case(self):
        assert _normalize_turkish("Türk Medeni KANUNU") == "türk medeni kanunu"

    def test_content_tokens_strips_stopwords(self):
        # "bu", "ile", "de" are in NLTK Turkish stopwords; "bir" is NOT
        toks = _content_tokens("Bu kanun ile duzenlenmistir de uygulanir")
        assert "bu" not in toks
        assert "ile" not in toks
        assert "de" not in toks
        assert "kanun" in toks
        assert "duzenlenmistir" in toks


# ---------------------------------------------------------------------------
# Sentence coverage (denominator = sentence tokens)
# ---------------------------------------------------------------------------

class TestSentenceCoverage:

    def test_full_coverage(self):
        s = {"hukuk", "kanun", "madde"}
        c = {"hukuk", "kanun", "madde", "extra", "tokens"}
        assert _sentence_coverage(s, c) == 1.0

    def test_partial_coverage(self):
        s = {"hukuk", "kanun", "madde", "uygulama"}
        c = {"hukuk", "kanun"}
        assert _sentence_coverage(s, c) == 0.5

    def test_no_coverage(self):
        s = {"hukuk", "kanun"}
        c = {"vergi", "tarh"}
        assert _sentence_coverage(s, c) == 0.0

    def test_empty_sentence(self):
        assert _sentence_coverage(set(), {"hukuk"}) == 0.0

    def test_denominator_is_sentence_not_min(self):
        """Critical fix: short sentence against large chunk must NOT get 1.0
        just because min(2, 100) = 2 and 2/2 = 1.0."""
        short = {"evet", "kanun"}
        big = {"evet", "kanun", "madde", "hukuk", "uygulama", "mahkeme",
               "dava", "hak", "yasa", "ceza"}
        # Old logic: min(2,10)=2, overlap=2, score=1.0
        # New logic: 2/2=1.0 — but that's actually correct when ALL sentence
        # tokens are in chunk.  The key difference is when only 1 of 2 matches:
        short2 = {"evet", "xyz"}
        assert _sentence_coverage(short2, big) == 0.5  # 1/2, not 1/1


# ---------------------------------------------------------------------------
# Short answer -> no citation
# ---------------------------------------------------------------------------

class TestShortAnswerNoCitation:

    def test_single_word_answer(self):
        """'Evet.' has < MIN_SENTENCE_TOKENS content tokens -> no citation."""
        answer = "Evet."
        chunks = _chunks("Evet, kanunla duzenlenmistir ve madde 5 uygulanir.")
        result, n_cited, n_total, n_short = inject_citations_strict(
            answer, chunks, min_tokens=MIN_SENTENCE_TOKENS,
        )
        assert "[Kaynak" not in result
        assert n_cited == 0
        assert n_short >= 1

    def test_two_word_answer(self):
        """'Madde 5.' -> 1 content token after stopword removal -> skip."""
        answer = "Bu madde."
        chunks = _chunks("Madde 5 hukuki dayanak olusturur ve uygulanir.")
        result, n_cited, _, n_short = inject_citations_strict(
            answer, chunks, min_tokens=3,
        )
        assert "[Kaynak" not in result
        assert n_short >= 1


# ---------------------------------------------------------------------------
# No overlap -> no citation (gold chunk absent from retrieval)
# ---------------------------------------------------------------------------

class TestNoOverlapNoCitation:

    def test_completely_unrelated(self):
        answer = "Vergi beyannamesi her yil verilmelidir."
        chunks = _chunks("Tapu sicili alenidir ve herkes tapu sicilini inceleyebilir.")
        result, n_cited, _, _ = inject_citations_strict(answer, chunks)
        assert result == answer
        assert n_cited == 0

    def test_below_threshold(self):
        """Slight overlap but below 0.35 threshold -> no citation."""
        # 1 out of ~5 content tokens = 0.20 < 0.35
        answer = "Tapu hukuku ozel bir alandir ve onemlidir."
        chunks = _chunks("Tapu sicili resmi kayit sistemidir ve devlet tarafindan tutulur.")
        result, n_cited, _, _ = inject_citations_strict(answer, chunks)
        assert "[Kaynak" not in result


# ---------------------------------------------------------------------------
# Gold chunk present + sufficient overlap -> correct Kaynak number
# ---------------------------------------------------------------------------

class TestCorrectKaynakNumber:

    def test_single_match_chunk2(self):
        answer = "Miras hakki kanunla duzenlenir ve miras paylari belirlenir."
        chunks = _chunks(
            "Vergi mevzuati genel esaslar ve vergi tahsili kurallari.",
            "Miras hakki ve miras paylari kanunla duzenlenir ve belirlenir.",
        )
        result, n_cited, _, _ = inject_citations_strict(answer, chunks)
        assert "[Kaynak 2]" in result
        assert "[Kaynak 1]" not in result
        assert n_cited == 1

    def test_two_sentences_two_chunks(self):
        answer = (
            "Tapu sicili alenidir ve herkes inceleyebilir. "
            "Miras hakki kanunla duzenlenir ve paylari belirlenir."
        )
        chunks = _chunks(
            "Tapu sicili alenidir ve herkes tapu sicilini inceleyebilir.",
            "Miras hakki ve miras paylari kanunla duzenlenir ve belirlenir.",
        )
        result, n_cited, _, _ = inject_citations_strict(answer, chunks)
        assert "[Kaynak 1]" in result
        assert "[Kaynak 2]" in result
        assert n_cited == 2


# ---------------------------------------------------------------------------
# Multi-line answer: newline structure preserved
# ---------------------------------------------------------------------------

class TestNewlinePreservation:

    def test_two_line_answer(self):
        answer = "Birinci satir hukuki dayanak bilgisi.\nIkinci satir kanun maddesi detayi."
        chunks = _chunks(
            "Birinci satir hukuki dayanak bilgisi ve aciklama metni.",
            "Ikinci satir kanun maddesi detayi ve uygulama alani.",
        )
        result, _, _, _ = inject_citations_strict(answer, chunks)
        lines = result.split("\n")
        assert len(lines) == 2, f"Expected 2 lines, got {len(lines)}: {result!r}"

    def test_three_line_with_empty_middle(self):
        answer = "Satir bir aciklama.\n\nSatir uc aciklama."
        chunks = _chunks("Satir bir aciklama ve hukuki dayanak bilgisi.")
        result, _, _, _ = inject_citations_strict(answer, chunks)
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[1] == ""  # empty middle line preserved

    def test_no_flattening(self):
        """Old code used ' '.join which flattened newlines. New code must not."""
        answer = "Satir A hukuki dayanak.\nSatir B kanun maddesi.\nSatir C uygulama alani."
        chunks = _chunks("Satir A hukuki dayanak ve madde bilgisi.")
        result, _, _, _ = inject_citations_strict(answer, chunks)
        assert "\n" in result


# ---------------------------------------------------------------------------
# Truncated/dropped statistics
# ---------------------------------------------------------------------------

class TestTruncationStats:

    def test_citation_stats_record(self):
        stats = CitationStats()
        stats.record(n_sentences=3, n_cited=2, n_short_skips=1,
                     has_truncation=True, has_drop=False)
        stats.record(n_sentences=2, n_cited=0, n_short_skips=0,
                     has_truncation=False, has_drop=True)
        d = stats.summary_dict()
        assert d["total_records"] == 2
        assert d["cited_records"] == 1
        assert d["uncited_records"] == 1
        assert d["total_sentences"] == 5
        assert d["cited_sentences"] == 2
        assert d["short_sentence_skips"] == 1
        assert d["truncated_records"] == 1
        assert d["dropped_records"] == 1

    def test_stats_cited_pct(self):
        stats = CitationStats()
        for _ in range(3):
            stats.record(n_sentences=1, n_cited=1, n_short_skips=0)
        for _ in range(7):
            stats.record(n_sentences=1, n_cited=0, n_short_skips=0)
        assert stats.summary_dict()["cited_pct"] == 30.0

    def test_log_summary_returns_string(self):
        stats = CitationStats()
        stats.record(n_sentences=1, n_cited=1, n_short_skips=0)
        summary = stats.log_summary()
        assert "Citation stats" in summary
        assert "1/1 cited" in summary


# ---------------------------------------------------------------------------
# Turkish I/I edge cases
# ---------------------------------------------------------------------------

class TestTurkishIEdgeCases:

    def test_uppercase_I_matches_dotless_i(self):
        """'ISIK' normalized to 'ısık'; 'Isik' also to 'ısık'."""
        answer = "ISIK kanunu onemli bir duzenlemedir ve uygulamada etkilidir."
        chunks = _chunks("Isik kanunu onemli bir duzenlemedir ve genis bir uygulama alani vardir.")
        result, n_cited, _, _ = inject_citations_strict(answer, chunks)
        assert "[Kaynak 1]" in result

    def test_dotted_I_matches_lowercase_i(self):
        """'İSTANBUL' -> 'istanbul' matches 'istanbul'."""
        answer = "Istanbul hukuk mahkemesi karari onemli bir emsal teskil etmektedir."
        chunks = _chunks("İstanbul hukuk mahkemesi karari onemli bir emsal teskil etmektedir.")
        result, n_cited, _, _ = inject_citations_strict(answer, chunks)
        assert "[Kaynak 1]" in result


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_empty_answer(self):
        result, n_cited, n_total, n_short = inject_citations_strict("", _chunks("text"))
        assert result == ""
        assert n_cited == 0

    def test_empty_chunks(self):
        result, n_cited, _, _ = inject_citations_strict("Bir cevap metni.", [])
        assert result == "Bir cevap metni."
        assert n_cited == 0

    def test_chunk_with_empty_text(self):
        result, _, _, _ = inject_citations_strict(
            "Hukuki dayanak mevcut degil ve bilgi yetersiz.",
            [{"text": "", "source": "x.pdf", "chunk_id": "c_0"}],
        )
        assert "[Kaynak" not in result


# ---------------------------------------------------------------------------
# Per-line sentence splitting internals
# ---------------------------------------------------------------------------

class TestSentenceSplitting:

    def test_simple_split(self):
        sents = _split_sentences_per_line("Birinci cumle. Ikinci cumle.")
        assert len(sents) == 2
        assert sents[0] == (0, "Birinci cumle.")
        assert sents[1] == (0, "Ikinci cumle.")

    def test_multiline(self):
        sents = _split_sentences_per_line("Satir A.\nSatir B.")
        assert sents[0][0] == 0  # line 0
        assert sents[1][0] == 1  # line 1

    def test_reassemble_preserves_lines(self):
        tagged = [(0, "A cited [Kaynak 1]"), (0, "B"), (1, "C"), (2, "D")]
        result = _reassemble(3, tagged)
        assert result == "A cited [Kaynak 1] B\nC\nD"
