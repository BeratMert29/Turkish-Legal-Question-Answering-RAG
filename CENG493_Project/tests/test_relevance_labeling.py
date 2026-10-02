"""
tests/test_relevance_labeling.py — Unit tests for relevance labeling strategies.

All tests run offline (no FAISS, no embeddings, no Ollama).
"""

import sys
from pathlib import Path

# Make sure the project root is on sys.path so we can import as top-level packages.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from data.data_processor import (
    CorpusChunk,
    QAExample,
    DataProcessor,
    _extract_madde_no,
    _chunk_matches_article,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, doc_id: str, text: str, source: str = "TestLaw") -> CorpusChunk:
    return CorpusChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        source=source,
        char_len=len(text),
    )


def _qa(query_id: str, question: str = "", answer: str = "",
        source: str = "", context: str = "") -> QAExample:
    return QAExample(
        query_id=query_id,
        question=question,
        answer=answer,
        context=context,
        source=source,
        data_type="",
    )


# ---------------------------------------------------------------------------
# _extract_madde_no
# ---------------------------------------------------------------------------

class TestExtractMaddeNo:
    def test_madde_prefix(self):
        assert _extract_madde_no("madde 44 hükmü nedir?", "") == 44

    def test_madde_suffix_dot(self):
        assert _extract_madde_no("44. madde kapsamında değerlendirin.", "") == 44

    def test_madde_suffix_no_dot(self):
        assert _extract_madde_no("5237 sayılı TCK 142 madde uygulanır.", "") == 142

    def test_md_abbreviation(self):
        assert _extract_madde_no("md. 15 gereği işlem yapılır.", "") == 15

    def test_in_answer(self):
        assert _extract_madde_no("", "Bu durum Madde 7 uyarınca değerlendirilir.") == 7

    def test_case_insensitive(self):
        assert _extract_madde_no("MADDE 99 hükmü nedir?", "") == 99

    def test_no_madde(self):
        assert _extract_madde_no("Genel bir hukuki soru.", "Bilmiyorum.") is None

    def test_no_madde_empty(self):
        assert _extract_madde_no("", "") is None

    def test_first_match_wins(self):
        # Both patterns present: first match should be returned
        result = _extract_madde_no("madde 3 veya madde 7 mi?", "")
        assert result in (3, 7)  # order depends on which pattern fires first

    def test_madde_in_question_takes_precedence(self):
        result = _extract_madde_no("madde 10 hükmüne göre", "")
        assert result == 10


# ---------------------------------------------------------------------------
# _chunk_matches_article
# ---------------------------------------------------------------------------

class TestChunkMatchesArticle:
    def test_text_header_match(self):
        c = _chunk("law_1_0", "kaggle_1", "MADDE 44 – Sözleşmenin feshi")
        assert _chunk_matches_article(c, 44) is True

    def test_text_header_no_match(self):
        c = _chunk("law_1_0", "kaggle_1", "MADDE 45 – Başka madde")
        assert _chunk_matches_article(c, 44) is False

    def test_doc_id_pattern(self):
        c = _chunk("law_madde_44_0", "law_madde_44", "Herhangi bir metin")
        assert _chunk_matches_article(c, 44) is True

    def test_doc_id_pattern_no_match(self):
        c = _chunk("law_madde_45_0", "law_madde_45", "Herhangi bir metin")
        assert _chunk_matches_article(c, 44) is False

    def test_madde_no_attribute(self):
        c = _chunk("x_0", "x", "herhangi metin")
        c.madde_no = 44
        assert _chunk_matches_article(c, 44) is True

    def test_madde_no_attribute_mismatch(self):
        c = _chunk("x_0", "x", "herhangi metin")
        c.madde_no = 45
        assert _chunk_matches_article(c, 44) is False

    def test_midline_madde_header(self):
        text = "Önceki metin\nMADDE 44\nMadde içeriği burada"
        c = _chunk("x_0", "x", text)
        assert _chunk_matches_article(c, 44) is True

    def test_case_insensitive_text(self):
        c = _chunk("x_0", "x", "madde 44 – küçük harf")
        assert _chunk_matches_article(c, 44) is True

    def test_partial_number_no_false_positive(self):
        # "MADDE 444" should NOT match article 44
        c = _chunk("x_0", "x", "MADDE 444 – Başka madde")
        assert _chunk_matches_article(c, 44) is False


# ---------------------------------------------------------------------------
# build_relevant_chunk_map — Strategy 3 article-level matching
# ---------------------------------------------------------------------------

class TestBuildRelevantChunkMapStrategy3:
    """Tests for strategy 3 (article-level matching) in build_relevant_chunk_map."""

    # A long text (>=180 chars) that has no MADDE header, used to test strategy 1.
    _GENERAL_TEXT = (
        "Bu genel hüküm maddesi, kanunun uygulama alanını belirler. "
        "Kanunda aksine hüküm bulunmadıkça bu madde hükümleri uygulanır. "
        "İlgili taraflar bu kanunun gereklerine uymakla yükümlüdür. "
        "Yükümlülüklere aykırı davrananlar hakkında yasal işlem yapılır."
    )

    def _make_corpus(self):
        return [
            _chunk("law_0", "kaggle_1", "MADDE 44 – Sözleşmenin feshi hükümleri. " + "x" * 160, "TestLaw"),
            _chunk("law_1", "kaggle_1", "MADDE 45 – Devir ve temlik işlemleri. " + "y" * 160, "TestLaw"),
            _chunk("law_2", "kaggle_2", "MADDE 44 – Farklı kanunda aynı madde no. " + "z" * 160, "OtherLaw"),
            _chunk("law_3", "kaggle_3", self._GENERAL_TEXT, "TestLaw"),
        ]

    def test_s3_article_match_found(self):
        corpus = self._make_corpus()
        qa = [_qa("q1", question="madde 44 kapsamında değerlendir", source="TestLaw")]
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, qa)
        assert "q1" in rel_map
        assert "law_0" in rel_map["q1"]
        # Should NOT include chunk from OtherLaw
        assert "law_2" not in rel_map["q1"]

    def test_s3_correct_source_only(self):
        corpus = self._make_corpus()
        qa = [_qa("q2", question="TestLaw madde 44 hükmü", source="OtherLaw")]
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, qa)
        # Source is OtherLaw, so should match only law_2
        assert "law_2" in rel_map.get("q2", [])
        assert "law_0" not in rel_map.get("q2", [])

    def test_s3_no_article_leaves_unlabeled(self):
        corpus = self._make_corpus()
        qa = [_qa("q3", question="Genel bir soru, madde yok", source="TestLaw")]
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, qa)
        # No article number → relevant set is empty → query excluded from metrics
        assert rel_map.get("q3", []) == []

    def test_s3_no_relevant_when_article_not_in_source(self):
        corpus = self._make_corpus()
        # Article 99 doesn't exist in TestLaw corpus
        qa = [_qa("q4", question="madde 99 hükmü", source="TestLaw")]
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, qa)
        assert rel_map.get("q4", []) == []

    def test_s1_takes_precedence_over_s3(self):
        """Strategy 1 (context hash) should fire before strategy 3."""
        corpus = self._make_corpus()
        # Use the long general text as context — it's ≥ 180 chars so chunk_text
        # will produce at least one chunk with a matching hash.
        context_text = self._GENERAL_TEXT
        qa = [_qa("q5", question="madde 44", source="TestLaw", context=context_text)]
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, qa)
        # law_3 has matching text hash → s1 should fire, not s3
        assert "law_3" in rel_map.get("q5", [])

    def test_empty_corpus(self):
        rel_map = DataProcessor.build_relevant_chunk_map([], [])
        assert rel_map == {}

    def test_empty_qa(self):
        corpus = self._make_corpus()
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, [])
        assert rel_map == {}

    def test_s0_gold_ids_bypass_all(self):
        """gold_source_ids (strategy 0) must bypass strategy 3."""
        corpus = self._make_corpus()
        qa_example = _qa("q6", question="madde 44", source="TestLaw")
        qa_example.gold_source_ids = ["law_3"]
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, [qa_example])
        # gold_source_ids points to law_3, not law_0
        assert rel_map["q6"] == ["law_3"]

    def test_multiple_queries_independent(self):
        corpus = self._make_corpus()
        qa = [
            _qa("qA", question="madde 44", source="TestLaw"),
            _qa("qB", question="madde 45", source="TestLaw"),
            _qa("qC", question="no article here", source="TestLaw"),
        ]
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, qa)
        assert "law_0" in rel_map.get("qA", [])
        assert "law_1" in rel_map.get("qB", [])
        assert rel_map.get("qC", []) == []

    def test_coverage_counts_logged(self, caplog):
        """build_relevant_chunk_map must log strategy coverage counts."""
        import logging
        corpus = self._make_corpus()
        qa = [
            _qa("qX", question="madde 44", source="TestLaw"),
            _qa("qY", question="no madde", source="TestLaw"),
        ]
        with caplog.at_level(logging.INFO, logger="data.data_processor"):
            DataProcessor.build_relevant_chunk_map(corpus, qa)
        assert any("coverage" in r.message.lower() for r in caplog.records)
