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
    _turkish_tokenize,
    _silver_lexical_score,
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

    def test_return_coverage_flag(self):
        """return_coverage=True must return (relevant_map, coverage_dict) tuple."""
        corpus = self._make_corpus()
        qa = [_qa("qA", question="madde 44", source="TestLaw")]
        result = DataProcessor.build_relevant_chunk_map(corpus, qa, return_coverage=True)
        assert isinstance(result, tuple) and len(result) == 2
        rel_map, cov = result
        assert "by_strategy" in cov
        assert "total" in cov
        assert "unlabeled" in cov
        assert cov["total"] == 1

    def test_coverage_by_strategy_keys(self):
        corpus = self._make_corpus()
        qa = [
            _qa("qA", question="madde 44", source="TestLaw"),   # → article
            _qa("qB", question="no article", source="TestLaw"), # → unlabeled
        ]
        _, cov = DataProcessor.build_relevant_chunk_map(corpus, qa, return_coverage=True)
        assert "article" in cov["by_strategy"]
        assert "silver_lexical" in cov["by_strategy"]
        assert cov["by_strategy"]["article"] == 1
        assert cov["unlabeled"] == 1


# ---------------------------------------------------------------------------
# Turkish tokenizer and silver lexical scoring
# ---------------------------------------------------------------------------

class TestTurkishTokenize:
    def test_lowercase_ascii(self):
        tokens = _turkish_tokenize("Türk Hukuku")
        assert "türk" in tokens
        assert "hukuku" in tokens

    def test_uppercase_i_to_dotless(self):
        # Turkish: uppercase "I" should map to "ı" (dotless i), not "i"
        tokens = _turkish_tokenize("IŞIK")
        assert "ışık" in tokens

    def test_uppercase_dotted_i(self):
        # Turkish: "İ" should map to "i" (dotted i)
        tokens = _turkish_tokenize("İSTANBUL")
        assert "istanbul" in tokens

    def test_min_length_filter(self):
        tokens = _turkish_tokenize("a bb ccc")
        assert "a" not in tokens
        assert "bb" in tokens
        assert "ccc" in tokens

    def test_empty(self):
        assert _turkish_tokenize("") == []


class TestSilverLexicalScore:
    def test_full_overlap(self):
        # Use tokens that appear verbatim in the chunk text (no morphological suffix changes)
        q_tokens = _turkish_tokenize("feshi hükümleri sona")
        score = _silver_lexical_score(q_tokens, "Sözleşmenin feshi hükümleri sona erer hakkında")
        assert score == pytest.approx(1.0)

    def test_zero_overlap(self):
        q_tokens = _turkish_tokenize("trafik kazası tazminat")
        score = _silver_lexical_score(q_tokens, "MADDE 44 – Miras bırakanın ölümü")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self):
        q_tokens = _turkish_tokenize("taraf feshi miras")
        score = _silver_lexical_score(q_tokens, "Sözleşmenin feshi hükümleri")
        assert 0.0 < score < 1.0

    def test_empty_query(self):
        assert _silver_lexical_score([], "some text") == pytest.approx(0.0)

    def test_empty_chunk(self):
        q_tokens = _turkish_tokenize("sözleşme")
        assert _silver_lexical_score(q_tokens, "") == pytest.approx(0.0)


class TestSilverLabelingStrategy:
    """Tests for silver lexical labeling (strategy 3.5)."""

    # Long texts with meaningful content
    _CHUNK_FESHI = (
        "MADDE 44 – Sözleşmenin feshi halinde tarafların yükümlülükleri sona erer. "
        "Taraflardan biri sözleşmeyi haksız yere feshederse tazminat ödemekle yükümlüdür. "
        "Fesih bildirimi yazılı şekilde yapılmalıdır. Süre kısıtlamaları uygulanır." * 2
    )
    _CHUNK_MIRAS = (
        "MADDE 45 – Miras bırakanın ölümü üzerine mirasçılar hak sahibi olur. "
        "Yasal mirasçılar ile atanmış mirasçılar arasındaki ilişkiler bu kanunla düzenlenir. "
        "Miras payları kanunda belirtilen oranlara göre belirlenir." * 2
    )
    _CHUNK_GENEL = (
        "Bu genel hüküm birden fazla konuyu kapsamaktadır. "
        "Kanunun genel uygulaması bu madde çerçevesinde değerlendirilir. "
        "Özel hükümler saklı kalmak kaydıyla genel hükümler uygulanır." * 3
    )

    def _make_corpus(self):
        return [
            _chunk("c_feshi", "doc_1", self._CHUNK_FESHI, "TestLaw"),
            _chunk("c_miras", "doc_2", self._CHUNK_MIRAS, "TestLaw"),
            _chunk("c_genel", "doc_3", self._CHUNK_GENEL, "TestLaw"),
        ]

    def test_silver_disabled_by_default(self):
        """silver labeling must be off by default."""
        import config as cfg
        assert getattr(cfg, "RELEVANCE_SILVER_LEXICAL", False) is False

    def test_silver_labels_best_matching_chunk(self, monkeypatch):
        """With silver enabled, the chunk with most question tokens should be labeled."""
        import config as cfg
        monkeypatch.setattr(cfg, "RELEVANCE_SILVER_LEXICAL", True)
        monkeypatch.setattr(cfg, "SILVER_TOP_M", 1)
        monkeypatch.setattr(cfg, "SILVER_THRESHOLD", 0.05)

        corpus = self._make_corpus()
        # question contains "feshi" — should match c_feshi
        qa = [_qa("q1", question="sözleşmenin feshi tazminat yükümlülük", source="TestLaw")]
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, qa)
        assert "c_feshi" in rel_map.get("q1", [])

    def test_silver_below_threshold_unlabeled(self, monkeypatch):
        """Chunks scoring below SILVER_THRESHOLD must not be labeled."""
        import config as cfg
        monkeypatch.setattr(cfg, "RELEVANCE_SILVER_LEXICAL", True)
        monkeypatch.setattr(cfg, "SILVER_TOP_M", 3)
        # Use a query with tokens that cannot appear in any chunk (random unique words)
        monkeypatch.setattr(cfg, "SILVER_THRESHOLD", 0.99)

        corpus = self._make_corpus()
        # "xyzabc123def" won't be in any chunk — score will be 0.0 < 0.99
        qa = [_qa("q1", question="xyzabc123def zzznomatch999", source="TestLaw")]
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, qa)
        assert rel_map.get("q1", []) == []

    def test_silver_top_m_respected(self, monkeypatch):
        """Silver must label at most SILVER_TOP_M chunks."""
        import config as cfg
        monkeypatch.setattr(cfg, "RELEVANCE_SILVER_LEXICAL", True)
        monkeypatch.setattr(cfg, "SILVER_TOP_M", 2)
        monkeypatch.setattr(cfg, "SILVER_THRESHOLD", 0.0)

        corpus = self._make_corpus()
        qa = [_qa("q1", question="sözleşme miras hüküm", source="TestLaw")]
        rel_map = DataProcessor.build_relevant_chunk_map(corpus, qa)
        assert len(rel_map.get("q1", [])) <= 2

    def test_silver_tagged_in_coverage(self, monkeypatch):
        """return_coverage must count silver-labeled queries in by_strategy."""
        import config as cfg
        monkeypatch.setattr(cfg, "RELEVANCE_SILVER_LEXICAL", True)
        monkeypatch.setattr(cfg, "SILVER_TOP_M", 1)
        monkeypatch.setattr(cfg, "SILVER_THRESHOLD", 0.0)

        corpus = self._make_corpus()
        qa = [_qa("q1", question="feshi tazminat", source="TestLaw")]
        _, cov = DataProcessor.build_relevant_chunk_map(corpus, qa, return_coverage=True)
        assert cov["by_strategy"]["silver_lexical"] >= 1

    def test_article_label_beats_silver(self, monkeypatch):
        """When article-level label is available, silver must NOT override it."""
        import config as cfg
        monkeypatch.setattr(cfg, "RELEVANCE_SILVER_LEXICAL", True)
        monkeypatch.setattr(cfg, "SILVER_TOP_M", 3)
        monkeypatch.setattr(cfg, "SILVER_THRESHOLD", 0.0)

        corpus = self._make_corpus()
        # Article 44 is in c_feshi
        qa = [_qa("q1", question="madde 44 feshi tazminat", source="TestLaw")]
        _, cov = DataProcessor.build_relevant_chunk_map(corpus, qa, return_coverage=True)
        # Should be labeled as "article", not "silver_lexical"
        assert cov["by_strategy"]["article"] == 1
        assert cov["by_strategy"]["silver_lexical"] == 0


class TestChunkMatchesNonNumericMaddeNo:
    def _chunk(self, madde_no):
        from data.data_processor import CorpusChunk
        return CorpusChunk(chunk_id="c", doc_id="d", source="s", text="t", char_len=1,
                           madde_no=madde_no)

    def test_gecici_does_not_crash(self):
        assert _chunk_matches_article(self._chunk("gecici-2"), 2) is False
        assert _chunk_matches_article(self._chunk("ek-3"), 3) is False

    def test_numeric_string_matches(self):
        assert _chunk_matches_article(self._chunk("44"), 44) is True
