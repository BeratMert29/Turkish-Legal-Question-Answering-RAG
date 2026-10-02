"""Tests for data.data_processor._article_chunk and madde_no extraction."""

import pytest

from data.data_processor import DataProcessor, CorpusChunk, _madde_no_from_text


# ---------------------------------------------------------------------------
# _madde_no_from_text unit tests
# ---------------------------------------------------------------------------

class TestMaddeNoFromText:
    def test_regular_madde_uppercase(self):
        assert _madde_no_from_text("MADDE 86- Kasten yaralama.") == "86"

    def test_regular_madde_titlecase(self):
        assert _madde_no_from_text("Madde 12 – Tanım.") == "12"

    def test_ek_madde(self):
        assert _madde_no_from_text("Ek Madde 3 – Ek hüküm.") == "ek-3"

    def test_ek_madde_allcaps_ek(self):
        assert _madde_no_from_text("EK Madde 5 – Ek hüküm.") == "ek-5"

    def test_gecici_madde_titlecase(self):
        assert _madde_no_from_text("Geçici Madde 1 – Geçici hüküm.") == "gecici-1"

    def test_gecici_madde_lowercase_g(self):
        assert _madde_no_from_text("geçici Madde 7 – Geçici madde.") == "gecici-7"

    def test_gecici_madde_allcaps(self):
        assert _madde_no_from_text("GEÇİCİ MADDE 4 – Allcaps.") == "gecici-4"

    def test_no_madde_heading_returns_none(self):
        assert _madde_no_from_text("BAŞLANGIÇ — preamble text.") is None

    def test_empty_text_returns_none(self):
        assert _madde_no_from_text("") is None

    def test_madde_within_first_600_chars(self):
        # text where MADDE appears after 500 chars should still be found
        padding = "x" * 490
        text = padding + " MADDE 99- Bir hüküm."
        result = _madde_no_from_text(text)
        assert result == "99"

    def test_madde_beyond_600_chars_not_found(self):
        # text where MADDE only appears after 600 chars: should return None
        padding = "x" * 601
        text = padding + "MADDE 50- Hüküm."
        result = _madde_no_from_text(text)
        assert result is None


# ---------------------------------------------------------------------------
# _article_chunk tests
# ---------------------------------------------------------------------------

# Each article must be >= MIN_CHUNK_CHARS (180) to survive the length filter.
_FILLER = "Bu madde hükmü Türk hukuk mevzuatında yer almaktadır ve uygulanması zorunludur. " * 4

_MULTI_ARTICLE_TEXT = (
    f"MADDE 1- {_FILLER}\n\n"
    f"MADDE 2- İkinci madde içeriği. Detaylı bilgi için Madde 1'e bakınız. {_FILLER}\n\n"
    f"MADDE 3- Üçüncü madde içeriği. {_FILLER}"
)

# Ek/Geçici Madde: these DON'T start with "MADDE N" so _ARTICLE_RE won't split
# them further — they arrive as standalone chunks when their text is long enough.
_EK_MADDE_TEXT = f"Ek Madde 1 – Ek hüküm içeriği. {_FILLER}"
_GECICI_MADDE_TEXT = f"Geçici Madde 2 – Geçici hüküm içeriği. {_FILLER}"

_PREAMBLE_TEXT = (
    "BAŞLANGIÇ\n\n"
    "Türk Vatanı ve Milletinin ebedi varlığını ve Yüce Türk Devletinin bölünmez bütünlüğünü "
    "belirleyen bu Anayasa Türk Milletinin tarihî birikiminin ürünü olarak hazırlanmıştır. "
    + "Preamble devamı burada yer almaktadır. " * 5
    + f"\n\nMADDE 1- Birinci madde içeriği. {_FILLER}"
)


class TestArticleChunk:
    def test_splits_at_madde_boundaries(self):
        chunks = DataProcessor._article_chunk(_MULTI_ARTICLE_TEXT, "kaggle_1", "TestSource")
        madde_nos = [c.madde_no for c in chunks]
        assert "1" in madde_nos
        assert "2" in madde_nos
        assert "3" in madde_nos

    def test_chunk_ids_are_sequential(self):
        chunks = DataProcessor._article_chunk(_MULTI_ARTICLE_TEXT, "kaggle_1", "TestSource")
        indices = [int(c.chunk_id.split("_")[-1]) for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_preamble_chunk_madde_no_none(self):
        """Preamble section before any MADDE heading has madde_no=None."""
        chunks = DataProcessor._article_chunk(_PREAMBLE_TEXT, "kaggle_anayasa", "Anayasa")
        # There should be a preamble chunk (no MADDE heading) and a madde-1 chunk
        none_chunks = [c for c in chunks if c.madde_no is None]
        assert len(none_chunks) >= 1, "Expected at least one preamble chunk with madde_no=None"

    def test_ek_madde_extraction(self):
        """Standalone Ek Madde text (not split further by ARTICLE_RE) gets ek-N."""
        chunks = DataProcessor._article_chunk(_EK_MADDE_TEXT, "law_extra_1", "TestSource")
        assert len(chunks) >= 1
        assert chunks[0].madde_no == "ek-1"

    def test_gecici_madde_extraction(self):
        """Standalone Geçici Madde text gets gecici-N."""
        chunks = DataProcessor._article_chunk(_GECICI_MADDE_TEXT, "law_extra_2", "TestSource")
        assert len(chunks) >= 1
        assert chunks[0].madde_no == "gecici-2"

    def test_chunk_id_format_unchanged(self):
        """chunk_id must remain f'{source}_{doc_id}_{index}' for backward compat."""
        chunks = DataProcessor._article_chunk(_MULTI_ARTICLE_TEXT, "kaggle_1", "TestSource")
        for chunk in chunks:
            assert chunk.chunk_id.startswith("TestSource_kaggle_1_")

    def test_source_and_doc_id_preserved(self):
        chunks = DataProcessor._article_chunk(_MULTI_ARTICLE_TEXT, "kaggle_5", "SomeSource")
        for chunk in chunks:
            assert chunk.source == "SomeSource"
            assert chunk.doc_id == "kaggle_5"

    def test_all_chunks_are_corpus_chunk_instances(self):
        chunks = DataProcessor._article_chunk(_MULTI_ARTICLE_TEXT, "kaggle_1", "TestSource")
        for c in chunks:
            assert isinstance(c, CorpusChunk)

    def test_madde_no_in_asdict(self):
        """madde_no must be serialisable via dataclasses.asdict."""
        from dataclasses import asdict
        chunks = DataProcessor._article_chunk(_MULTI_ARTICLE_TEXT, "kaggle_1", "TestSource")
        for c in chunks:
            d = asdict(c)
            assert "madde_no" in d

    def test_oversized_article_sub_chunks_share_madde_no(self):
        """When an article exceeds CHUNK_SIZE, all its sub-chunks share madde_no."""
        import config
        long_article = "MADDE 99- " + ("uzun metin içeriği. " * 500)
        chunks = DataProcessor._article_chunk(long_article, "kaggle_x", "BigSource")
        # All sub-chunks should have the same madde_no
        unique_madde = {c.madde_no for c in chunks}
        assert unique_madde == {"99"}, f"Expected only madde_no='99', got {unique_madde}"

    def test_char_len_positive(self):
        chunks = DataProcessor._article_chunk(_MULTI_ARTICLE_TEXT, "doc1", "Src")
        for c in chunks:
            assert c.char_len > 0
            assert c.char_len == len(c.text)


def test_gecici_madde_no_not_matched_as_int():
    from data.data_processor import CorpusChunk, _chunk_matches_article
    c = CorpusChunk(chunk_id="c", doc_id="d", source="s", text="t", char_len=1, madde_no="gecici-2")
    assert _chunk_matches_article(c, 2) is False
