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

    def test_ek_madde_allcaps_both(self):
        """EK MADDE N (both words uppercase) must return ek-N, not plain N."""
        assert _madde_no_from_text("EK MADDE 1 – ek hüküm.") == "ek-1"

    def test_madde_suffix_slash_a(self):
        """MADDE 183/A must return '183-a' (slash normalised to hyphen, lowercased)."""
        assert _madde_no_from_text("MADDE 183/A – Suç ve ceza.") == "183-a"

    def test_madde_suffix_hyphen_b(self):
        """MADDE 5-B must return '5-b'."""
        assert _madde_no_from_text("MADDE 5-B – Hüküm.") == "5-b"

    def test_mid_text_madde_not_matched(self):
        """'Madde 5 uyarınca' in the middle of a line must NOT set article number."""
        assert _madde_no_from_text("Bu hüküm Madde 5 uyarınca uygulanır.") is None

    def test_mid_text_madde_with_leading_content(self):
        """Inline mid-sentence madde ref must not match when a real heading follows."""
        # "Madde 3" is embedded mid-sentence (not at line start).
        # "MADDE 7" at line start is the actual heading.
        text = "Kanun hükmüne göre Madde 3 uyarınca karar verilmiştir.\nMADDE 7- Asıl hüküm."
        assert _madde_no_from_text(text) == "7"

    def test_madde_within_first_600_chars(self):
        # MADDE heading at the start of a new line after 490 chars of padding
        padding = "x" * 490 + "\n"
        text = padding + "MADDE 99- Bir hüküm."
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


class TestContinuationChunks:
    """Every chunk of an article carries its madde_no, not only the heading chunk."""

    def _article(self, n, filler=3000):
        return f"MADDE {n}- " + "Bu madde uzun bir hüküm içerir. " * (filler // 33)

    def test_article_continuation_chunks_inherit_madde_no(self):
        chunks = DataProcessor._article_chunk(self._article(7), "d", "L")
        assert len(chunks) >= 3
        assert {c.madde_no for c in chunks} == {"7"}

    def test_char_chunk_carries_heading_forward(self):
        text = self._article(12, 4000)
        chunks = DataProcessor._char_chunk(text, "d", "L")
        assert len(chunks) >= 3 and {c.madde_no for c in chunks} == {"12"}

    def test_chunk_with_new_heading_mid_text_keeps_previous_lead_article(self):
        from data.data_processor import _assign_madde_nos
        out = _assign_madde_nos(["MADDE 1- a", "devam eden metin\nMADDE 2- b", "devam 2"])
        assert out == ["1", "1", "2"]

    def test_heading_beyond_600_chars_is_found(self):
        text = "Giriş metni " * 100 + "\nMADDE 44- Hüküm."
        assert len(text) > 600
        assert _madde_no_from_text(text) == "44"

    def test_legacy_chunks_without_madde_no_label_all_chunks_of_gold_article(self):
        long = "MADDE 9- " + "uzun hüküm metni. " * 120
        parts = DataProcessor._article_chunk(long + "\nMADDE 10- diğer " + "x " * 100, "d", "L")
        legacy = [CorpusChunk(c.chunk_id, c.doc_id, c.text, c.source, c.char_len) for c in parts]
        qa = [{"query_id": "q", "question": "", "answer": "", "context": "",
               "source": "L", "madde_no": "9"}]
        rel = DataProcessor.build_relevant_chunk_map(legacy, qa)["q"]
        nine = [c.chunk_id for c in parts if c.madde_no == "9"]
        assert len(nine) >= 2 and rel == nine

    def test_article_tail_before_next_heading_is_labelled_for_both(self):
        # character chunks: chunk 2 starts with the end of article 94's body
        legacy = [
            CorpusChunk("c1", "d", "MADDE 94- (1) Hüküm metni burada başlar.", "L", 40),
            CorpusChunk("c2", "d", "(3) Tutuklama kararı gerekçeli olarak verilir ve "
                        "şüpheliye ile müdafiine derhâl yazılı olarak bildirilir; "
                        "itiraz yolu açıktır.\n"
                        "MADDE 95- (1) Sonraki hüküm.", "L", 140),
        ]
        qa = [{"query_id": "q", "question": "", "answer": "", "context": "",
               "source": "L", "madde_no": "94"}]
        assert DataProcessor.build_relevant_chunk_map(legacy, qa)["q"] == ["c1", "c2"]

    def test_title_lead_is_not_a_tail_of_the_previous_article(self):
        legacy = [
            CorpusChunk("c1", "d", "MADDE 1- Hüküm.", "L", 15),
            CorpusChunk("c2", "d", "I. Devletin şekli\nMadde 2 – Diğer hüküm.", "L", 40),
        ]
        qa = [{"query_id": "q", "question": "", "answer": "", "context": "",
               "source": "L", "madde_no": "1"}]
        assert DataProcessor.build_relevant_chunk_map(legacy, qa)["q"] == ["c1"]


# ---------------------------------------------------------------------------
# Article boundaries on real code layout (title above the heading)
# ---------------------------------------------------------------------------

_LAW = """BİRİNCİ KISIM
Genel Hükümler

Devletin şekli
Madde 1 – Türkiye Devleti bir Cumhuriyettir.

Cumhuriyetin nitelikleri
Madde 2 – Türkiye Cumhuriyeti, toplumun huzuru, millî dayanışma ve adalet anlayışı içinde, insan haklarına saygılı, Atatürk milliyetçiliğine bağlı, başlangıçta belirtilen temel ilkelere dayanan, demokratik, lâik ve sosyal bir hukuk Devletidir.

Geçiş hükmü
Geçici Madde 1 – Bu Kanunun yürürlüğe girdiği tarihte görevde bulunanlar görevlerine devam eder.
"""


class TestArticleBoundaries:
    def _chunks(self, text=_LAW):
        return DataProcessor._article_chunk(text, "d", "Anayasa")

    def test_title_belongs_to_its_own_article(self):
        by = {c.madde_no: c.text for c in self._chunks()}
        assert by["2"].startswith("Cumhuriyetin nitelikleri\nMadde 2")
        assert "Cumhuriyetin nitelikleri" not in by["1"]
        assert by["1"].rstrip().endswith("Cumhuriyettir.")

    def test_section_headers_go_with_the_first_article(self):
        first = self._chunks()[0]
        assert first.madde_no == "1"
        assert first.text.startswith("BİRİNCİ KISIM\nGenel Hükümler")

    def test_short_article_is_kept(self):
        assert "1" in {c.madde_no for c in self._chunks()}

    def test_gecici_madde_is_its_own_chunk(self):
        by = {c.madde_no: c.text for c in self._chunks()}
        assert by["gecici-1"].startswith("Geçiş hükmü\nGeçici Madde 1")
        assert "Geçici Madde" not in by["2"]

    def test_letter_glued_to_dash_is_not_a_suffix(self):
        assert _madde_no_from_text("Madde 605-Yasal ve atanmış mirasçılar") == "605"
        assert _madde_no_from_text("MADDE 6/A – (Ek) Hüküm") == "6-a"

    @pytest.mark.parametrize("line", [
        "Madde 3 14/4/2011",      # amendment table row
        "MADDE 1,",
        "Madde 9, Geçici",
        "Madde 5 uyarınca işlem yapılır.",
    ])
    def test_amendment_rows_and_references_are_not_headings(self, line):
        assert _madde_no_from_text(line) is None

    def test_mukerrer_madde(self):
        assert _madde_no_from_text("Mükerrer\nMadde 30 – (Ek: 1963)") == "mukerrer-30"

    def test_short_trailing_sub_chunk_merged_into_previous(self):
        body = "Madde 9 – " + ("Uzun bir fıkra metni. " * 80) + "Son."
        chunks = DataProcessor._article_chunk(body, "d", "S")
        assert all(c.madde_no == "9" for c in chunks)
        assert all(len(c.text) >= 180 for c in chunks)
        assert chunks[-1].text.rstrip().endswith("Son.")


def test_consecutive_repealed_articles_are_not_cut_inside_a_heading():
    text = ("A. Kuruluş\n\nMadde 109 – (Mülga: 21/1/2017-6771/16 md.)\n\n \n\n"
            "B. Göreve başlama\n\nMadde 110 – (Mülga: 21/1/2017-6771/16 md.)\n\n"
            "C. Görev\n\nMadde 111 – Hüküm metni burada yer alır.")
    by = {c.madde_no: c.text for c in DataProcessor._article_chunk(text, "d", "A")}
    assert by["109"].startswith("A. Kuruluş\n\nMadde 109")
    assert by["110"].startswith("B. Göreve başlama\n\nMadde 110")
    assert by["111"].startswith("C. Görev\n\nMadde 111")
    assert all("\nadde" not in t and not t.startswith("adde") for t in by.values())
