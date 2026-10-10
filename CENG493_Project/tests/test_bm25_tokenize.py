"""retrieval/bm25_retriever.tokenize with stemming/stopwords controlled."""

import pytest

from retrieval import bm25_retriever as bm


@pytest.fixture
def plain(monkeypatch):
    monkeypatch.setattr(bm, "_stem", lambda t: t)
    monkeypatch.setattr(bm, "_STOPWORDS", {"ve", "bir"})


def test_turkish_case_and_punctuation(plain):
    assert bm.tokenize("İSTANBUL, Işık ve bir madde.") == ["istanbul", "ışık", "madde"]


def test_single_digit_article_numbers_are_kept(plain):
    assert bm.tokenize("madde 5 ile 18/A") == ["madde", "5", "ile", "18"]


def test_tokenizer_info_reports_environment(plain):
    info = bm.tokenizer_info()
    assert info["n_stopwords"] == 2 and "stemmer" in info
