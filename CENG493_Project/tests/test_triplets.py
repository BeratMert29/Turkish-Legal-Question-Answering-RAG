"""scripts/11 -> scripts/12 triplet hand-off and article-matched positives."""

import importlib.util
import pathlib

import pytest

_SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), _SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s11():
    return _load("11_build_embedding_triplets.py")


@pytest.fixture(scope="module")
def s12():
    return _load("12_finetune_embeddings.py")


def test_finetune_reads_triplet_format_written_by_builder(s12):
    raw = [
        {"query": "q1", "pos": ["p1"], "neg": ["n1", "n2", "n3"]},
        {"query": "q2", "pos": ["p2"], "neg": ["m1", "m2"]},
    ]
    records, n = s12.build_records(raw, max_negatives=7)
    assert n == 2  # every row gets the same number of negative columns
    assert records[0] == {"anchor": "q1", "positive": "p1",
                          "negative": "n1", "negative_1": "n2"}
    assert set(records[1]) == set(records[0])


def test_finetune_accepts_legacy_keys(s12):
    records, n = s12.build_records(
        [{"query": "q", "positive_passage": "p", "negative_passage": "n"}], 7)
    assert n == 1 and records[0]["negative"] == "n"


def test_explicit_article_from_question(s11):
    assert s11.explicit_article("Anayasa madde 1'e göre devlet şekli nedir") == (
        "Türkiye Cumhuriyeti Anayasası", "1")
    assert s11.explicit_article("TCK 86. madde neyi düzenler?") == ("Türk Ceza Kanunu", "86")
    assert s11.explicit_article("Kira sözleşmesi nedir?") is None
