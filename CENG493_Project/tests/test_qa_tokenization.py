import pytest

from evaluation import qa_metrics as qm
from evaluation.qa_metrics import (
    _tokenize, token_f1, answer_containment, rouge_l_score, answer_length_words,
    compute_all_qa_metrics, compute_all_qa_metrics_with_citation,
    compute_per_query_qa_metrics,
)


def test_punctuation_does_not_glue():
    assert _tokenize("Madde 86, hırsızlıktır.") == ["madde", "86", "hırsızlıktır"]
    assert token_f1("madde.", "madde") == 1.0
    assert answer_containment("Ceza (beş yıl).", "beş yıl") == 1.0


def test_turkish_casing():
    assert _tokenize("IŞIK İSTANBUL") == ["ışık", "istanbul"]
    assert token_f1("ISPARTA", "ısparta") == 1.0
    assert token_f1("İzmir", "izmir") == 1.0


def test_turkish_letters_kept_in_tokens():
    assert _tokenize("çağrı öğüt şiir") == ["çağrı", "öğüt", "şiir"]


def test_rouge_fallback_uses_unicode_tokens(monkeypatch):
    monkeypatch.setattr(qm, "_USE_HF_EVALUATE", False)
    assert rouge_l_score("Çocuk, okula gitti.", "çocuk okula gitti") == pytest.approx(1.0)


def test_rouge_hf_path_receives_custom_tokenizer(monkeypatch):
    seen = {}

    class R:
        def compute(self, **kw):
            seen.update(kw)
            return {"rougeL": 0.5}

    monkeypatch.setattr(qm, "_USE_HF_EVALUATE", True)
    monkeypatch.setattr(qm, "_ROUGE_METRIC", R(), raising=False)
    assert rouge_l_score("Çağrı.", "çağrı") == 0.5
    assert seen["tokenizer"]("Çağrı, ÖĞÜT") == ["çağrı", "öğüt"]


def test_answer_length_strips_citations():
    assert answer_length_words("Bir iki üç [Kaynak 1].") == 3


def test_mean_answer_len_in_aggregate():
    r = compute_all_qa_metrics([
        {"predicted": "bir iki", "expected": "x"},
        {"predicted": "a b c d", "expected": "y"},
    ])
    assert r["mean_answer_len_words"] == 3.0


def test_per_query_metrics():
    rows = compute_per_query_qa_metrics([{"query_id": "q", "predicted": "a b", "expected": "a b"}])
    assert rows[0]["query_id"] == "q" and rows[0]["f1"] == 1.0 and rows[0]["answer_len_words"] == 2


def _p(native, injected):
    return {
        "predicted": injected, "predicted_native": native, "expected": "x",
        "retrieved_chunks": [{"source": "LawA", "text": "t"}], "expected_source": "LawA",
    }


def test_citation_native_vs_injected_separate():
    r = compute_all_qa_metrics_with_citation([_p("no cite", "no cite [Kaynak 1]")])
    assert r["citation_accuracy_native"] == 0.0
    assert r["citation_accuracy_injected"] == 1.0
    assert r["citation_presence_rate_native"] == 0.0
    assert r["citation_presence_rate_injected"] == 1.0
    assert r["source_in_context_rate"] == 1.0


def test_citation_native_none_without_native_text():
    p = _p("a", "a [Kaynak 1]")
    del p["predicted_native"]
    r = compute_all_qa_metrics_with_citation([p])
    assert r["citation_accuracy_native"] is None
    assert r["citation_accuracy_injected"] == 1.0
