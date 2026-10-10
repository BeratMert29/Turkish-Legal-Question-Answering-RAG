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


def test_rouge_uses_unicode_tokens():
    assert rouge_l_score("Çocuk, okula gitti.", "çocuk okula gitti") == pytest.approx(1.0)
    assert rouge_l_score("Çağrı yapıldı", "çağrı") == pytest.approx(2 * 0.5 * 1 / 1.5)


@pytest.mark.skipif(not qm._USE_SACREBLEU, reason="sacrebleu not installed")
def test_bleu_and_chrf_match_sacrebleu_on_normalised_text():
    import sacrebleu
    pred, ref = "Süre BEŞ gündür, itiraz edilebilir.", "süre beş gündür"
    norm_p, norm_r = "süre beş gündür itiraz edilebilir", "süre beş gündür"
    assert qm.chrf_score(pred, ref) == pytest.approx(
        sacrebleu.sentence_chrf(norm_p, [norm_r], word_order=2).score / 100)
    agg = compute_all_qa_metrics([{"predicted": pred, "expected": ref}])
    assert agg["bleu"] == pytest.approx(
        sacrebleu.corpus_bleu([norm_p], [[norm_r]], tokenize="none").score / 100)
    assert agg["lexical_impl"]["bleu"].startswith("sacrebleu")


def test_short_perfect_answer_bleu_is_not_zero():
    if qm._USE_SACREBLEU:
        assert qm.bleu_score("Ankara", "Ankara") == pytest.approx(1.0)


def test_fallback_without_sacrebleu(monkeypatch):
    monkeypatch.setattr(qm, "_USE_SACREBLEU", False)
    assert qm.chrf_score("a b", "a b") is None
    agg = compute_all_qa_metrics([{"predicted": "a b c d e", "expected": "a b c d e"}])
    assert agg["bleu"] == pytest.approx(1.0) and agg["chrf"] is None
    assert agg["lexical_impl"]["bleu"] == "fallback"


def test_token_precision_recall_split_length_effect():
    from evaluation.qa_metrics import token_prf
    p, r, f = token_prf("süre beş gündür ve ayrıca uzun bir açıklama metni", "süre beş gündür")
    assert r == 1.0 and p == pytest.approx(3 / 9) and f == pytest.approx(0.5)


@pytest.mark.parametrize("pred,gold,em", [
    ("Cevap: C) 25", "5", 0.0),            # "5" inside "25"
    ("28/10/2023 tarihinde", "2", 0.0),    # "2" inside a date
    ("Süre 5 gündür.", "5", 1.0),
    ("Başkent Ankara'dır.", "Ankara", 1.0),
    ("Türkiye Devleti bir Cumhuriyettir.", "bir cumhuriyettir", 1.0),
])
def test_exact_match_respects_token_boundaries(pred, gold, em):
    from evaluation.qa_metrics import exact_match
    assert exact_match(pred, gold) == em


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
