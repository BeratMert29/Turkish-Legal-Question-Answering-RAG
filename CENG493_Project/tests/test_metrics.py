"""
tests/test_metrics.py — Unit tests for evaluation metrics.

Tests qa_metrics.py (including answer_containment and EM HMGS limitation)
and retrieval_metrics.py (unlabeled query exclusion).
All tests run offline: no network, GPU, or Ollama required.
"""

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.qa_metrics import (
    exact_match,
    token_f1,
    answer_containment,
    compute_qa_metrics,
    compute_all_qa_metrics,
    compute_all_qa_metrics_with_citation,
)


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_identical(self):
        assert exact_match("hello world", "hello world") == 1.0

    def test_substring(self):
        # EM uses containment: expected substring of predicted
        assert exact_match("The answer is 42 exactly.", "42") == 1.0

    def test_no_match(self):
        assert exact_match("completely different text", "expected answer") == 0.0

    def test_empty_expected_returns_zero(self):
        assert exact_match("anything", "") == 0.0

    def test_hmgs_style_paraphrase_returns_zero(self):
        # LLM paraphrase: exact text not present → EM = 0.
        # This demonstrates the known HMGS EM ~0 limitation.
        expected = "Karar yeter sayısı sağlanmış olsa da toplantı yeter sayısı sağlanamamıştır."
        predicted = "Toplantı yeter sayısı eksik olduğundan karar geçerli değildir."
        assert exact_match(predicted, expected) == 0.0


# ---------------------------------------------------------------------------
# answer_containment
# ---------------------------------------------------------------------------

class TestAnswerContainment:
    def test_all_tokens_present(self):
        # Every expected token appears in predicted
        assert answer_containment("madde kırk dört hükmü", "kırk dört") == pytest.approx(1.0)

    def test_partial_overlap(self):
        # 2 out of 4 expected tokens
        result = answer_containment("some predicted text", "predicted other missing tokens")
        assert 0.0 < result < 1.0

    def test_no_overlap(self):
        assert answer_containment("completely unrelated", "different words here") == pytest.approx(0.0)

    def test_empty_expected_returns_zero(self):
        assert answer_containment("anything", "") == pytest.approx(0.0)

    def test_empty_predicted_returns_zero(self):
        assert answer_containment("", "something") == pytest.approx(0.0)

    def test_higher_than_em_for_paraphrase(self):
        """answer_containment should be higher than EM for paraphrased HMGS answers."""
        expected = "Yüce Divan'a sevk kararı bakanlık görevini sona erdirir."
        predicted = (
            "Yüce Divan'a sevk kararı alındıktan sonra bakanın bakanlık görevi "
            "kendiliğinden sona erer."
        )
        em = exact_match(predicted, expected)
        cont = answer_containment(predicted, expected)
        assert em == 0.0
        assert cont > em

    def test_full_sentence_match(self):
        # When predicted contains all expected words
        result = answer_containment("bu madde kapsamında değerlendirme yapılır",
                                    "bu madde kapsamında")
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# token_f1
# ---------------------------------------------------------------------------

class TestTokenF1:
    def test_identical(self):
        assert token_f1("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert token_f1("abc def", "xyz uvw") == pytest.approx(0.0)

    def test_partial(self):
        result = token_f1("hello world foo", "hello world bar")
        assert 0.0 < result < 1.0

    def test_empty_both(self):
        assert token_f1("", "") == pytest.approx(1.0)

    def test_empty_one(self):
        assert token_f1("hello", "") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_qa_metrics
# ---------------------------------------------------------------------------

class TestComputeQaMetrics:
    def test_keys_present(self):
        result = compute_qa_metrics("test answer", "test answer")
        expected_keys = {"em", "f1", "bleu", "rouge_l", "answer_containment"}
        assert expected_keys.issubset(result.keys())

    def test_answer_containment_in_output(self):
        result = compute_qa_metrics("a b c d", "a b")
        assert "answer_containment" in result
        assert result["answer_containment"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# compute_all_qa_metrics
# ---------------------------------------------------------------------------

class TestComputeAllQaMetrics:
    def test_empty_returns_zeros(self):
        result = compute_all_qa_metrics([])
        assert result["em"] == 0.0
        assert result["f1"] == 0.0
        assert result["answer_containment"] == 0.0
        assert result["num_samples"] == 0

    def test_answer_containment_key_present(self):
        preds = [
            {"predicted": "madde kırk dört kapsamında", "expected": "kırk dört"},
            {"predicted": "farklı cevap", "expected": "kırk dört"},
        ]
        result = compute_all_qa_metrics(preds)
        assert "answer_containment" in result
        assert 0.0 <= result["answer_containment"] <= 1.0

    def test_num_samples(self):
        preds = [{"predicted": f"p{i}", "expected": f"e{i}"} for i in range(7)]
        result = compute_all_qa_metrics(preds)
        assert result["num_samples"] == 7


# ---------------------------------------------------------------------------
# compute_all_qa_metrics_with_citation
# ---------------------------------------------------------------------------

class TestComputeAllQaMetricsWithCitation:
    def test_keys_present(self):
        preds = [{
            "predicted": "test [Kaynak 1]",
            "expected": "test",
            "retrieved_chunks": [{"source": "LawA", "text": "chunk"}],
            "expected_source": "LawA",
        }]
        result = compute_all_qa_metrics_with_citation(preds)
        assert "answer_containment" in result
        assert "citation_accuracy_injected" in result and "citation_accuracy_native" in result and "citation_accuracy" not in result
        assert "num_samples" in result

    def test_empty(self):
        result = compute_all_qa_metrics_with_citation([])
        assert result["num_samples"] == 0
        assert result["answer_containment"] == 0.0


# ---------------------------------------------------------------------------
# retrieval_metrics — unlabeled query exclusion
# ---------------------------------------------------------------------------

class TestRetrievalMetricsExclusion:
    """Queries with empty relevant sets must be excluded from metric computation."""

    def test_unlabeled_queries_excluded(self):
        """compute_all_metrics must skip queries with empty relevant set."""
        try:
            from evaluation.retrieval_metrics import compute_all_metrics
        except ImportError:
            pytest.skip("ranx not installed")

        results = [
            # Labeled query: retrieved the relevant doc
            {"query_id": "q1", "relevant": ["doc1"], "retrieved": ["doc1", "doc2"]},
            # Unlabeled query: empty relevant set (no ground-truth)
            {"query_id": "q2", "relevant": [],        "retrieved": ["doc1", "doc2"]},
        ]
        metrics = compute_all_metrics(results)
        # q2 has no ground-truth → excluded
        assert metrics["num_queries"] == 1
        assert metrics["total_queries"] == 2
        # With 1 query and perfect retrieval, recall@1 should be 1.0
        assert metrics["recall_at_5"] == pytest.approx(1.0)

    def test_all_unlabeled_returns_none(self):
        try:
            from evaluation.retrieval_metrics import compute_all_metrics
        except ImportError:
            pytest.skip("ranx not installed")

        results = [
            {"query_id": "q1", "relevant": [], "retrieved": ["doc1"]},
            {"query_id": "q2", "relevant": [], "retrieved": ["doc2"]},
        ]
        metrics = compute_all_metrics(results)
        assert metrics["num_queries"] == 0
        assert metrics["recall_at_5"] is None  # unknown, not zero
