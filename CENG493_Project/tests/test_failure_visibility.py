"""Tests for failure visibility: judge None, generation_error, parse_score, ranx empty, BLEU fallback."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evaluation import llm_judge
from evaluation.llm_judge import _aggregate, _parse_score, llm_judge_answer
from evaluation.retrieval_metrics import compute_all_metrics
from pipeline import evaluation as pe


# --- _parse_score ----------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("0,8", 0.8),
    ("Score: 0,75", 0.75),
    ("1/1", 1.0),
    ("0/1", 0.0),
    ("7/10", 0.7),
    ("4/5", 0.8),
    ("0.7", 0.7),
])
def test_parse_score_ok(text, expected):
    assert _parse_score(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "no idea", "great answer", "17", "3/7"])
def test_parse_score_unrecognized_is_none(text):
    assert _parse_score(text) is None


# --- judge: empty response -> None, excluded from mean ----------------------

def test_judge_empty_response_is_failure(monkeypatch):
    replies = iter([None, "1", None, "0"])
    monkeypatch.setattr(llm_judge, "_ollama_generate", lambda *a, **k: next(replies))
    preds = [{"query_id": str(i), "question": "q", "expected": "e", "predicted": "p"} for i in range(4)]
    res = llm_judge_answer(preds, "http://x", "m", sample_size=10)
    assert res["parse_fail_count"] == 2
    assert res["score"] == pytest.approx(0.5)  # mean over the two valid only
    assert sum(1 for s in res["per_sample"] if s["score"] is None) == 2


def test_ollama_generate_empty_returns_none(monkeypatch):
    class R:
        def raise_for_status(self): pass
        def json(self): return {"response": "  "}
    monkeypatch.setattr(llm_judge.requests, "post", lambda *a, **k: R())
    monkeypatch.setattr(llm_judge.time, "sleep", lambda *_: None)
    assert llm_judge._ollama_generate("p", "http://x", "m") is None


def test_aggregate_all_failed():
    res = _aggregate([{"score": None, "parse_failed": True}])
    assert res["score"] is None and res["parse_fail_count"] == 1


# --- generation errors ------------------------------------------------------

class _Pipe:
    def assemble_context(self, chunks):
        return "ctx", chunks

    def generate(self, q, ctx):
        if q == "bad":
            raise RuntimeError("boom")
        return "ans"


def test_generation_error_flagged(capsys):
    qas = [SimpleNamespace(query_id="1", question="ok", answer="a", source="s"),
           SimpleNamespace(query_id="2", question="bad", answer="a", source="s")]
    preds = pe.run_generation_loop(_Pipe(), qas, [[], []])
    assert not preds[0].get("generation_error")
    assert preds[1]["generation_error"] is True
    assert preds[1]["predicted"] == ""
    assert "boom" in preds[1]["error"]


def test_failure_rate_exceeded():
    assert pe.failure_rate_exceeded(3, 10, 0.2) is True
    assert pe.failure_rate_exceeded(2, 10, 0.2) is False
    assert pe.failure_rate_exceeded(0, 0, 0.2) is False


def test_max_failure_rate_config():
    import config
    assert config.MAX_FAILURE_RATE == pytest.approx(0.2)


def test_run_llm_judge_eval_reports_failure_counts(monkeypatch):
    monkeypatch.setattr(llm_judge, "_ollama_generate", lambda *a, **k: None)
    preds = [{"query_id": "1", "predicted": "p", "expected": "e", "retrieved_chunks": []}]
    qas = [SimpleNamespace(query_id="1", question="q")]
    out = pe.run_llm_judge_eval(preds, qas, base_url="http://x", judge_model="m", sample_size=5)
    assert out["failure_count"] == 4 and out["call_count"] == 4
    assert out["score"] is None
    assert pe.failure_rate_exceeded(out["failure_count"], out["call_count"], 0.2)


# --- ranx empty retrieved ---------------------------------------------------

def test_empty_retrieved_scores_zero():
    pytest.importorskip("ranx")
    res = compute_all_metrics([
        {"query_id": "1", "retrieved": [], "relevant": ["a"]},
        {"query_id": "2", "retrieved": ["a"], "relevant": ["a"]},
    ])
    assert res["num_queries"] == 2
    assert res["recall_at_5"] == pytest.approx(0.5)
    assert res["mrr"] == pytest.approx(0.5)


# --- BLEU fallback ----------------------------------------------------------

def test_bleu_is_a_real_number_not_a_stub():
    from evaluation import qa_metrics
    assert qa_metrics.bleu_score("tamamen farklı kelimeler burada", "bambaşka cümle yok ki") < 0.5
    assert 0.0 <= qa_metrics.bleu_score("a b c d e", "a b c d e") <= 1.0
