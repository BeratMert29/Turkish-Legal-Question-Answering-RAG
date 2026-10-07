"""
tests/test_llm_judge.py — Unit tests for evaluation/llm_judge.py.

All tests run offline: Ollama is mocked/patched.  No network or GPU required.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.llm_judge import (
    _parse_score,
    _subsample,
    _aggregate,
    save_raw_responses,
    llm_judge_answer,
    llm_judge_faithfulness,
    llm_judge_relevancy,
    llm_judge_coherence,
)


# ---------------------------------------------------------------------------
# _parse_score
# ---------------------------------------------------------------------------

class TestParseScore:
    def test_direct_float(self):
        assert _parse_score("0.7") == pytest.approx(0.7)

    def test_direct_zero(self):
        assert _parse_score("0") == pytest.approx(0.0)

    def test_direct_one(self):
        assert _parse_score("1") == pytest.approx(1.0)

    def test_n_over_10(self):
        assert _parse_score("7/10") == pytest.approx(0.7)

    def test_n_over_10_decimal(self):
        assert _parse_score("8.5/10") == pytest.approx(0.85)

    def test_n_over_5(self):
        assert _parse_score("4/5") == pytest.approx(0.8)

    def test_float_in_text(self):
        assert _parse_score("Score: 0.85 overall") == pytest.approx(0.85)

    def test_clamp_above_one(self):
        # "1.0" is valid but values above 1 via N/10 should be clamped
        assert _parse_score("15/10") == pytest.approx(1.0)

    def test_returns_none_on_failure(self):
        """_parse_score must return None (not 0.5) when nothing matches."""
        result = _parse_score("no score here at all!")
        assert result is None

    def test_returns_none_on_empty(self):
        result = _parse_score("")
        assert result is None

    def test_returns_none_on_prose(self):
        result = _parse_score("Bu cevap oldukça iyi bir cevaptır.")
        assert result is None

    def test_no_false_positive_from_years(self):
        # "2024" should not be parsed as a score
        result = _parse_score("Yıl 2024'te çıkarılan kanun.")
        assert result is None


# ---------------------------------------------------------------------------
# _aggregate
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_all_valid(self):
        samples = [
            {"score": 0.8, "parse_failed": False},
            {"score": 0.6, "parse_failed": False},
        ]
        r = _aggregate(samples)
        assert r["score"] == pytest.approx(0.7)
        assert r["parse_fail_count"] == 0
        assert r["sample_size"] == 2

    def test_some_none_excluded(self):
        samples = [
            {"score": 0.8, "parse_failed": False},
            {"score": None, "parse_failed": True},
            {"score": 0.6, "parse_failed": False},
        ]
        r = _aggregate(samples)
        # Mean of 0.8 and 0.6 only
        assert r["score"] == pytest.approx(0.7)
        assert r["parse_fail_count"] == 1

    def test_all_failed_returns_none_score(self):
        samples = [
            {"score": None, "parse_failed": True},
            {"score": None, "parse_failed": True},
        ]
        r = _aggregate(samples)
        assert r["score"] is None
        assert r["parse_fail_count"] == 2

    def test_empty_returns_none_score(self):
        r = _aggregate([])
        assert r["score"] is None
        assert r["parse_fail_count"] == 0


# ---------------------------------------------------------------------------
# _subsample
# ---------------------------------------------------------------------------

class TestSubsample:
    def test_subsample_smaller_than_list(self):
        items = list(range(100))
        result = _subsample(items, 10, seed=42)
        assert len(result) == 10

    def test_subsample_larger_returns_all(self):
        items = list(range(5))
        result = _subsample(items, 10, seed=42)
        assert result == items

    def test_subsample_deterministic(self):
        items = list(range(100))
        r1 = _subsample(items, 20, seed=42)
        r2 = _subsample(items, 20, seed=42)
        assert r1 == r2

    def test_different_seeds_give_different_samples(self):
        items = list(range(100))
        r1 = _subsample(items, 20, seed=42)
        r2 = _subsample(items, 20, seed=43)
        assert r1 != r2


# ---------------------------------------------------------------------------
# save_raw_responses
# ---------------------------------------------------------------------------

class TestSaveRawResponses:
    def test_creates_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            per_sample = [
                {"query_id": "q1", "score": 0.8, "raw_response": "0.8", "parse_failed": False},
                {"query_id": "q2", "score": None, "raw_response": "dunno", "parse_failed": True},
            ]
            out = save_raw_responses("answer", per_sample, tmp)
            assert out.exists()
            lines = out.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 2
            loaded = [json.loads(l) for l in lines]
            assert loaded[0]["query_id"] == "q1"
            assert loaded[1]["parse_failed"] is True

    def test_creates_directory_if_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "dir"
            save_raw_responses("test", [], target)
            assert target.exists()

    def test_appends_on_second_call(self):
        """Calling save_raw_responses twice accumulates records (mode 'a')."""
        with tempfile.TemporaryDirectory() as tmp:
            s1 = [{"query_id": "q1", "score": 0.8, "raw_response": "0.8", "parse_failed": False}]
            s2 = [{"query_id": "q2", "score": 0.6, "raw_response": "0.6", "parse_failed": False}]
            out = save_raw_responses("answer", s1, tmp)
            save_raw_responses("answer", s2, tmp)
            lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert len(lines) == 2, "Second call should append, not overwrite"
            ids = [json.loads(l)["query_id"] for l in lines]
            assert ids == ["q1", "q2"]

    def test_run_id_present_in_each_record(self):
        """Every record written by save_raw_responses has a 'run_id' key."""
        with tempfile.TemporaryDirectory() as tmp:
            per_sample = [
                {"query_id": "q1", "score": 0.9, "raw_response": "0.9", "parse_failed": False},
                {"query_id": "q2", "score": 0.7, "raw_response": "0.7", "parse_failed": False},
            ]
            out = save_raw_responses("faithfulness", per_sample, tmp)
            loaded = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]
            assert all("run_id" in r for r in loaded), "run_id must be in every record"

    def test_same_run_id_within_call(self):
        """All records from a single call share the same run_id."""
        with tempfile.TemporaryDirectory() as tmp:
            per_sample = [
                {"query_id": f"q{i}", "score": 0.5, "raw_response": "0.5", "parse_failed": False}
                for i in range(3)
            ]
            out = save_raw_responses("coherence", per_sample, tmp)
            loaded = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l]
            run_ids = {r["run_id"] for r in loaded}
            assert len(run_ids) == 1, "All records in one call must share the same run_id"

    def test_different_run_ids_across_calls(self):
        """Two successive calls produce different run_ids."""
        with tempfile.TemporaryDirectory() as tmp:
            s = [{"query_id": "q1", "score": 0.8, "raw_response": "0.8", "parse_failed": False}]
            out = save_raw_responses("answer", s, tmp)
            save_raw_responses("answer", s, tmp)
            lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l]
            loaded = [json.loads(l) for l in lines]
            assert loaded[0]["run_id"] != loaded[1]["run_id"]


# ---------------------------------------------------------------------------
# llm_judge_* — mocked Ollama
# ---------------------------------------------------------------------------

def _make_predictions(n=5):
    return [
        {
            "query_id": f"q{i}",
            "question": f"Soru {i}?",
            "expected": f"Beklenen {i}",
            "predicted": f"Tahmin {i}",
            "retrieved_chunks": [{"text": f"Bağlam {i}"}],
        }
        for i in range(n)
    ]


class TestLlmJudgeAnswer:
    def test_valid_responses(self):
        with patch("evaluation.llm_judge._ollama_generate", return_value="0.8"):
            result = llm_judge_answer(
                _make_predictions(5), "http://localhost:11434", "test-model",
                sample_size=5,
            )
        assert result["score"] == pytest.approx(0.8)
        assert result["parse_fail_count"] == 0
        assert result["sample_size"] == 5

    def test_all_parse_failures_return_none_score(self):
        with patch("evaluation.llm_judge._ollama_generate", return_value="not a score"):
            result = llm_judge_answer(
                _make_predictions(5), "http://localhost:11434", "test-model",
                sample_size=5,
            )
        assert result["score"] is None
        assert result["parse_fail_count"] == 5

    def test_partial_failures_excluded_from_mean(self):
        responses = iter(["0.8", "not_a_score", "0.6"])
        with patch("evaluation.llm_judge._ollama_generate", side_effect=responses):
            result = llm_judge_answer(
                _make_predictions(3), "http://localhost:11434", "test-model",
                sample_size=3,
            )
        assert result["score"] == pytest.approx(0.7)
        assert result["parse_fail_count"] == 1

    def test_saves_raw_responses_when_results_dir_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("evaluation.llm_judge._ollama_generate", return_value="0.7"):
                llm_judge_answer(
                    _make_predictions(3), "http://localhost:11434", "test-model",
                    sample_size=3, results_dir=tmp,
                )
            assert (Path(tmp) / "judge_raw_answer.jsonl").exists()

    def test_no_save_when_results_dir_none(self):
        """No JSONL should be written if results_dir is None."""
        with tempfile.TemporaryDirectory() as tmp:
            with patch("evaluation.llm_judge._ollama_generate", return_value="0.7"):
                llm_judge_answer(
                    _make_predictions(3), "http://localhost:11434", "test-model",
                    sample_size=3, results_dir=None,
                )
            # No files should have been written to tmp (it's unused)
            assert list(Path(tmp).iterdir()) == []


class TestLlmJudgeFaithfulness:
    def test_valid_responses(self):
        with patch("evaluation.llm_judge._ollama_generate", return_value="0.9"):
            result = llm_judge_faithfulness(
                _make_predictions(4), "http://localhost:11434", "test-model",
                sample_size=4,
            )
        assert result["score"] == pytest.approx(0.9)

    def test_empty_predictions(self):
        result = llm_judge_faithfulness(
            [], "http://localhost:11434", "test-model", sample_size=5,
        )
        assert result["score"] is None
        assert result["sample_size"] == 0


class TestLlmJudgeRelevancy:
    def test_valid_responses(self):
        with patch("evaluation.llm_judge._ollama_generate", return_value="0.5"):
            result = llm_judge_relevancy(
                _make_predictions(4), "http://localhost:11434", "test-model",
                sample_size=4,
            )
        assert result["score"] == pytest.approx(0.5)


class TestLlmJudgeCoherence:
    def test_valid_responses(self):
        with patch("evaluation.llm_judge._ollama_generate", return_value="0.75"):
            result = llm_judge_coherence(
                _make_predictions(4), "http://localhost:11434", "test-model",
                sample_size=4,
            )
        assert result["score"] == pytest.approx(0.75)


class TestDistinctSeeds:
    """Verify that different judge functions use different subsampling seeds.

    The identical-score bug (judge==coherence==0.2675) was caused by all
    functions using seed=42, which with temperature=0 guaranteed identical
    subsamples and hence identical mean scores when Ollama returned the same
    value for both prompt types.
    """

    def test_answer_coherence_use_different_seeds(self):
        """With a large list, answer and coherence should subsample differently."""
        preds = _make_predictions(50)
        seen_calls: dict[str, list] = {}

        def capture(prompt, *args, **kwargs):
            # Record which query_id is being judged by looking for it in the prompt
            for p in preds:
                if p["query_id"] in prompt or p["question"] in prompt:
                    seen_calls.setdefault("answer" if "Beklenen" in prompt else "coherence", []).append(p["query_id"])
                    break
            return "0.5"

        with patch("evaluation.llm_judge._ollama_generate", side_effect=capture):
            llm_judge_answer(preds, "http://x", "m", sample_size=10)
        with patch("evaluation.llm_judge._ollama_generate", side_effect=capture):
            llm_judge_coherence(preds, "http://x", "m", sample_size=10)

        answer_set = set(seen_calls.get("answer", []))
        coherence_set = set(seen_calls.get("coherence", []))
        # With different seeds, at least one query should differ between the two sets
        # (statistically guaranteed for n=50, sample=10 with seeds 42 vs 45)
        assert answer_set != coherence_set or len(answer_set) == 0
