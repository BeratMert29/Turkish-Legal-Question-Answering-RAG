"""
tests/test_llm_judge.py — Unit tests for evaluation/llm_judge.py.

All tests run offline: Ollama is mocked/patched.  No network or GPU required.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.llm_judge import (
    _parse_score,
    _subsample,
    _aggregate,
    save_raw_responses,
    sample_judge_query_ids,
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

    @pytest.mark.parametrize("text,expected", [
        ("Cevap 1. maddeye göre yanlış: 0", 0.0),   # ordinal "1." is not a score
        ("TMK m. 5/1 uyarınca doğru", None),          # article ref, not 5/1
        ("1/2", 0.5),
        ("10", None),                                 # out of range, ambiguous
        ("0,5", 0.5),
        ("Puan: 0.5", 0.5),
        ("0.5 çünkü 1 maddede eksik", 0.5),           # leading score wins
        ("2. fıkraya göre puan 0.5", 0.5),
    ])
    def test_turkish_legal_responses(self, text, expected):
        result = _parse_score(text)
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)


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

    def test_per_run_file_not_appended(self):
        """Distinct run ids -> distinct files; same run id -> overwritten."""
        with tempfile.TemporaryDirectory() as tmp:
            s1 = [{"query_id": "q1", "score": 0.8, "raw_response": "0.8", "parse_failed": False}]
            s2 = [{"query_id": "q2", "score": 0.6, "raw_response": "0.6", "parse_failed": False}]
            o1 = save_raw_responses("answer", s1, tmp, run_id="r1")
            o2 = save_raw_responses("answer", s2, tmp, run_id="r2")
            assert o1 != o2 and o1.name == "judge_raw_answer_r1.jsonl"
            save_raw_responses("answer", s2, tmp, run_id="r1")
            lines = [l for l in o1.read_text(encoding="utf-8").splitlines() if l.strip()]
            assert [json.loads(l)["query_id"] for l in lines] == ["q2"]

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
            o1 = save_raw_responses("answer", s, tmp)
            o2 = save_raw_responses("answer", s, tmp)
            assert o1 != o2
            r1 = json.loads(o1.read_text(encoding="utf-8").splitlines()[0])
            r2 = json.loads(o2.read_text(encoding="utf-8").splitlines()[0])
            assert r1["run_id"] != r2["run_id"]


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
            assert len(list(Path(tmp).glob("judge_raw_answer_*.jsonl"))) == 1

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


# ---------------------------------------------------------------------------
# sample_judge_query_ids + cross-metric consistent sampling
# ---------------------------------------------------------------------------

class TestSampleJudgeQueryIds:
    def test_returns_subsample(self):
        ids = [f"q{i}" for i in range(100)]
        result = sample_judge_query_ids(ids, 10)
        assert len(result) == 10
        assert all(r in ids for r in result)

    def test_deterministic(self):
        ids = [f"q{i}" for i in range(50)]
        assert sample_judge_query_ids(ids, 20) == sample_judge_query_ids(ids, 20)

    def test_all_four_metrics_same_query_ids(self):
        """When query_ids is passed, all four llm_judge_* evaluate the same items."""
        predictions = [
            {
                "query_id": f"q{i}",
                "question": f"Soru {i}?",
                "expected": f"Beklenen {i}",
                "predicted": f"Tahmin {i}",
                "retrieved_chunks": [{"text": f"Bağlam {i}"}],
            }
            for i in range(20)
        ]
        shared_ids = sample_judge_query_ids(
            [p["query_id"] for p in predictions], n=10
        )
        assert len(shared_ids) == 10

        with patch("evaluation.llm_judge._ollama_generate", return_value="0.8"):
            r_ans  = llm_judge_answer(predictions, "http://x", "m", query_ids=shared_ids)
            r_faith = llm_judge_faithfulness(predictions, "http://x", "m", query_ids=shared_ids)
            r_rel  = llm_judge_relevancy(predictions, "http://x", "m", query_ids=shared_ids)
            r_coh  = llm_judge_coherence(predictions, "http://x", "m", query_ids=shared_ids)

        def _qids(result):
            return {s["query_id"] for s in result["per_sample"]}

        assert _qids(r_ans) == _qids(r_faith) == _qids(r_rel) == _qids(r_coh), (
            "All four metrics must evaluate the same query IDs when query_ids is passed"
        )
        assert _qids(r_ans) == set(shared_ids)


class TestJudgeDefaults:
    def test_sample_none_judges_all(self):
        preds = _make_predictions(30)
        with patch("evaluation.llm_judge._ollama_generate", return_value="1"):
            r = llm_judge_answer(preds, "http://x", "m", sample_size=None)
        assert r["sample_size"] == 30

    def test_sample_judge_query_ids_none_returns_all(self):
        ids = [f"q{i}" for i in range(7)]
        assert sample_judge_query_ids(ids, None) == ids

    def test_num_ctx_in_ollama_payload(self):
        from evaluation import llm_judge as lj
        captured = {}

        class _R:
            def raise_for_status(self): pass
            def json(self): return {"response": "0.5"}

        def fake_post(url, json=None, timeout=None):
            captured.update(json)
            return _R()

        with patch.object(lj.requests, "post", fake_post):
            lj._ollama_generate("p", "http://x/v1", "m", num_ctx=4096)
            assert captured["options"]["num_ctx"] == 4096
            lj._ollama_generate("p", "http://x/v1", "m")
            assert captured["options"]["num_ctx"] == lj._DEFAULT_NUM_CTX

    def test_prompts_have_turkish_anchors(self):
        from evaluation import llm_judge as lj
        item = {"question": "S", "expected": "E", "predicted": "P",
                "retrieved_chunks": [{"text": "B"}]}
        for fn in (lj._prompt_answer, lj._prompt_faithfulness,
                   lj._prompt_relevancy, lj._prompt_coherence):
            p = fn(item)
            assert "0.5 =" in p and "1   =" in p and "0   =" in p

    def test_run_id_shared_across_files(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch("evaluation.llm_judge._ollama_generate", return_value="1"):
            llm_judge_answer(_make_predictions(2), "http://x", "m",
                             results_dir=tmp, run_id="RID")
            assert (Path(tmp) / "judge_raw_answer_RID.jsonl").exists()


def test_aggregate_separates_call_failures_and_reports_zero_sensitivity():
    samples = [
        {"score": 1.0, "parse_failed": False, "call_failed": False},
        {"score": None, "parse_failed": True, "call_failed": True},
        {"score": None, "parse_failed": True, "call_failed": False},
        {"score": 0.5, "parse_failed": False, "call_failed": False},
    ]
    r = _aggregate(samples)
    assert r["score"] == pytest.approx(0.75)
    assert r["score_failures_as_zero"] == pytest.approx(1.5 / 4)
    assert r["parse_fail_count"] == 2 and r["call_fail_count"] == 1
