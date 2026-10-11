"""generation/rag_pipeline.py: native Ollama chat request and runaway cut."""

from unittest.mock import MagicMock, patch

import pytest
import requests as requests_lib

import config
from generation.rag_pipeline import RAGPipeline, cut_runaway


def test_cut_runaway_removes_invented_follow_up_turn():
    text = "Süre beş gündür [Kaynak 1].\n\nSoru: Peki itiraz?\nCevap: ..."
    assert cut_runaway(text) == ("Süre beş gündür [Kaynak 1].", True)
    assert cut_runaway("Kısa cevap.") == ("Kısa cevap.", False)
    # a marker in the first 10 characters is not a runaway turn
    assert cut_runaway("Soru: x ise y.")[1] is False


def _pipe(short=False):
    return RAGPipeline(retriever=None, model="m", short_answer_mode=short)


def test_generate_sends_same_options_for_every_model():
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": "Süre beş gündür.\nSoru: ek"},
                              "done_reason": "length", "eval_count": 512}
    with patch("requests.post", return_value=resp) as post:
        out = _pipe().generate("S?", "ctx")
    url, = post.call_args.args
    body = post.call_args.kwargs["json"]
    assert url.endswith("/api/chat") and body["stream"] is False
    opts = body["options"]
    assert opts["num_ctx"] == config.LLM_NUM_CTX
    assert opts["num_predict"] == config.LLM_MAX_TOKENS
    assert opts["seed"] == config.SEED and opts["stop"] == config.LLM_STOP
    assert body["messages"][1]["content"] == "Bağlam:\nctx\n\nSoru: S?"
    assert out == "Süre beş gündür."


def test_generate_records_truncation_and_runaway():
    p = _pipe()
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": "Uzun cevap\nSoru: x"},
                              "done_reason": "length", "eval_count": 512}
    with patch("requests.post", return_value=resp):
        p.generate("q", "c")
    assert p.last_meta["done_reason"] == "length" and p.last_meta["runaway_cut"]


def test_generate_empty_response_raises():
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": ""}}
    with patch("requests.post", return_value=resp), pytest.raises(ValueError):
        _pipe().generate("q", "c")


def test_qa_metrics_report_truncation_rates():
    from evaluation.qa_metrics import compute_all_qa_metrics_with_citation
    preds = [{"predicted": "a", "expected": "a", "truncated": True, "runaway_cut": False},
             {"predicted": "b", "expected": "b", "truncated": False, "runaway_cut": False}]
    r = compute_all_qa_metrics_with_citation(preds)
    assert r["truncated_rate"] == 0.5 and r["runaway_cut_rate"] == 0.0


# ---------------------------------------------------------------------------
# _chat retry behaviour
# ---------------------------------------------------------------------------

def _make_http_error(status_code: int) -> requests_lib.HTTPError:
    """Build a requests.HTTPError with a fake response for the given status."""
    resp = MagicMock()
    resp.status_code = status_code
    err = requests_lib.HTTPError(response=resp)
    return err


def _ok_resp() -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = {"message": {"content": "Cevap."}, "done_reason": "stop", "eval_count": 10}
    return resp


def test_chat_retries_on_5xx_and_eventually_succeeds():
    """5xx triggers a retry; succeeds on the second attempt."""
    pipe = _pipe()
    ok = _ok_resp()
    side_effects = [_make_http_error(503), ok]

    def _post_side(*args, **kwargs):
        effect = side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect

    with patch("requests.post", side_effect=_post_side) as mock_post, \
         patch("time.sleep") as mock_sleep:
        result = pipe.generate("q", "c")

    assert result == "Cevap."
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


def test_chat_does_not_retry_on_4xx():
    """4xx re-raises immediately without retry or sleep."""
    pipe = _pipe()

    with patch("requests.post", side_effect=_make_http_error(404)) as mock_post, \
         patch("time.sleep") as mock_sleep:
        with pytest.raises(requests_lib.HTTPError):
            pipe.generate("q", "c")

    assert mock_post.call_count == 1
    mock_sleep.assert_not_called()


def test_chat_raises_runtime_error_after_all_5xx_retries_exhausted():
    """All attempts return 5xx → RuntimeError with attempt count."""
    pipe = _pipe()

    with patch("requests.post", side_effect=_make_http_error(500)), \
         patch("time.sleep"):
        with pytest.raises(RuntimeError, match=str(config.LLM_MAX_RETRIES)):
            pipe.generate("q", "c")


def test_chat_retries_on_connection_error():
    """ConnectionError still retries (pre-existing behaviour not regressed)."""
    pipe = _pipe()
    ok = _ok_resp()
    side_effects = [requests_lib.ConnectionError("refused"), ok]

    def _post_side(*args, **kwargs):
        effect = side_effects.pop(0)
        if isinstance(effect, Exception):
            raise effect
        return effect

    with patch("requests.post", side_effect=_post_side) as mock_post, \
         patch("time.sleep"):
        result = pipe.generate("q", "c")

    assert result == "Cevap."
    assert mock_post.call_count == 2
