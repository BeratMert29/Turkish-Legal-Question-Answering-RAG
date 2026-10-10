"""
evaluation/llm_judge.py — LLM-as-Judge metrics via Ollama

Provides four scoring functions using Turkish prompts:
  - llm_judge_answer      : answer quality given question + expected
  - llm_judge_faithfulness: faithfulness of answer to retrieved context
  - llm_judge_relevancy   : relevance of answer to question
  - llm_judge_coherence   : linguistic coherence of answer

All functions accept a sample_size param (default config.LLM_JUDGE_SAMPLE_SIZE;
None = judge every prediction). Rubrics are Turkish with explicit 0 / 0.5 / 1
anchors; Ollama is called with an explicit num_ctx.

Each function returns a dict with keys:
  "score"           : float mean (None-excluded), or None if all parses failed
  "per_sample"      : list of {"query_id", "score", "raw_response", "parse_failed"}
  "parse_fail_count": int number of samples where score could not be parsed
  "sample_size"     : int number of samples actually judged

Raw judge responses are saved per run to judge_raw_<metric>_<run_id>.jsonl via
save_raw_responses() (one file per run, never appended across runs).

_parse_score returns None on failure (never 0.5), so failed parses are
excluded from the mean; ``score_failures_as_zero`` reports the sensitivity
variant and ``call_fail_count`` separates dead Ollama calls from bad parses.

Identical-score investigation: base stage had judge==coherence==0.2675 exactly.
Root cause: with temperature=0.0 the judge LLM is deterministic; _subsample
originally used seed=42 for every function, so all four metrics sampled the
same 20 items.  When Ollama is unavailable or the model returns unparseable
text (e.g. long reasoning before the score), _parse_score used to silently
return 0.5 for all samples, and partial failures (some real scores + some 0.5
fallbacks) could accidentally produce the same mean across two metrics if the
failure pattern was identical.

Fix: return None on failure + exclude from mean + use per-function seed offsets
(42/43/44/45) so independent subsamples differ across metrics by default.

For intentional cross-metric comparison on the same items, call
``sample_judge_query_ids`` once and pass the result to all four functions via
their ``query_ids`` parameter; per-function sampling is then bypassed.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Default sample size: None = judge all predictions (config.LLM_JUDGE_SAMPLE_SIZE).
_DEFAULT_SAMPLE_SIZE: Optional[int] = None
_DEFAULT_NUM_CTX: int = 8192
try:
    import config as _cfg
    _DEFAULT_SAMPLE_SIZE = getattr(_cfg, "LLM_JUDGE_SAMPLE_SIZE", None)
    _DEFAULT_NUM_CTX = getattr(_cfg, "LLM_JUDGE_NUM_CTX", 8192)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NUM = r"(?:\d+(?:\.\d+)?|\.\d+)"
# A score candidate: a fraction N/D or a bare number, not glued to other digits,
# a slash (article refs like "5/1") or a following ".digit".
_FRACTION_RE = re.compile(r"(?<![\d./])(" + _NUM + r")\s*/\s*(\d+)(?![\d/])")
_NUMBER_RE = re.compile(r"(?<![\d./])(" + _NUM + r")(?![\d/]|\.\d|\s*/)")
_KEYWORD_RE = re.compile(r"(?:puan|skor|score|not)\w*\s*[:=]?\s*", re.IGNORECASE)
_ORDINAL_AFTER = re.compile(r"\.\s*[a-zçğıöşü]")  # "1. maddeye", "2. fıkra"


def _candidate(text: str, m: "re.Match", fraction: bool) -> Optional[float]:
    """Score in [0, 1] for a matched number/fraction, or None if implausible."""
    if fraction:
        num, den = float(m.group(1)), float(m.group(2))
        if den == 0 or den not in (1, 2, 4, 5, 10, 100):
            return None
        if den == 1 and num > 1:          # "m. 5/1" is an article, not a score
            return None
        return max(0.0, min(1.0, num / den))
    if _ORDINAL_AFTER.match(text, m.end()):  # Turkish ordinal "1. madde"
        return None
    val = float(m.group(1))
    return val if 0.0 <= val <= 1.0 else None


def _parse_score(text: str) -> Optional[float]:
    """Extract a score in [0, 1] from LLM judge response text.

    The rubric asks for 0 / 0.5 / 1; N/D fractions (N/10, N/5, 1/2) are also
    accepted.  Order of preference:
      1. a score at the very start of the response (optionally after
         "Puan:"), unless it is an ordinal such as "1. maddeye";
      2. a number after a "puan/skor/score" keyword;
      3. the last plausible score in the text (explanations usually end with
         the verdict).
    Article references ("m. 5/1"), years and other out-of-range numbers are
    never read as scores.

    Returns:
        Float score clamped to [0, 1], or **None** if the response cannot be
        parsed.  Callers must treat None as a failed parse and exclude it from
        aggregate statistics rather than substituting a default value.
    """
    if not text:
        return None
    # Decimal comma ("0,8") -> decimal point
    text = re.sub(r'(?<=\d),(?=\d)', '.', text.strip())

    def _at(pos: int) -> Optional[float]:
        for rx, frac in ((_FRACTION_RE, True), (_NUMBER_RE, False)):
            m = rx.match(text, pos)
            if m:
                return _candidate(text, m, frac)
        return None

    lead = re.match(r"\s*(?:(?:puan|skor|score)\w*\s*[:=]?\s*)?", text, re.IGNORECASE)
    val = _at(lead.end())
    if val is not None:
        return val
    for km in _KEYWORD_RE.finditer(text):
        val = _at(km.end())
        if val is not None:
            return val

    found: list[tuple[int, float]] = []
    for rx, frac in ((_FRACTION_RE, True), (_NUMBER_RE, False)):
        for m in rx.finditer(text):
            v = _candidate(text, m, frac)
            if v is not None:
                found.append((m.start(), v))
    if found:
        return max(found)[1]

    logger.warning(
        "LLM judge _parse_score: could not parse score from response: %r",
        text[:120],
    )
    return None


def _ollama_generate(
    prompt: str,
    base_url: str,
    model: str,
    max_retries: int = 3,
    num_ctx: Optional[int] = None,
) -> Optional[str]:
    """Call Ollama /api/generate and return the response text.

    Returns **None** on unrecoverable error (or an empty response after
    retries) so callers can count it as a failure and exclude it from means.
    """
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    url = f"{base}/api/generate"

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 16,
            "num_ctx": int(num_ctx or _DEFAULT_NUM_CTX),
        },
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response", "").strip()
            if text:
                return text
            raise ValueError("empty response from judge")
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                logger.warning(
                    "_ollama_generate failed after %d retries: %s", max_retries, exc
                )
                return None
    return None


def _subsample(items: list, sample_size: Optional[int], seed: int = 42) -> list:
    """Deterministic random subsample; sample_size None (or >= len) keeps all."""
    if sample_size is None or len(items) <= sample_size:
        return items
    rng = random.Random(seed)
    return rng.sample(items, sample_size)


def sample_judge_query_ids(
    query_ids: "list[str]",
    n: Optional[int],
    seed: int = 42,
) -> "list[str]":
    """Sample *n* query IDs for consistent cross-metric judging (and NLI).

    ``n=None`` returns every ID. Call once and pass the result to all four
    ``llm_judge_*`` functions (and to the NLI scorer) via ``query_ids`` so
    every metric is evaluated on an identical subset.
    """
    return _subsample(query_ids, n, seed=seed)


def new_run_id() -> str:
    """Timestamped unique id for naming per-run raw judge files."""
    import uuid as _uuid
    return time.strftime("%Y%m%dT%H%M%S") + "-" + _uuid.uuid4().hex[:8]


def save_raw_responses(
    metric_name: str,
    per_sample: list[dict],
    results_dir: "str | Path",
    run_id: Optional[str] = None,
) -> Path:
    """Write raw judge responses to ``judge_raw_<metric>_<run_id>.jsonl``.

    One file per run (the file is written fresh, not appended), so reruns
    never mix with earlier runs. A ``run_id`` key is added to every record.
    """
    run_id = run_id or new_run_id()
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judge_raw_{metric_name}_{run_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in per_sample:
            fh.write(json.dumps({**rec, "run_id": run_id}, ensure_ascii=False) + "\n")
    logger.debug("LLM judge raw responses written to %s", out_path)
    return out_path


def _aggregate(per_sample: list[dict]) -> dict:
    """Compute mean score excluding None/failed parses and return summary dict.

    ``parse_fail_count`` counts every sample without a score (kept for the
    failure-rate check); ``call_fail_count`` is the subset where the Ollama
    call itself failed, so a dead judge is not mistaken for parse noise.
    ``score_failures_as_zero`` is the sensitivity variant that counts every
    missing score as 0 instead of dropping it.
    """
    valid = [s["score"] for s in per_sample if s["score"] is not None]
    fail_count = sum(1 for s in per_sample if s.get("parse_failed", False) or s["score"] is None)
    call_fails = sum(1 for s in per_sample if s.get("call_failed", False))
    mean_score: Optional[float] = sum(valid) / len(valid) if valid else None
    return {
        "score": mean_score,
        "score_failures_as_zero": sum(valid) / len(per_sample) if per_sample else None,
        "per_sample": per_sample,
        "parse_fail_count": fail_count,
        "call_fail_count": call_fails,
        "sample_size": len(per_sample),
    }


_ANSWER_ONLY = "Sadece sayıyı yaz (0, 0.5 veya 1); başka hiçbir şey yazma."


def _prompt_answer(item: dict) -> str:
    question = item.get("question", item.get("query_id", ""))
    return (
        "Bir Türk hukuku sorusuna verilen cevabı, referans cevaba göre değerlendir.\n\n"
        f"Soru: {question}\n"
        f"Beklenen Cevap: {item.get('expected', '')}\n"
        f"Verilen Cevap: {item.get('predicted', '')}\n\n"
        "Puanlama:\n"
        "1   = Verilen cevap referans cevapla aynı hukuki sonuca varıyor ve temel bilgileri içeriyor.\n"
        "0.5 = Kısmen doğru: doğru bilgi var ama eksik, belirsiz veya bir kısmı yanlış.\n"
        "0   = Yanlış, referans cevapla çelişen veya soruyla ilgisiz.\n"
        + _ANSWER_ONLY
    )


def _prompt_faithfulness(item: dict) -> str:
    chunks = item.get("retrieved_chunks", [])
    context = "\n\n".join(c.get("text", "") for c in chunks[:5])
    return (
        "Cevabın yalnızca verilen bağlamdaki bilgilere dayanıp dayanmadığını değerlendir.\n\n"
        f"Bağlam:\n{context}\n\n"
        f"Cevap: {item.get('predicted', '')}\n\n"
        "Puanlama:\n"
        "1   = Cevaptaki tüm iddialar bağlamda yer alıyor veya bağlamdan doğrudan çıkarılabiliyor.\n"
        "0.5 = Bazı iddialar bağlamda var, bazıları bağlamda bulunmuyor.\n"
        "0   = İddiaların çoğu bağlamda yok veya bağlamla çelişiyor.\n"
        + _ANSWER_ONLY
    )


def _prompt_relevancy(item: dict) -> str:
    question = item.get("question", item.get("query_id", ""))
    return (
        "Cevabın soruyla ne kadar ilgili olduğunu değerlendir.\n\n"
        f"Soru: {question}\n"
        f"Cevap: {item.get('predicted', '')}\n\n"
        "Puanlama:\n"
        "1   = Cevap doğrudan sorulan soruyu yanıtlıyor.\n"
        "0.5 = Cevap konuyla ilgili ama sorulan noktayı tam yanıtlamıyor.\n"
        "0   = Cevap soruyla alakasız.\n"
        + _ANSWER_ONLY
    )


def _prompt_coherence(item: dict) -> str:
    return (
        "Cevabın Türkçe dil bilgisi ve anlaşılırlık açısından tutarlılığını değerlendir.\n\n"
        f"Cevap: {item.get('predicted', '')}\n\n"
        "Puanlama:\n"
        "1   = Akıcı, dil bilgisi doğru, kendi içinde tutarlı ve anlaşılır.\n"
        "0.5 = Anlaşılır ama hatalı veya kopuk ifadeler içeriyor.\n"
        "0   = Anlamsız, tekrarlı veya kendi içinde çelişkili.\n"
        + _ANSWER_ONLY
    )


def _run_metric(
    name: str,
    seed: int,
    prompt_fn,
    predictions: list[dict],
    ollama_base_url: str,
    model: str,
    sample_size: Optional[int],
    results_dir,
    query_ids,
    num_ctx: Optional[int],
    run_id: Optional[str],
) -> dict:
    if query_ids is not None:
        qid_set = set(query_ids)
        sample = [p for p in predictions if p.get("query_id") in qid_set]
    else:
        # Distinct seed per metric so independent subsamples differ unless a
        # shared ``query_ids`` list is passed.
        sample = _subsample(predictions, sample_size, seed=seed)
    per_sample = []
    for item in sample:
        raw = _ollama_generate(prompt_fn(item), ollama_base_url, model, num_ctx=num_ctx)
        score = None if raw is None else _parse_score(raw)
        per_sample.append({
            "query_id": item.get("query_id", ""),
            "score": score,
            "raw_response": raw,
            "parse_failed": score is None,
            "call_failed": raw is None,
        })
    result = _aggregate(per_sample)
    if results_dir is not None:
        save_raw_responses(name, per_sample, results_dir, run_id=run_id)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
# Each function: (predictions, ollama_base_url, model, sample_size=config default
# (None = all), results_dir=None, query_ids=None, num_ctx=None, run_id=None)
# -> {"score": float|None, "per_sample": [...], "parse_fail_count": int,
#     "sample_size": int}

def llm_judge_answer(predictions, ollama_base_url, model,
                     sample_size=_DEFAULT_SAMPLE_SIZE, results_dir=None,
                     query_ids=None, num_ctx=None, run_id=None) -> dict:
    """Answer quality vs expected answer (needs question, expected, predicted)."""
    return _run_metric("answer", 42, _prompt_answer, predictions, ollama_base_url,
                       model, sample_size, results_dir, query_ids, num_ctx, run_id)


def llm_judge_faithfulness(predictions, ollama_base_url, model,
                           sample_size=_DEFAULT_SAMPLE_SIZE, results_dir=None,
                           query_ids=None, num_ctx=None, run_id=None) -> dict:
    """Faithfulness to retrieved context (needs predicted, retrieved_chunks)."""
    return _run_metric("faithfulness", 43, _prompt_faithfulness, predictions,
                       ollama_base_url, model, sample_size, results_dir, query_ids,
                       num_ctx, run_id)


def llm_judge_relevancy(predictions, ollama_base_url, model,
                        sample_size=_DEFAULT_SAMPLE_SIZE, results_dir=None,
                        query_ids=None, num_ctx=None, run_id=None) -> dict:
    """Relevance of answer to question (needs question, predicted)."""
    return _run_metric("relevancy", 44, _prompt_relevancy, predictions,
                       ollama_base_url, model, sample_size, results_dir, query_ids,
                       num_ctx, run_id)


def llm_judge_coherence(predictions, ollama_base_url, model,
                        sample_size=_DEFAULT_SAMPLE_SIZE, results_dir=None,
                        query_ids=None, num_ctx=None, run_id=None) -> dict:
    """Linguistic coherence of answer (needs predicted)."""
    return _run_metric("coherence", 45, _prompt_coherence, predictions,
                       ollama_base_url, model, sample_size, results_dir, query_ids,
                       num_ctx, run_id)
