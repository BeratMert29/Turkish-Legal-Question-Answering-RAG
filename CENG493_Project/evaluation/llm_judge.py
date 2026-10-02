"""
evaluation/llm_judge.py — LLM-as-Judge metrics via Ollama

Provides four scoring functions using Turkish prompts:
  - llm_judge_answer      : answer quality given question + expected
  - llm_judge_faithfulness: faithfulness of answer to retrieved context
  - llm_judge_relevancy   : relevance of answer to question
  - llm_judge_coherence   : linguistic coherence of answer

All functions accept a sample_size param (default from config.LLM_JUDGE_SAMPLE_SIZE)
and run on a random subsample to stay within time budgets.

Each function returns a dict with keys:
  "score"           : float mean (None-excluded), or None if all parses failed
  "per_sample"      : list of {"query_id", "score", "raw_response", "parse_failed"}
  "parse_fail_count": int number of samples where score could not be parsed
  "sample_size"     : int number of samples actually judged

Raw judge responses are saved per-stage to a JSONL file via save_raw_responses().

Bug fix: _parse_score now returns None on failure instead of 0.5, so failed
parses are excluded from the mean rather than biasing it toward 0.5.

Identical-score investigation: base stage had judge==coherence==0.2675 exactly.
Root cause: with temperature=0.0 the judge LLM is deterministic; _subsample used
seed=42 for every function, so all four metrics sampled the same 20 items.
When Ollama is unavailable or the model returns unparseable text (e.g. long
reasoning before the score), _parse_score used to silently return 0.5 for all
samples, and partial failures (some real scores + some 0.5 fallbacks) could
accidentally produce the same mean across two metrics if the failure pattern
was identical.  Fix: return None on failure + exclude from mean + use
per-function seed offsets so subsamples differ across metrics.
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

# Default sample size — can be overridden by config.LLM_JUDGE_SAMPLE_SIZE
_DEFAULT_SAMPLE_SIZE: int = 20
try:
    import config as _cfg
    _DEFAULT_SAMPLE_SIZE = getattr(_cfg, "LLM_JUDGE_SAMPLE_SIZE", 20)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_score(text: str) -> Optional[float]:
    """Extract a score in [0, 1] from LLM judge response text.

    Handles N/10, N/5, and direct floats.

    Returns:
        Float score clamped to [0, 1], or **None** if the response cannot be
        parsed.  Callers must treat None as a failed parse and exclude it from
        aggregate statistics rather than substituting a default value.
    """
    if not text:
        return None
    # Decimal comma ("0,8") -> decimal point
    text = re.sub(r'(?<=\d),(?=\d)', '.', text.strip())

    # 1. Exact standalone float in [0,1] (e.g. "0.7", "1", "0.85")
    m = re.match(r'^([01](?:\.\d+)?)\s*$', text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))

    # 2. N/D formats (N/10, N/5, N/1): the denominator is the scale
    m = re.search(r'(?<![\d.])(\d+(?:\.\d+)?)\s*/\s*(10|5|1)(?!\d)', text)
    if m:
        return max(0.0, min(1.0, float(m.group(1)) / float(m.group(2))))

    # 3. Standalone decimal in [0,1] anywhere in text
    m = re.search(r'(?<![\d/.])([01]\.\d+)(?!\d)(?!\s*/)', text)
    if m:
        return max(0.0, min(1.0, float(m.group(1))))

    # 4. Standalone "0" or "1" not part of a larger number
    m = re.search(r'(?<![\d.])([01])(?![\d/]|\.\d)', text)
    if m:
        return float(m.group(1))

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
        "options": {"temperature": 0.0, "num_predict": 16},
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
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


def _subsample(items: list, sample_size: int, seed: int = 42) -> list:
    """Return a deterministic random subsample of *items*."""
    if len(items) <= sample_size:
        return items
    rng = random.Random(seed)
    return rng.sample(items, sample_size)


def save_raw_responses(
    metric_name: str,
    per_sample: list[dict],
    results_dir: "str | Path",
) -> Path:
    """Append raw judge responses for *metric_name* to a JSONL file in *results_dir*.

    Args:
        metric_name: Short identifier, e.g. "answer", "faithfulness".
        per_sample:  List of dicts as returned by each llm_judge_* function.
        results_dir: Directory where the JSONL is written.

    Returns:
        Path to the written file.
    """
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"judge_raw_{metric_name}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for rec in per_sample:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.debug("LLM judge raw responses saved to %s", out_path)
    return out_path


def _aggregate(per_sample: list[dict]) -> dict:
    """Compute mean score excluding None/failed parses and return summary dict."""
    valid = [s["score"] for s in per_sample if s["score"] is not None]
    fail_count = sum(1 for s in per_sample if s.get("parse_failed", False) or s["score"] is None)
    mean_score: Optional[float] = sum(valid) / len(valid) if valid else None
    return {
        "score": mean_score,
        "per_sample": per_sample,
        "parse_fail_count": fail_count,
        "sample_size": len(per_sample),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def llm_judge_answer(
    predictions: list[dict],
    ollama_base_url: str,
    model: str,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    results_dir: "str | Path | None" = None,
) -> dict:
    """Judge answer quality.

    Each prediction must have keys: "question", "expected", "predicted".

    Args:
        predictions:    List of prediction dicts.
        ollama_base_url: Base URL for the Ollama API.
        model:          Ollama model name.
        sample_size:    Maximum number of samples to judge (default from config).
        results_dir:    If provided, raw responses are saved to this directory.

    Returns:
        {"score": float|None, "per_sample": [...], "parse_fail_count": int,
         "sample_size": int}
    """
    # seed=42 for answer quality; distinct seed from coherence to avoid
    # accidentally identical subsamples across metrics (see module docstring).
    sample = _subsample(predictions, sample_size, seed=42)
    per_sample = []

    for item in sample:
        question  = item.get("question",  item.get("query_id", ""))
        expected  = item.get("expected",  "")
        predicted = item.get("predicted", "")

        prompt = (
            f"Soru: {question}\n"
            f"Beklenen Cevap: {expected}\n"
            f"Verilen Cevap: {predicted}\n\n"
            "Verilen cevabın kalitesini 0 ile 1 arasında bir sayı ile değerlendir.\n"
            "1 = mükemmel cevap, 0 = tamamen yanlış.\n"
            "Sadece sayıyı yaz, başka hiçbir şey yazma."
        )

        raw = _ollama_generate(prompt, ollama_base_url, model)
        score = None if raw is None else _parse_score(raw)
        per_sample.append({
            "query_id": item.get("query_id", ""),
            "score": score,
            "raw_response": raw,
            "parse_failed": score is None,
        })

    result = _aggregate(per_sample)
    if results_dir is not None:
        save_raw_responses("answer", per_sample, results_dir)
    return result


def llm_judge_faithfulness(
    predictions: list[dict],
    ollama_base_url: str,
    model: str,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    results_dir: "str | Path | None" = None,
) -> dict:
    """Judge faithfulness of answer to context.

    Each prediction must have: "predicted" (answer), "retrieved_chunks"
    (list of dicts with "text").

    Args:
        predictions:    List of prediction dicts.
        ollama_base_url: Base URL for the Ollama API.
        model:          Ollama model name.
        sample_size:    Maximum number of samples to judge.
        results_dir:    If provided, raw responses are saved to this directory.

    Returns:
        {"score": float|None, "per_sample": [...], "parse_fail_count": int,
         "sample_size": int}
    """
    sample = _subsample(predictions, sample_size, seed=43)
    per_sample = []

    for item in sample:
        answer  = item.get("predicted", "")
        chunks  = item.get("retrieved_chunks", [])
        context = "\n\n".join(c.get("text", "") for c in chunks[:5])

        prompt = (
            f"Bağlam:\n{context}\n\n"
            f"Cevap: {answer}\n\n"
            "Cevap yalnızca bağlamdaki bilgilere dayanıyor mu? "
            "0 ile 1 arasında bir sayı ile değerlendir.\n"
            "1 = tamamen sadık, 0 = tamamen uydurulmuş.\n"
            "Sadece sayıyı yaz, başka hiçbir şey yazma."
        )

        raw   = _ollama_generate(prompt, ollama_base_url, model)
        score = None if raw is None else _parse_score(raw)
        per_sample.append({
            "query_id": item.get("query_id", ""),
            "score": score,
            "raw_response": raw,
            "parse_failed": score is None,
        })

    result = _aggregate(per_sample)
    if results_dir is not None:
        save_raw_responses("faithfulness", per_sample, results_dir)
    return result


def llm_judge_relevancy(
    predictions: list[dict],
    ollama_base_url: str,
    model: str,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    results_dir: "str | Path | None" = None,
) -> dict:
    """Judge whether answer is relevant to question.

    Each prediction must have: "question" (or query_id), "predicted".

    Args:
        predictions:    List of prediction dicts.
        ollama_base_url: Base URL for the Ollama API.
        model:          Ollama model name.
        sample_size:    Maximum number of samples to judge.
        results_dir:    If provided, raw responses are saved to this directory.

    Returns:
        {"score": float|None, "per_sample": [...], "parse_fail_count": int,
         "sample_size": int}
    """
    sample = _subsample(predictions, sample_size, seed=44)
    per_sample = []

    for item in sample:
        question = item.get("question", item.get("query_id", ""))
        answer   = item.get("predicted", "")

        prompt = (
            f"Soru: {question}\n"
            f"Cevap: {answer}\n\n"
            "Cevap soruyla ne kadar ilgili? 0 ile 1 arasında bir sayı ile değerlendir.\n"
            "1 = tamamen ilgili, 0 = tamamen alakasız.\n"
            "Sadece sayıyı yaz, başka hiçbir şey yazma."
        )

        raw   = _ollama_generate(prompt, ollama_base_url, model)
        score = None if raw is None else _parse_score(raw)
        per_sample.append({
            "query_id": item.get("query_id", ""),
            "score": score,
            "raw_response": raw,
            "parse_failed": score is None,
        })

    result = _aggregate(per_sample)
    if results_dir is not None:
        save_raw_responses("relevancy", per_sample, results_dir)
    return result


def llm_judge_coherence(
    predictions: list[dict],
    ollama_base_url: str,
    model: str,
    sample_size: int = _DEFAULT_SAMPLE_SIZE,
    results_dir: "str | Path | None" = None,
) -> dict:
    """Judge linguistic coherence of answer.

    Each prediction must have: "predicted".

    Args:
        predictions:    List of prediction dicts.
        ollama_base_url: Base URL for the Ollama API.
        model:          Ollama model name.
        sample_size:    Maximum number of samples to judge.
        results_dir:    If provided, raw responses are saved to this directory.

    Returns:
        {"score": float|None, "per_sample": [...], "parse_fail_count": int,
         "sample_size": int}
    """
    # seed=45 — distinct from answer (42), faithfulness (43), relevancy (44) so
    # the four metrics never accidentally sample the same subset from predictions,
    # which was the root cause of the identical-score issue (judge==coherence==0.2675).
    sample = _subsample(predictions, sample_size, seed=45)
    per_sample = []

    for item in sample:
        answer = item.get("predicted", "")

        prompt = (
            f"Cevap: {answer}\n\n"
            "Bu cevap dil bilgisi açısından doğru ve anlaşılır mı? "
            "0 ile 1 arasında bir sayı ile değerlendir.\n"
            "1 = tamamen tutarlı ve anlaşılır, 0 = anlamsız veya tutarsız.\n"
            "Sadece sayıyı yaz, başka hiçbir şey yazma."
        )

        raw   = _ollama_generate(prompt, ollama_base_url, model)
        score = None if raw is None else _parse_score(raw)
        per_sample.append({
            "query_id": item.get("query_id", ""),
            "score": score,
            "raw_response": raw,
            "parse_failed": score is None,
        })

    result = _aggregate(per_sample)
    if results_dir is not None:
        save_raw_responses("coherence", per_sample, results_dir)
    return result
