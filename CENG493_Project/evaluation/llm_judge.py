"""
evaluation/llm_judge.py — LLM-as-Judge metrics via Ollama

Provides four scoring functions using Turkish prompts:
  - llm_judge_answer      : answer quality given question + expected
  - llm_judge_faithfulness: faithfulness of answer to retrieved context
  - llm_judge_relevancy   : relevance of answer to question
  - llm_judge_coherence   : linguistic coherence of answer

All functions accept a sample_size param (default 20) and run on a random
subsample to stay within time budgets. Returns dicts with at least a "score"
key (float 0–1).
"""

from __future__ import annotations

import random
import re
import time
from typing import Optional

import requests


# Internal helpers

def _parse_score(text: str) -> float:
    """Extract first float in [0,1] from text. Returns 0.5 on failure."""
    text = text.strip()
    # Try exact float/int
    for pattern in [r"^\s*([01](?:\.\d+)?)\s*$", r"([01](?:\.\d+)?)"]:
        m = re.search(pattern, text)
        if m:
            try:
                val = float(m.group(1))
                return max(0.0, min(1.0, val))
            except ValueError:
                pass
    return 0.5


def _ollama_generate(
    prompt: str,
    base_url: str,
    model: str,
    max_retries: int = 3,
) -> str:
    """Call Ollama /api/generate and return the response text."""
    # Normalize base_url — strip /v1 suffix if present, add /api/generate
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
            return data.get("response", "").strip()
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(1.5 * (attempt + 1))
            else:
                return "0.5"
    return "0.5"


def _subsample(items: list, sample_size: int, seed: int = 42) -> list:
    if len(items) <= sample_size:
        return items
    rng = random.Random(seed)
    return rng.sample(items, sample_size)


# Public API

def llm_judge_answer(
    predictions: list[dict],
    ollama_base_url: str,
    model: str,
    sample_size: int = 20,
) -> dict:
    """
    Judge answer quality. Each prediction must have keys:
      "question", "expected", "predicted"

    Returns:
      {"score": float, "per_sample": [{"query_id": ..., "score": float}]}
    """
    sample = _subsample(predictions, sample_size)
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
        score = _parse_score(raw)
        per_sample.append({"query_id": item.get("query_id", ""), "score": score})

    mean_score = sum(s["score"] for s in per_sample) / len(per_sample) if per_sample else 0.5
    return {"score": mean_score, "per_sample": per_sample}


def llm_judge_faithfulness(
    predictions: list[dict],
    ollama_base_url: str,
    model: str,
    sample_size: int = 20,
) -> dict:
    """
    Judge faithfulness of answer to context. Each prediction must have:
      "predicted" (answer), "retrieved_chunks" (list of dicts with "text")

    Returns:
      {"score": float, "per_sample": [...]}
    """
    sample = _subsample(predictions, sample_size)
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
        score = _parse_score(raw)
        per_sample.append({"query_id": item.get("query_id", ""), "score": score})

    mean_score = sum(s["score"] for s in per_sample) / len(per_sample) if per_sample else 0.5
    return {"score": mean_score, "per_sample": per_sample}


def llm_judge_relevancy(
    predictions: list[dict],
    ollama_base_url: str,
    model: str,
    sample_size: int = 20,
) -> dict:
    """
    Judge whether answer is relevant to question. Each prediction must have:
      "question" (or query_id), "predicted"

    Returns:
      {"score": float, "per_sample": [...]}
    """
    sample = _subsample(predictions, sample_size)
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
        score = _parse_score(raw)
        per_sample.append({"query_id": item.get("query_id", ""), "score": score})

    mean_score = sum(s["score"] for s in per_sample) / len(per_sample) if per_sample else 0.5
    return {"score": mean_score, "per_sample": per_sample}


def llm_judge_coherence(
    predictions: list[dict],
    ollama_base_url: str,
    model: str,
    sample_size: int = 20,
) -> dict:
    """
    Judge linguistic coherence of answer. Each prediction must have: "predicted"

    Returns:
      {"score": float, "per_sample": [...]}
    """
    sample = _subsample(predictions, sample_size)
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
        score = _parse_score(raw)
        per_sample.append({"query_id": item.get("query_id", ""), "score": score})

    mean_score = sum(s["score"] for s in per_sample) / len(per_sample) if per_sample else 0.5
    return {"score": mean_score, "per_sample": per_sample}
