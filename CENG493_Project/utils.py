"""Shared utilities for the Turkish Legal RAG pipeline."""
import json
import logging
import pathlib
import re
import unicodedata
import random
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)


def read_jsonl(
    path: "str | pathlib.Path",
    *,
    on_error: str = "warn",
) -> "Iterator[dict]":
    """Yield parsed dicts from a JSONL file, skipping blank lines.

    Args:
        path:     Path to the JSONL file.
        on_error: ``"warn"`` (default) — skip bad lines and emit a
                  ``logging.WARNING`` naming the file and 1-based line number.
                  ``"raise"`` — raise a ``ValueError`` with the same context
                  instead of skipping.

    Yields:
        Parsed dicts, one per non-blank line.

    Raises:
        ValueError: When *on_error* is ``"raise"`` and a line cannot be parsed.
    """
    p = pathlib.Path(path)
    with p.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError as exc:
                msg = f"{p}:{lineno}: JSON parse error: {exc}"
                if on_error == "raise":
                    raise ValueError(msg) from exc
                log.warning(msg)


def normalize_turkish(text: str) -> str:
    """
    Turkish-aware text normalization.

    CRITICAL: replaces 'I' → 'ı' and 'İ' → 'i' BEFORE calling .lower(),
    because Python's built-in .lower() maps 'I' → 'i' (not Turkish dotless 'ı').
    """
    text = text.replace('İ', 'i').replace('I', 'ı')
    text = text.lower()
    text = unicodedata.normalize('NFC', text)
    return text


def set_seeds(seed: int = 42) -> None:
    """Set all RNG seeds for reproducibility (Python, NumPy, PyTorch CPU+GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def inject_citations(answer: str, chunks: list) -> str:
    """Append [Kaynak N] citation markers to answer sentences via token-overlap heuristic."""
    THRESHOLD = 0.15

    def _tok(text: str) -> set:
        return {t.lower() for t in re.split(r"[\s\.,;:!?()\[\]{}'\"]+", text) if t}

    def _overlap(a: set, b: set) -> float:
        return len(a & b) / min(len(a), len(b)) if a and b else 0.0

    sents = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if not sents or not chunks:
        return answer

    sent_toks = [_tok(s) for s in sents]
    pending: list[tuple[int, int, float]] = []

    for ci, chunk in enumerate(chunks):
        raw = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, "text", "")
        chunk_toks = _tok(raw)
        best_score, best_si = 0.0, -1
        for si, st in enumerate(sent_toks):
            sc = _overlap(st, chunk_toks)
            if sc > best_score:
                best_score, best_si = sc, si
        if best_score >= THRESHOLD and best_si >= 0:
            pending.append((best_si, ci + 1, best_score))

    if not pending:
        return answer

    from collections import defaultdict
    s2l: dict = defaultdict(list)
    for si, lbl, _ in pending:
        s2l[si].append(lbl)

    parts = []
    for i, sent in enumerate(sents):
        if i in s2l:
            tags = " ".join(f"[Kaynak {lbl}]" for lbl in sorted(s2l[i]))
            parts.append(f"{sent} {tags}")
        else:
            parts.append(sent)
    return " ".join(parts)


def check_ollama(base_url: str, model_name: str) -> bool:
    """Return True if Ollama is running and model_name is available."""
    import httpx
    import logging
    log = logging.getLogger(__name__)
    try:
        root = base_url.rstrip("/").removesuffix("/v1")
        resp = httpx.get(f"{root}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return False
        tags = resp.json().get("models", [])
        model_names = [m.get("name", "") for m in tags]
        if not any(model_name in name for name in model_names):
            log.warning(
                "Ollama is running but model '%s' is not pulled. Run: ollama pull %s",
                model_name, model_name,
            )
            return False
        return True
    except Exception:
        return False
