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


def set_seeds(seed: int = 42) -> None:  # callers pass config.SEED
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


def _ollama_name_matches(name: str, model: str) -> bool:
    """Exact Ollama tag match; an untagged model name also matches ':latest'."""
    return name == model or name == f"{model}:latest"


def ollama_models(base_url: str) -> "dict[str, str] | None":
    """``{model name: digest}`` of the models Ollama has pulled, or None when
    the server is unreachable."""
    import httpx
    try:
        root = base_url.rstrip("/").removesuffix("/v1")
        resp = httpx.get(f"{root}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return None
        return {m.get("name", ""): m.get("digest", "") for m in resp.json().get("models", [])}
    except Exception:
        return None


def check_ollama(base_url: str, model_name: str) -> bool:
    """Return True if Ollama is running and model_name (exact tag) is pulled."""
    models = ollama_models(base_url)
    if models is None:
        return False
    if not any(_ollama_name_matches(name, model_name) for name in models):
        log.warning(
            "Ollama is running but model '%s' is not pulled. Run: ollama pull %s",
            model_name, model_name,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Run provenance
# ---------------------------------------------------------------------------

def file_sha256(path: "str | pathlib.Path") -> "str | None":
    """sha256 of a file's bytes, or None when it does not exist."""
    import hashlib
    p = pathlib.Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def corpus_fingerprint(chunks) -> str:
    """sha256 over (chunk_id, text) of every chunk, in order: identifies the
    exact index content the gold labels were built against."""
    import hashlib
    h = hashlib.sha256()
    for c in chunks:
        cid = c["chunk_id"] if isinstance(c, dict) else c.chunk_id
        text = c["text"] if isinstance(c, dict) else c.text
        h.update(cid.encode("utf-8") + b"\0" + text.encode("utf-8") + b"\0")
    return h.hexdigest()


def git_revision(cwd: "str | pathlib.Path | None" = None) -> dict:
    """``{"commit": sha | None, "dirty": bool | None}`` of the working tree."""
    import subprocess
    cwd = str(cwd or pathlib.Path(__file__).resolve().parent)
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, timeout=10, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    cwd=cwd, capture_output=True, text=True,
                                    timeout=10).stdout.strip())
        return {"commit": sha, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def package_versions(names=("torch", "transformers", "sentence-transformers", "faiss-cpu",
                            "faiss-gpu", "rank-bm25", "ranx", "numpy", "requests",
                            "langchain-text-splitters", "snowballstemmer", "peft")) -> dict:
    """Installed versions of the packages that affect results (absent ones omitted)."""
    from importlib import metadata
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            continue
    return out


def run_provenance(*, seed: int, base_url: str, models: "list[str]",
                   eval_file: "str | pathlib.Path | None" = None,
                   corpus_chunks=None, extra: "dict | None" = None) -> dict:
    """Everything needed to tell whether two result files are comparable:
    code revision, seed, input hashes, Ollama model digests, package versions."""
    import platform
    import sys
    import time
    pulled = ollama_models(base_url) or {}
    digests = {
        m: next((d for name, d in pulled.items() if _ollama_name_matches(name, m)), None)
        for m in models
    }
    prov = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": git_revision(),
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "ollama_model_digests": digests,
        "eval_file": str(eval_file) if eval_file else None,
        "eval_file_sha256": file_sha256(eval_file) if eval_file else None,
        "corpus_n_chunks": len(corpus_chunks) if corpus_chunks is not None else None,
        "corpus_fingerprint": corpus_fingerprint(corpus_chunks) if corpus_chunks is not None else None,
    }
    if extra:
        prov.update(extra)
    return prov
