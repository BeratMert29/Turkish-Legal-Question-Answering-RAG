"""Stricter citation injection for RAG training data (script 10).

This module is intentionally SEPARATE from ``utils.inject_citations`` which is
used at inference/evaluation time and must remain unchanged.  Training-data
citations need tighter controls because bad citations teach the LLM to
hallucinate sources.

Key differences from ``utils.inject_citations``:
  - Denominator is always the *sentence/answer* token count (coverage of the
    answer BY the chunk), not ``min(len(a), len(b))``.  This prevents short
    answers from getting 1.0 overlap with every chunk.
  - Minimum token count: sentences with fewer than ``MIN_SENTENCE_TOKENS``
    content tokens are never cited (too short to reliably attribute).
  - Turkish stopwords are stripped before overlap computation so that common
    function words do not inflate scores.
  - Newline structure (``\\n``) in the answer is preserved: sentence splitting
    and joining operate per-line, never flattening with ``' '.join``.
  - Optional gold-source validation (not currently usable because the training
    data lacks gold chunk/article labels; documented for future use).

Design rationale for thresholds (see ``OVERLAP_THRESHOLD``, ``MIN_SENTENCE_TOKENS``):
  - OVERLAP_THRESHOLD = 0.35: a sentence must share >35% of its *content* tokens
    (stopwords excluded) with a chunk to be cited.  Lower than 0.5 because legal
    answers often paraphrase the statute; higher than the old 0.15 to avoid
    spurious matches on short common phrases.  Empirically tuned on 20-record
    sample.
  - MIN_SENTENCE_TOKENS = 3: a sentence with fewer than 3 content tokens (after
    stopword removal) is too short for reliable token overlap; skip citation to
    avoid false positives (e.g. "Evet.", "Hayir.", "Madde 5.").
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Turkish stopwords (same set as BM25 retriever uses)
# ---------------------------------------------------------------------------

try:
    import nltk as _nltk
    try:
        _STOPWORDS: frozenset[str] = frozenset(_nltk.corpus.stopwords.words("turkish"))
    except LookupError:
        _nltk.download("stopwords", quiet=True)
        _STOPWORDS = frozenset(_nltk.corpus.stopwords.words("turkish"))
except Exception:
    _STOPWORDS = frozenset()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OVERLAP_THRESHOLD: float = 0.35
"""Minimum fraction of sentence content-tokens covered by a chunk."""

MIN_SENTENCE_TOKENS: int = 3
"""Sentences with fewer content tokens than this are never cited."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_turkish(text: str) -> str:
    """Turkish-aware lowercase (same logic as ``utils.normalize_turkish``)."""
    text = text.replace("İ", "i").replace("I", "ı")  # İ->i, I->ı
    text = text.lower()
    return unicodedata.normalize("NFC", text)


def _tokenize(text: str) -> list[str]:
    """Split into lowercase Turkish-normalized tokens, no punctuation."""
    return [
        t
        for t in re.split(r"[\s\.,;:!?()\[\]{}'\"]+", _normalize_turkish(text))
        if t
    ]


def _content_tokens(text: str) -> list[str]:
    """Tokens with stopwords removed (for overlap scoring)."""
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


def _sentence_coverage(sentence_toks: set[str], chunk_toks: set[str]) -> float:
    """Fraction of *sentence* content tokens that appear in *chunk*.

    Denominator is always ``len(sentence_toks)`` -- how much of the sentence
    is *covered* by the chunk, not the other way round.  This is the critical
    fix: the old ``min(len(a), len(b))`` denominator made every short sentence
    score 1.0 against any chunk sharing a few words.
    """
    if not sentence_toks:
        return 0.0
    return len(sentence_toks & chunk_toks) / len(sentence_toks)


# ---------------------------------------------------------------------------
# Per-line sentence splitting
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences_per_line(text: str) -> list[tuple[int, str]]:
    """Split *text* into ``(line_index, sentence)`` pairs.

    Operates per-line (``\\n`` boundaries) so that the original line structure
    is preserved when reassembling.
    """
    result: list[tuple[int, str]] = []
    for li, line in enumerate(text.split("\n")):
        parts = _SENT_RE.split(line)
        for part in parts:
            stripped = part.strip()
            if stripped:
                result.append((li, stripped))
    return result


def _reassemble(
    lines_count: int,
    tagged_sents: list[tuple[int, str]],
) -> str:
    """Rebuild the answer preserving ``\\n`` boundaries.

    ``tagged_sents`` is ``[(line_index, sentence_with_possible_citation), ...]``.
    Sentences belonging to the same line are joined with ``' '``; lines are
    joined with ``'\\n'``.
    """
    buckets: dict[int, list[str]] = defaultdict(list)
    for li, sent in tagged_sents:
        buckets[li].append(sent)
    parts = []
    for li in range(lines_count):
        parts.append(" ".join(buckets.get(li, [])))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CitationStats:
    """Accumulator for citation statistics across a dataset."""

    total_records: int = 0
    cited_records: int = 0
    uncited_records: int = 0
    total_sentences: int = 0
    cited_sentences: int = 0
    short_sentence_skips: int = 0
    gold_missing_skips: int = 0  # reserved: always 0 until gold labels exist
    truncated_records: int = 0
    dropped_records: int = 0

    def record(
        self,
        n_sentences: int,
        n_cited: int,
        n_short_skips: int,
        has_truncation: bool = False,
        has_drop: bool = False,
    ) -> None:
        self.total_records += 1
        self.total_sentences += n_sentences
        self.cited_sentences += n_cited
        self.short_sentence_skips += n_short_skips
        if n_cited > 0:
            self.cited_records += 1
        else:
            self.uncited_records += 1
        if has_truncation:
            self.truncated_records += 1
        if has_drop:
            self.dropped_records += 1

    def summary_dict(self) -> dict:
        n = self.total_records or 1
        return {
            "total_records": self.total_records,
            "cited_records": self.cited_records,
            "uncited_records": self.uncited_records,
            "cited_pct": round(100 * self.cited_records / n, 1),
            "total_sentences": self.total_sentences,
            "cited_sentences": self.cited_sentences,
            "short_sentence_skips": self.short_sentence_skips,
            "gold_missing_skips": self.gold_missing_skips,
            "truncated_records": self.truncated_records,
            "dropped_records": self.dropped_records,
        }

    def log_summary(self) -> str:
        d = self.summary_dict()
        lines = [
            f"  Citation stats:",
            f"    Records: {d['cited_records']}/{d['total_records']} cited ({d['cited_pct']}%)",
            f"    Sentences: {d['cited_sentences']}/{d['total_sentences']} cited",
            f"    Short-sentence skips: {d['short_sentence_skips']}",
            f"    Gold-missing skips: {d['gold_missing_skips']} (gold labels unavailable in training data)",
            f"    Truncated records: {d['truncated_records']}",
            f"    Dropped records: {d['dropped_records']}",
        ]
        return "\n".join(lines)


def inject_citations_strict(
    answer: str,
    included_chunks: list[dict],
    *,
    truncated_ids: Optional[list[str]] = None,
    threshold: float = OVERLAP_THRESHOLD,
    min_tokens: int = MIN_SENTENCE_TOKENS,
) -> tuple[str, int, int, int]:
    """Inject ``[Kaynak N]`` citations into *answer* with strict controls.

    Args:
        answer: Raw gold answer text (may contain ``\\n``).
        included_chunks: Chunks in context order (index 0 = [Kaynak 1]).
        truncated_ids: chunk_ids that were truncated during context assembly.
            Citations to truncated chunks are only added if the overlapping
            tokens come from the portion of text that survived truncation.
            For simplicity, we still allow citation to truncated chunks because
            the overlap is computed against the full chunk text retrieved, and
            the truncated portion is still semantically attributable.  The
            alternative (skipping truncated chunks entirely) would lose valid
            citations.  This is documented as a known simplification.
        threshold: Minimum coverage ratio.
        min_tokens: Minimum content-token count for a sentence to be citeable.

    Returns:
        ``(cited_answer, n_cited_sentences, n_total_sentences, n_short_skips)``
    """
    if not answer or not included_chunks:
        return answer, 0, 0, 0

    lines = answer.split("\n")
    line_sents = _split_sentences_per_line(answer)
    if not line_sents:
        return answer, 0, 0, 0

    # Pre-tokenize chunks (content tokens, no stopwords)
    chunk_tok_sets: list[set[str]] = []
    for ch in included_chunks:
        raw = ch.get("text", "") if isinstance(ch, dict) else getattr(ch, "text", "")
        chunk_tok_sets.append(set(_content_tokens(raw)))

    n_cited = 0
    n_short_skips = 0
    tagged: list[tuple[int, str]] = []

    for li, sent in line_sents:
        s_toks = _content_tokens(sent)
        s_tok_set = set(s_toks)

        # Short-sentence guard
        if len(s_tok_set) < min_tokens:
            n_short_skips += 1
            tagged.append((li, sent))
            continue

        # Find all chunks exceeding threshold
        matches: list[int] = []
        for ci, ct_set in enumerate(chunk_tok_sets):
            score = _sentence_coverage(s_tok_set, ct_set)
            if score >= threshold:
                matches.append(ci + 1)  # 1-indexed Kaynak number

        if matches:
            tags = " ".join(f"[Kaynak {m}]" for m in sorted(matches))
            tagged.append((li, f"{sent} {tags}"))
            n_cited += 1
        else:
            tagged.append((li, sent))

    result = _reassemble(len(lines), tagged)
    return result, n_cited, len(line_sents), n_short_skips
