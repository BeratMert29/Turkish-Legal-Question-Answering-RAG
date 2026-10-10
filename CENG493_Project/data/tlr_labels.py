"""Check the gold article labels of the ``turkish_legal_rag`` eval set
against the law text.

The HF ``madde_no`` field is often off by one to three articles (the answer
is in article N+1 while the label says N: e.g. "Ara dinlenmesi" is İş Kanunu
68, labelled 67).  A label is checked by how much of the gold answer's
content is found in the labelled article's text:

* ``ok``          the labelled article covers >= ``low`` of the answer;
* ``relabelled``  it covers less, and either an article the question or
                  answer names explicitly ("101. maddesi") covers >= ``low``,
                  or an article of the same law and kind within
                  ±``max_offset`` covers >= ``high`` and beats the label by
                  >= ``margin`` -> the label moves there (named first);
* ``unverified``  neither (paraphrased answer, or the article is missing).

The original HF label is kept in ``madde_no_hf`` and every decision is
recorded in ``label_check``; ``config.TLR_USE_LABEL_FIXES`` switches between
the checked and the original labels.  Pure functions, no I/O.
"""

from __future__ import annotations

import collections
import re

from utils import normalize_turkish

LOW_COVERAGE = 0.5
HIGH_COVERAGE = 0.75
MIN_MARGIN = 0.25
MAX_OFFSET = 3
# Turkish is agglutinative: compare word stems by their first characters.
STEM_CHARS = 5

_STOPWORDS = frozenset(
    "ve veya ile bir bu da de için olarak göre olan ise ki gibi kadar daha en çok "
    "her şu o ancak hem ya mı mi mu mü ne nedir hangi nasıl evet hayır".split()
)
_WORD_RE = re.compile(r"\w+")
_KEY_RE = re.compile(r"^(?P<prefix>(?:[a-z]+-)?)(?P<num>\d+)(?P<suffix>-[a-z])?$")
_NAMED_ARTICLE_RES = (
    re.compile(r"(?i)(\d{1,4})\s*(?:\.|['’]?\s*(?:inci|ıncı|nci|ncı|üncü|uncu))?\s*madde"),
    re.compile(r"(?i)madde\s+(\d{1,4})\b"),
)


def named_articles(*texts: str) -> list[str]:
    """Plain article numbers named in the texts ("101. maddesi", "madde 5")."""
    found = [m for t in texts for rx in _NAMED_ARTICLE_RES for m in rx.findall(t or "")]
    return list(dict.fromkeys(found))


def content_stems(text: str) -> list[str]:
    """Stems of the content words of *text* (Turkish-lowercased, >= 3 chars,
    no stopwords, no bare numbers)."""
    return [
        w[:STEM_CHARS] for w in _WORD_RE.findall(normalize_turkish(text or ""))
        if len(w) >= 3 and not w.isdigit() and w not in _STOPWORDS
    ]


def answer_coverage(answer: str, article_text: str) -> float:
    """Fraction of the answer's content stems that occur in *article_text*."""
    stems = content_stems(answer)
    if not stems or not article_text:
        return 0.0
    vocab = set(content_stems(article_text))
    return sum(1 for s in stems if s in vocab) / len(stems)


def _body(text: str, madde: str) -> str:
    """Text from the article's own heading line on: titles and section
    headers above it ("Çocuk Düşürtme, Düşürme veya Kısırlaştırma") would
    otherwise lend the article words of its neighbours."""
    from data.data_processor import _MADDE_HEADING_RE, _heading_key

    for m in _MADDE_HEADING_RE.finditer(text):
        if _heading_key(m) == madde:
            return text[m.start():]
    return text


def article_texts(corpus_chunks) -> dict[tuple[str, str], str]:
    """``{(source, madde_no): body text}`` joining every chunk of each article."""
    out: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for c in corpus_chunks:
        madde = c["madde_no"] if isinstance(c, dict) else getattr(c, "madde_no", None)
        if not madde:
            continue
        madde = str(madde).lower()
        source = c["source"] if isinstance(c, dict) else c.source
        text = c["text"] if isinstance(c, dict) else c.text
        out[(source, madde)].append(_body(text, madde))
    return {k: "\n".join(v) for k, v in out.items()}


def neighbour_keys(madde_no: str, max_offset: int = MAX_OFFSET) -> list[tuple[int, str]]:
    """``[(offset, key)]`` of the same kind of article within ±max_offset
    ("12" -> "9".."15", "gecici-2" -> "gecici-1".."gecici-5").  Letter
    articles ("183-a") have no neighbours."""
    m = _KEY_RE.match(str(madde_no).lower())
    if not m or m.group("suffix"):
        return []
    num = int(m.group("num"))
    return [
        (off, f"{m.group('prefix')}{num + off}")
        for off in sorted(range(-max_offset, max_offset + 1), key=lambda o: (abs(o), -o))
        if off and num + off > 0
    ]


def check_label(row: dict, texts: dict[tuple[str, str], str], *,
                low: float = LOW_COVERAGE, high: float = HIGH_COVERAGE,
                margin: float = MIN_MARGIN, max_offset: int = MAX_OFFSET) -> dict:
    """Decision for one row: ``{"status", "coverage", "madde_no"[, "from",
    "offset"]}``; ``madde_no`` is the label to use."""
    label = str(row.get("madde_no_hf", row.get("madde_no")) or "").lower()
    source = row.get("source", "")
    if not label or not source:
        return {"status": "unverified", "coverage": None, "madde_no": label or None}
    answer = row.get("answer", "")
    cov = answer_coverage(answer, texts.get((source, label), ""))
    result = {"status": "ok", "coverage": round(cov, 3), "madde_no": label}
    if cov >= low:
        return result
    # Candidates: articles the question/answer names ("101. maddesi"; need
    # >= low) and same-kind neighbours within ±max_offset (need >= high).
    # The best coverage wins; ties go to a named article, then the nearest.
    cands = []
    for n in named_articles(row.get("question", ""), answer):
        if n != label:
            c = answer_coverage(answer, texts.get((source, n), ""))
            if c >= low:
                cands.append((c, 1, -abs(int(n) - int(label)) if label.isdigit() else 0,
                              n, "named"))
    for off, key in neighbour_keys(label, max_offset):
        c = answer_coverage(answer, texts.get((source, key), ""))
        if c >= high:
            cands.append((c, 0, -abs(off), key, "neighbour"))
    cands = [x for x in cands if x[0] - cov >= margin]
    if cands:
        best_cov, _, neg_dist, key, reason = max(cands)
        return {"status": "relabelled", "coverage": round(best_cov, 3), "madde_no": key,
                "from": label, "from_coverage": round(cov, 3), "reason": reason,
                "offset": (int(key) - int(label)
                           if key.isdigit() and label.isdigit() else neg_dist)}
    return {**result, "status": "unverified"}


def apply_label_checks(rows: list[dict], corpus_chunks) -> tuple[list[dict], list[dict], dict]:
    """Check every row (main set and HF-label conflicts).

    Returns ``(eval_rows, conflict_rows, report)``.  Every row keeps its HF
    label in ``madde_no_hf`` and the HF conflict flag in
    ``label_conflict_hf``; ``madde_no`` becomes the checked label.  A row
    flagged as a conflict by scripts/16 joins the eval set only when the
    check confirms its label (``ok``) or finds the right article
    (``relabelled``).  Re-running starts from the HF labels, so it is
    idempotent.
    """
    texts = article_texts(corpus_chunks)
    eval_rows, conflicts = [], []
    counts: collections.Counter = collections.Counter()
    offsets: collections.Counter = collections.Counter()
    for r in rows:
        r = dict(r)
        r.setdefault("madde_no_hf", r.get("madde_no"))
        r.setdefault("label_conflict_hf", bool(r.get("label_conflict")))
        decision = check_label(r, texts)
        r["label_check"] = decision
        r["madde_no"] = decision["madde_no"] or r["madde_no_hf"]
        kind = "conflict" if r["label_conflict_hf"] else "main"
        counts[f"{kind}_{decision['status']}"] += 1
        if decision["status"] == "relabelled":
            offsets[decision["offset"]] += 1
            counts[f"relabelled_by_{decision['reason']}"] += 1
        if r["label_conflict_hf"] and decision["status"] == "unverified":
            r["label_conflict"] = True
            conflicts.append(r)
        else:
            r["label_conflict"] = False
            eval_rows.append(r)
    report = {
        "thresholds": {"low": LOW_COVERAGE, "high": HIGH_COVERAGE,
                       "margin": MIN_MARGIN, "max_offset": MAX_OFFSET,
                       "stem_chars": STEM_CHARS},
        "counts": dict(counts),
        "relabel_offsets": {str(k): v for k, v in sorted(offsets.items())},
        "eval_rows": len(eval_rows),
        "conflict_rows": len(conflicts),
        "relabelled_query_ids": [r["query_id"] for r in eval_rows
                                 if r["label_check"]["status"] == "relabelled"],
    }
    return eval_rows, conflicts, report
