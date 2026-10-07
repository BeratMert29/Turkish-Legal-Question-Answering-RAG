"""Clean the scraped supplementary-law records (``extra_laws.jsonl``).

The scrape has three defects that make ``doc_id`` / ``chunk_id`` collide and
mislabel articles:

1. Section headings are split from their article: "Ek Madde N", "Geçici Madde
   N" and "Ek Geçici Madde N" lost their prefix and the bare word ("Ek",
   "GEÇİCİ", ...) is stranded as the last line of the *previous* record.  The
   records that follow are then labelled ``madde_N`` and collide with the real
   article N.  We recover the prefix from the stranded marker.
2. Letter articles ("MADDE 5/A") get ``doc_id`` ``..._madde_5`` and collide
   with article 5.  We keep the letter (``..._madde_5-a``).
3. PDF amendment tables / amending-law bodies were captured as records:
   absurd article numbers (``madde_32313``, law numbers such as 6728) and
   amending-law sections ("MADDE 1 – 12/1/2011 tarihli ve 6100 sayılı ...").
   These are dropped.
4. Article titles (and section headers) sit at the end of the *previous*
   record, because the scrape split right before "MADDE N".  They are moved
   to the start of the article they name.

``clean_extra_law_records`` is pure (no I/O) so it is unit-testable.
"""
from __future__ import annotations

import collections
import re

# A real law has far fewer articles than this; larger numbers are law numbers
# or PDF-table residue (madde_32313, madde_6728, ...).
MAX_ARTICLE_NUMBER = 2000

_LETTER = r"[A-Za-zÇĞİÖŞÜçğıöşü]"
_HEADING = re.compile(
    r"^(?P<lead>\s*)(?P<word>MADDE|Madde)(?P<sp>\s+)(?P<num>\d+)(?:\s*/\s*(?P<letter>" + _LETTER + r"))?(?P<tail>\s*[-–—(]|\s*\n\s*\()?",
)
# Stranded section marker: the whole last line of a record.
_MARKER = re.compile(r"^(Ek\s+Geçici|EK\s+GEÇİCİ|Ek|EK|GEÇİCİ|Geçici)$")
_PREFIX = {"ek": "EK MADDE", "gecici": "GEÇİCİ MADDE", "ekgecici": "EK GEÇİCİ MADDE"}
_SLUG = {"ek": "ek_madde", "gecici": "gecici_madde", "ekgecici": "ekgecici_madde"}

# Text that identifies amendment tables / amending-law bodies.
_JUNK_TEXT = re.compile(
    r"Yürürlüğe\s+Giriş|Değiştiren\s+Kanunun|İptal\s+Eden|"
    r"^\s*(?:MADDE|Madde)\s+\d+\s*[–—-]\s*\d{1,2}/\d{1,2}/\d{4}\s+tarihli",
    re.IGNORECASE | re.MULTILINE,
)
# Amending-law instruction, e.g. "MADDE 2 – 1086 sayılı Kanuna aşağıdaki ... eklenmiştir."
_AMENDING_STMT = re.compile(r"\d+\s+sayılı\s+Kanun\w*\s+aşağıdaki", re.IGNORECASE)
_AMENDING_BODY = re.compile(
    r"\bBu\s+Kanunla\b|\bBu\s+Kanun[^\n]{0,60}(?:yayımı|yürürlüğe\s+girer)|Bakanlar\s+Kurulu\s+yürütür",
    re.IGNORECASE,
)


# Trailing title block: short lines without sentence-final punctuation.
_TITLE_MAX_CHARS = 120
_TITLE_MAX_LINES = 8


def split_trailing_titles(text: str) -> "tuple[str, str]":
    """``(body, titles)``: the trailing article-title / section-header lines
    of *text* (short, no sentence-final punctuation, at most
    _TITLE_MAX_LINES) split off the body."""
    lines = text.rstrip().split("\n")
    k = len(lines)
    taken = 0
    while k > 1 and taken < _TITLE_MAX_LINES:
        line = lines[k - 1].strip()
        if line:
            if (len(line) > _TITLE_MAX_CHARS or line[-1] in ".;:,!?)"
                    or _HEADING.match(line)):
                break
            taken += 1
        k -= 1
    if not taken:
        return text, ""
    return "\n".join(lines[:k]).rstrip(), "\n".join(lines[k:]).strip()


def _move_titles_forward(out: "list[dict]", stats: collections.Counter) -> None:
    """Give each article record the title lines stranded at the end of the
    record before it (same source, next record is an article)."""
    # Decide which records are articles before any text is moved: a record
    # that received titles no longer starts with its heading.
    is_article = [bool(_HEADING.match(r.get("text", ""))) for r in out]
    for k, (prev, cur) in enumerate(zip(out, out[1:])):
        if prev.get("source") != cur.get("source") or not (is_article[k] and is_article[k + 1]):
            continue
        body, titles = split_trailing_titles(prev["text"])
        if titles:
            prev["text"] = body
            cur["text"] = f"{titles}\n{cur['text']}"
            stats["moved_title_lines"] += titles.count("\n") + 1


def _key(num: int, letter: "str | None") -> tuple:
    return (num, letter or "")


def _marker_mode(text: str) -> "str | None":
    last = text.rstrip().split("\n")[-1].strip()
    if not _MARKER.match(last):
        return None
    low = last.replace("İ", "i").replace("I", "ı").lower()
    if "geçici" in low:
        return "ekgecici" if low.startswith("ek") else "gecici"
    return "ek"


def clean_extra_law_records(records: "list[dict]") -> "tuple[list[dict], dict]":
    """Return ``(cleaned, stats)`` for raw ``extra_laws`` records.

    ``cleaned`` keeps file order; each kept record is a new dict whose
    ``doc_id`` is unique per (source, section, article) wherever the scrape
    allowed.  ``stats`` counts what was dropped / relabelled.
    """
    stats: collections.Counter = collections.Counter()
    state: dict = {}  # source -> {"mode", "prev"}
    out: list[dict] = []
    for rec in records:
        source = rec.get("source", "")
        text = rec.get("text", "")
        st = state.setdefault(source, {"mode": "main", "prev": None, "max": 0})
        m = _HEADING.match(text)
        if m is None:
            # Not an article record (preamble / stray text): leave untouched.
            out.append(rec)
            stats["kept_no_heading"] += 1
            continue
        num = int(m.group("num"))
        letter = (m.group("letter") or "").lower() or None
        if num > MAX_ARTICLE_NUMBER:
            stats["dropped_pdf_table_number"] += 1
            continue

        mode = st["mode"]
        key = _key(num, letter)
        # Number sequence restarted without a marker: a new, unlabelled
        # section (typically an amending law appended after the body).
        if st["prev"] is not None and mode != "unknown":
            big_reset = key < st["prev"] and (num * 2 < st["max"] or st["max"] - num >= 100)
            if mode == "main" and big_reset:
                mode = st["mode"] = "unknown"
            elif mode != "main" and key < st["prev"]:
                mode = st["mode"] = "unknown"
        if mode == "unknown" and (_JUNK_TEXT.search(text) or _AMENDING_BODY.search(text)):
            stats["dropped_amending_law"] += 1
            st["prev"] = key
            marker = _marker_mode(text)
            if marker:
                st["mode"] = marker
                st["prev"] = None
            continue
        if _JUNK_TEXT.search(text) or m.group("tail") is None or _AMENDING_STMT.search(text[:250]):
            stats["dropped_amendment_table"] += 1
            continue

        new = dict(rec)
        slug = _SLUG.get(mode)
        if slug:
            new["text"] = text[: m.start("word")] + _PREFIX[mode] + text[m.end("word"):]
            new["doc_id"] = f"{source}_{slug}_{num}" + (f"-{letter}" if letter else "")
            stats[f"relabelled_{mode}"] += 1
        elif letter:
            new["doc_id"] = f"{source}_madde_{num}-{letter}"
            stats["relabelled_letter_article"] += 1

        marker = _marker_mode(new["text"])
        if marker:
            new["text"] = new["text"].rstrip()[: -len(new["text"].rstrip().split("\n")[-1])].rstrip()
            st["mode"] = marker
            st["prev"] = None
            st["max"] = 0
        else:
            st["prev"] = key
            if mode == "main":
                st["max"] = max(st["max"], num)
        out.append(new)
        stats["kept"] += 1
    _move_titles_forward(out, stats)
    return out, dict(stats)
