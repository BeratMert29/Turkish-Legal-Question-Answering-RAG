from dataclasses import dataclass, asdict
from typing import Iterator
import collections
import hashlib
import json
import pathlib
import re

import pandas as pd
import config
from utils import read_jsonl as _read_jsonl
from langchain_text_splitters import RecursiveCharacterTextSplitter


_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Patterns for extracting a Turkish law article (madde) number from free text.
_MADDE_PATTERNS = [
    re.compile(r"(\d+)\s*\.?\s*madde", re.IGNORECASE),   # "44. madde", "44 madde"
    re.compile(r"madde\s*(\d+)", re.IGNORECASE),           # "madde 44", "MADDE44"
    re.compile(r"md\.\s*(\d+)", re.IGNORECASE),            # "md. 44"
    re.compile(r"(?<!\d)m\.\s*(\d+)", re.IGNORECASE),     # "m. 44" (short form)
]

# Leading MADDE header in a chunk: "MADDE 44" or "MADDE 44 –" or "MADDE 44-"
_MADDE_HEADER_RE = re.compile(r"(?m)^\s*MADDE\s+(\d+)\b", re.IGNORECASE)


def _extract_madde_no(question: str, answer: str) -> "int | None":
    """Return the first article (madde) number found in question or answer text.

    Tries several Turkish legal shorthand patterns and returns None when no
    article number can be unambiguously determined.

    Args:
        question: The query text.
        answer:   The expected answer text.

    Returns:
        Integer article number, or None if not found.
    """
    text = f"{question} {answer}"
    for pat in _MADDE_PATTERNS:
        m = pat.search(text)
        if m:
            return int(m.group(1))
    return None


def _chunk_matches_article(chunk: "CorpusChunk", madde_no: int) -> bool:
    """Return True if *chunk* belongs to Turkish law article *madde_no*.

    Checks (in order):
    1. Optional ``madde_no`` field on the chunk (added by the corpus builder
       agent for article-chunked corpora).
    2. doc_id pattern  e.g. ``"LawName_madde_44"`` or ``"law_madde_44_0"``.
    3. Chunk text starts with ``"MADDE <N>"`` (article header at top of chunk).
    4. Any ``"MADDE <N>"`` header on its own line anywhere in the chunk text
       (article header may appear mid-chunk when sub-splitting oversized articles).

    Args:
        chunk:    A CorpusChunk instance.
        madde_no: The article number to match.

    Returns:
        True if the chunk is associated with article *madde_no*.
    """
    # 1. Explicit madde_no field (set by corpus builder for article-chunked corpora)
    stored = getattr(chunk, "madde_no", None)
    if stored is not None:
        return str(stored).strip() == str(madde_no)

    # 2. doc_id pattern — e.g. "Anayasa_madde_44" / "law_madde_44_2"
    if re.search(rf"madde[_\s]{madde_no}(?:[^\d]|$)", chunk.doc_id, re.IGNORECASE):
        return True

    # 3 & 4. Scan chunk text for a MADDE header on its own line
    for m in _MADDE_HEADER_RE.finditer(chunk.text):
        if int(m.group(1)) == madde_no:
            return True

    return False


def _chunk_matches_madde_str(
    chunk: "CorpusChunk", madde_no: str, inherited: "str | None" = None,
) -> bool:
    """Return True if *chunk* belongs to the article named by the normalised
    string *madde_no* (``"12"``, ``"183-a"``, ``"ek-3"``, ``"gecici-2"``).

    A chunk belongs to an article when its stored ``madde_no`` is that
    article, when *inherited* (the article whose text continues into the
    chunk, see :func:`_inherited_madde_nos`) is that article, or when the chunk
    holds that article's line-anchored heading."""
    want = str(madde_no).strip().lower()
    for cand in (getattr(chunk, "madde_no", None), inherited):
        if cand is not None and str(cand).strip().lower() == want:
            return True
    # The stored/leading article can differ from the one asked for when the
    # chunk opens with a section title or holds several articles, so also scan
    # every line-anchored article heading in the chunk text.
    for m in _MADDE_HEADING_RE.finditer(chunk.text):
        if _heading_key(m) == want:
            return True
    return False


# Turkish-aware tokenizer for silver lexical scoring.
# İ→i and I→ı to handle Turkish case-folding correctly (avoiding ASCII lowercasing
# that would map İ→i but leave I as i, conflating two different letters).
_TR_UPPER_MAP = str.maketrans("İIĞÜŞÖÇ", "iığüşöç")
_SILVER_STOPWORDS = frozenset(
    "bir bu o ve ya da ile de mi ne için bir de ya olan olan olan olan".split()
)


def _turkish_tokenize(text: str) -> list[str]:
    """Tokenize Turkish text with correct case folding (İ→i, I→ı).

    Returns a list of word tokens with length >= 2, excluding common
    Turkish stopwords that carry little retrieval signal.
    """
    lowered = text.translate(_TR_UPPER_MAP).lower()
    tokens = re.findall(r'\w+', lowered)
    return [t for t in tokens if len(t) >= 2 and t not in _SILVER_STOPWORDS]


def _silver_lexical_score(query_tokens: "list[str]", chunk_text: str) -> float:
    """Normalized token recall: fraction of query tokens present in chunk text.

    Args:
        query_tokens: Pre-tokenized query (question + answer tokens).
        chunk_text:   Raw chunk text (tokenized internally).

    Returns:
        Float in [0, 1].
    """
    if not query_tokens:
        return 0.0
    chunk_token_set = set(_turkish_tokenize(chunk_text))
    if not chunk_token_set:
        return 0.0
    matched = sum(1 for t in query_tokens if t in chunk_token_set)
    return matched / len(query_tokens)


# Anchored to line-start ((?m)^\s*) so mid-text references like
# "Madde 5 uyarınca" are never mistaken for article headings, and followed by
# a heading separator ("Madde 12 –", "MADDE 12. -", "Madde 12- (1)",
# "MADDE 12 İşyeri…" or end of line), so amendment-table rows such as
# "Madde 3 14/4/2011" or "Madde 9," are not headings either.
# Suffix "[/-][A-Za-z]" captures "183/A" or "183-A" (only when no further
# letter follows: "Madde 605-Yasal" is article 605); normalised to "183-a"
# via _normalize_madde_suffix().
# ekg : ekgecici-N — "Ek Geçici Madde 2"
# ek  : ek-N       — "Ek Madde 3", "EK MADDE 3"
# gec : gecici-N   — "Geçici Madde 7", "GEÇİCİ MADDE 4"
# muk : mukerrer-N — "Mükerrer Madde 5"
# reg : N          — "MADDE 86", "MADDE 183/A"
_TR_LETTERS = "A-Za-zÇĞİÖŞÜçğıöşüÂâÎîÛû"
_NUM = r"\d+(?:[/-][" + _TR_LETTERS + r"](?![" + _TR_LETTERS + r"]))?"
_HEADING_SEP = r"(?=[ \t]*(?:\.[ \t]*)?[-–—(]|[ \t]*\.?[ \t]*$|[ \t]+[A-ZÇĞİÖŞÜ])"
_MADDE_HEADING_RE = re.compile(
    r"(?m)^\s*(?:"
    r"(?:Ek|EK)\s+(?:[Gg]eçici|GEÇİCİ)\s+(?:[Mm][Aa][Dd][Dd][Ee])\s+(?P<ekg>" + _NUM + r")"
    r"|(?:Ek|EK)\s+(?:[Mm][Aa][Dd][Dd][Ee])\s+(?P<ek>" + _NUM + r")"
    r"|(?:[Gg]eçici|GEÇİCİ)\s+(?:[Mm][Aa][Dd][Dd][Ee])\s+(?P<gec>" + _NUM + r")"
    r"|(?:[Mm]ükerrer|MÜKERRER)\s+(?:[Mm][Aa][Dd][Dd][Ee])\s+(?P<muk>" + _NUM + r")"
    r"|(?:MADDE|Madde)\s+(?P<reg>" + _NUM + r")"
    r")" + _HEADING_SEP
)
# Longest line (chars) still treated as an article title above a heading.
_TITLE_MAX_CHARS = 200
_TITLE_MAX_LINES = 8


def _heading_line_start(text: str, m: "re.Match") -> int:
    """Offset of the start of the line holding heading match *m*."""
    pos = m.start() + (len(m.group(0)) - len(m.group(0).lstrip()))
    return text.rfind("\n", 0, pos) + 1


def _title_start(text: str, line_start: int, floor: int = 0) -> int:
    """Move a split point back over the title lines above an article heading.

    Turkish codes put the article title (and section headers such as
    "İKİNCİ BÖLÜM") on the lines just above "Madde N –".  Up to
    _TITLE_MAX_LINES short lines without sentence-final punctuation are
    taken, blank lines skipped; another heading or a sentence stops the walk.
    Never goes below *floor* (the line after the previous heading line).
    """
    start = i = line_start
    taken = 0
    while taken < _TITLE_MAX_LINES and i > floor:
        j = text.rfind("\n", floor, i - 1) + 1
        j = max(j, floor)
        line = text[j:i].strip()
        if line:
            if (len(line) > _TITLE_MAX_CHARS or line[-1] in ".;:,!?"
                    or _MADDE_HEADING_RE.match(line)):
                break
            start = j
            taken += 1
        i = j
    return start


def _article_parts(text: str) -> "list[str]":
    """Split a law text into [preamble, article, article, ...]; each article
    part starts with its title lines and heading."""
    cuts: list[int] = []
    floor = 0
    for m in _MADDE_HEADING_RE.finditer(text):
        line_start = _heading_line_start(text, m)
        cut = _title_start(text, line_start, floor)
        if cuts and cut <= cuts[-1]:
            cut = line_start
        if cuts and cut <= cuts[-1]:
            continue
        cuts.append(cut)
        # Title lines of the next article must come after this heading line.
        line_end = text.find("\n", m.end())
        floor = len(text) if line_end == -1 else line_end + 1
    bounds = [0, *cuts, len(text)]
    return [text[a:b] for a, b in zip(bounds, bounds[1:])]


def _normalize_madde_suffix(raw: str) -> str:
    """Normalise a MADDE number: ``'183/A'`` → ``'183-a'``, ``'5'`` → ``'5'``."""
    return re.sub(
        r"[/-]([A-Za-zÇĞİÖŞÜçğıöşü])",
        lambda m: "-" + m.group(1).replace("İ", "i").lower(),
        raw,
    )


def _heading_key(m: "re.Match") -> str:
    """Normalised article key for a ``_MADDE_HEADING_RE`` match."""
    if m.group("ekg"):
        return f"ekgecici-{_normalize_madde_suffix(m.group('ekg'))}"
    if m.group("ek"):
        return f"ek-{_normalize_madde_suffix(m.group('ek'))}"
    if m.group("gec"):
        return f"gecici-{_normalize_madde_suffix(m.group('gec'))}"
    if m.group("muk"):
        return f"mukerrer-{_normalize_madde_suffix(m.group('muk'))}"
    return _normalize_madde_suffix(m.group("reg"))


def _madde_no_from_text(text: str) -> "str | None":
    """Return the leading article number from text, or None if absent.

    Only matches headings anchored to a line boundary so that inline
    references such as "Madde 5 uyarınca" do not set the chunk's article.
    The whole text is scanned (a heading can sit after a long preamble).

    Returns:
        ``"N"`` for a regular article (e.g. ``"86"`` or ``"183-a"``),
        ``"ek-N"`` for supplementary articles (Ek Madde),
        ``"gecici-N"`` for transitory articles (Geçici Madde),
        ``"ekgecici-N"`` for Ek Geçici Madde,
        ``"mukerrer-N"`` for Mükerrer Madde,
        or ``None`` when no heading is found.
    """
    m = _MADDE_HEADING_RE.search(text)
    return None if m is None else _heading_key(m)


def _assign_madde_nos(texts: "list[str]", carry: "str | None" = None) -> "list[str | None]":
    """Article number for each consecutive chunk of one document.

    A chunk that opens with an article heading takes that article; a chunk
    that starts mid-article (a continuation) inherits the last article seen
    in the previous chunk, so every chunk of an article carries its number.
    """
    out: "list[str | None]" = []
    for text in texts:
        keys = [(m.start(), _heading_key(m)) for m in _MADDE_HEADING_RE.finditer(text)]
        if keys:
            lead = text[:keys[0][0]].strip()
            out.append(carry if (lead and carry) else keys[0][1])
            carry = keys[-1][1]
        else:
            out.append(carry)
    return out


# Text before a chunk's first heading counts as the tail of the previous
# article only when it reads like body text (a sentence end after a lowercase
# letter / closing paren), not like a title block ("I. Devletin şekli").
_TAIL_MIN_CHARS = 100
_SENTENCE_END_RE = re.compile(r"[a-zçğıöşü)][.;:](?:\s|$)")


def _is_article_tail(lead: str) -> bool:
    lead = lead.strip()
    return len(lead) >= _TAIL_MIN_CHARS and bool(_SENTENCE_END_RE.search(lead))


def _inherited_madde_nos(corpus_chunks) -> "dict[str, str]":
    """Article whose text continues into each chunk, from the previous
    chunks of the same document.

    A chunk with no heading continues the last article seen; a chunk whose
    text before its first heading is body text (the end of the previous
    article, e.g. a character chunk "...fıkra (3) ... MADDE 95 –") also
    continues it, so that article is labelled gold too.  Needed for corpora
    built before ``madde_no`` was stored, by external tools, or by the
    character chunker.
    """
    out: dict[str, str] = {}
    carry: dict[tuple, "str | None"] = {}
    for c in corpus_chunks:
        key = (c.source, c.doc_id)
        heads = list(_MADDE_HEADING_RE.finditer(c.text))
        prev = carry.get(key)
        if prev and (not heads or _is_article_tail(c.text[:heads[0].start()])):
            out[c.chunk_id] = prev
        if heads:
            carry[key] = _heading_key(heads[-1])
        elif getattr(c, "madde_no", None):
            carry[key] = c.madde_no
    return out


def normalize_question(text) -> str:
    """Question key for leakage checks: Turkish-lowercased, punctuation and
    whitespace runs collapsed."""
    from utils import normalize_turkish
    return re.sub(r"\W+", " ", normalize_turkish(str(text or ""))).strip()


@dataclass
class CorpusChunk:
    chunk_id: str   # f"{source}_{doc_id}_{chunk_index}"
    doc_id: str
    text: str
    source: str
    char_len: int
    madde_no: "str | None" = None  # e.g. "12", "ek-3", "gecici-2", or None


@dataclass
class QAExample:
    query_id: str
    question: str
    answer: str
    context: str    # "" for test/train rows (null in CSV)
    source: str
    data_type: str
    madde_no: "str | None" = None   # explicit gold article (turkish_legal_rag)
    hf_row_id: "str | None" = None


# ---------------------------------------------------------------------------
# Gold-label strategies used by DataProcessor.build_relevant_chunk_map
# ---------------------------------------------------------------------------

class _LabelIndex:
    """Lookup structures over the corpus, built once per labeling call."""

    def __init__(self, corpus_chunks) -> None:
        self.chunks = list(corpus_chunks)
        self.hash_to_ids: dict[str, list[str]] = {}
        self.by_source: dict[str, list] = {}
        for chunk in self.chunks:
            h = hashlib.md5(chunk.text.encode()).hexdigest()
            self.hash_to_ids.setdefault(h, []).append(chunk.chunk_id)
            self.by_source.setdefault(chunk.source, []).append(chunk)
        self.valid_ids = {c.chunk_id for c in self.chunks}
        self.inherited = _inherited_madde_nos(self.chunks)


@dataclass
class _QAFields:
    """The QA fields labeling needs, from a QAExample or a plain dict."""
    query_id: str
    question: str
    answer: str
    context: str
    source: str
    madde_no: "str | None"
    gold_ids: list

    @classmethod
    def of(cls, qa) -> "_QAFields":
        get = qa.get if isinstance(qa, dict) else (lambda k, d=None: getattr(qa, k, d))
        return cls(
            query_id=get("query_id"),
            question=get("question", "") or "",
            answer=get("answer", "") or "",
            context=get("context", "") or "",
            source=get("source", "") or "",
            madde_no=get("madde_no"),
            gold_ids=list(get("gold_source_ids") or []),
        )


def _label_gold_ids(q: _QAFields, idx: _LabelIndex) -> list[str]:
    """0: chunk ids supplied by an evaluator benchmark (those in the corpus)."""
    return [gid for gid in q.gold_ids if gid in idx.valid_ids]


def _label_explicit_madde(q: _QAFields, idx: _LabelIndex) -> list[str]:
    """0.5: the explicit (source, madde_no) of the row (turkish_legal_rag)."""
    if not (q.madde_no and q.source):
        return []
    return [c.chunk_id for c in idx.by_source.get(q.source, [])
            if _chunk_matches_madde_str(c, q.madde_no, idx.inherited.get(c.chunk_id))]


def _label_context_hash(q: _QAFields, idx: _LabelIndex) -> list[str]:
    """1: re-chunk the row's context with the corpus chunker, match by hash."""
    if not q.context:
        return []
    found: list[str] = []
    for chunk in DataProcessor.chunk_text(q.context, q.query_id, q.source):
        found.extend(idx.hash_to_ids.get(hashlib.md5(chunk.text.encode()).hexdigest(), []))
    return list(dict.fromkeys(found))


def _label_doc_id(q: _QAFields, idx: _LabelIndex) -> list[str]:
    """2: chunks of the document whose id is the query id."""
    return [c.chunk_id for c in idx.chunks if c.doc_id == q.query_id]


def _label_answer_substring(q: _QAFields, idx: _LabelIndex) -> list[str]:
    """2.5: chunks containing the first 80 characters of a long answer."""
    if len(q.answer) < 40:
        return []
    needle = q.answer.lower().strip()[:80]
    pool = idx.by_source.get(q.source, idx.chunks) if q.source else idx.chunks
    return [c.chunk_id for c in pool if needle in c.text.lower()]


def _label_article_mention(q: _QAFields, idx: _LabelIndex) -> list[str]:
    """3: the article number named in the question/answer, in the gold law.

    Queries naming no article stay unlabeled rather than labelling a whole law.
    """
    if not q.source:
        return []
    madde_no = _extract_madde_no(q.question, q.answer)
    if madde_no is None:
        return []
    return [c.chunk_id for c in idx.by_source.get(q.source, [])
            if _chunk_matches_article(c, madde_no)]


def _label_silver_lexical(q: _QAFields, idx: _LabelIndex) -> list[str]:
    """3.5 (optional): top-m chunks of the gold law by token overlap with
    question+answer, above config.SILVER_THRESHOLD -- silver, not gold."""
    source_chunks = idx.by_source.get(q.source, []) if q.source else []
    if not source_chunks:
        return []
    q_tokens = _turkish_tokenize(f"{q.question} {q.answer}")
    scored = sorted(((c, _silver_lexical_score(q_tokens, c.text)) for c in source_chunks),
                    key=lambda x: -x[1])
    return [c.chunk_id for c, score in scored[:config.SILVER_TOP_M]
            if score >= config.SILVER_THRESHOLD]


class DataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Loading / validation
    # ------------------------------------------------------------------

    def load_and_validate(self) -> dict:
        """Load CSV, check required columns, return summary dict."""
        self._df = pd.read_csv(self.csv_path)

        if self._df.empty:
            raise ValueError(f"CSV is empty: {self.csv_path}")

        required_columns = {"id", "question", "answer", "context", "source", "data_type", "score", "split"}
        missing = required_columns - set(self._df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {missing}")

        summary = {
            "total_rows": len(self._df),
            "columns": list(self._df.columns),
            "split_counts": self._df["split"].value_counts().to_dict(),
            "null_context_count": int(self._df["context"].isna().sum()),
        }
        return summary

    def _ensure_loaded(self):
        if self._df is None:
            self.load_and_validate()

    # ------------------------------------------------------------------
    # Row accessors
    # ------------------------------------------------------------------

    def get_corpus_rows(self) -> pd.DataFrame:
        """Rows where split == 'kaggle' (have context)."""
        self._ensure_loaded()
        return self._df[self._df["split"] == "kaggle"].reset_index(drop=True)

    def get_qa_split(self, split: str) -> pd.DataFrame:
        self._ensure_loaded()
        return self._df[self._df["split"] == split].reset_index(drop=True)

    def kaggle_eval_df(self) -> pd.DataFrame:
        """The kaggle rows used as the ``kaggle`` eval set.

        Up to ``config.QA_EVAL_EXPECTED`` rows, taken round-robin over the
        distinct contexts (one question per context first, then a second,
        ...), each in a fixed hash order, so the questions are as independent
        as the data allows and the set is deterministic.  Questions that also
        occur in the ``train`` split are left out (train/eval leakage).

        The contexts stay in the index: a retrieval eval needs its gold
        passages in the corpus, so leakage control happens on the training
        data (:meth:`build_qa_train_set`, :meth:`build_kaggle_train_set`),
        not by removing passages from the index.
        """
        if hasattr(self, "_kaggle_eval_cache"):
            return self._kaggle_eval_cache

        def _md5(val) -> str:
            return hashlib.md5(str(val).encode("utf-8")).hexdigest()

        df = self.get_corpus_rows()
        df = df[df["context"].notna() & (df["context"].astype(str) != "")]
        train_keys = {normalize_question(q) for q in self.get_qa_split("train")["question"].dropna()}
        df = df[~df["question"].fillna("").map(normalize_question).isin(train_keys)]
        groups = [
            sorted(g.index, key=lambda i: _md5(df.at[i, "id"]))
            for _, g in sorted(df.groupby(df["context"].map(_md5)), key=lambda kv: kv[0])
        ]
        picked: list[int] = []
        depth = 0
        while len(picked) < config.QA_EVAL_EXPECTED and any(len(g) > depth for g in groups):
            for g in groups:
                if depth < len(g) and len(picked) < config.QA_EVAL_EXPECTED:
                    picked.append(g[depth])
            depth += 1
        self._kaggle_eval_cache = df.loc[picked].reset_index(drop=True)
        return self._kaggle_eval_cache

    def eval_question_keys(self) -> set[str]:
        """Normalised questions of every eval set (kaggle, turkish_legal_rag,
        HMGS) -- to be kept out of any training data."""
        keys = {normalize_question(q) for q in self.kaggle_eval_df()["question"].dropna()}
        return keys | DataProcessor.saved_eval_question_keys()

    @staticmethod
    def saved_eval_question_keys() -> set[str]:
        """Normalised questions of the eval sets available without the raw
        CSV: turkish_legal_rag (incl. label conflicts), HMGS and any saved
        qa_eval.jsonl."""
        keys: set[str] = set()
        files = [
            pathlib.Path(config.TLR_DATA_PATH),
            pathlib.Path(config.TLR_PROCESSED_DIR) / "qa_turkish_legal_rag.label_conflicts.jsonl",
            pathlib.Path(config.TLR_PROCESSED_DIR) / "qa_eval.jsonl",
            pathlib.Path(config.PROCESSED_DIR) / "qa_eval.jsonl",
        ]
        for f in files:
            if f.exists():
                keys |= {normalize_question(r.get("question", "")) for r in _read_jsonl(f)}
        if pathlib.Path(config.HMGS_DATA_PATH).exists():
            keys |= {normalize_question(q.question) for q in DataProcessor.build_gold_eval_set()}
        keys.discard("")
        return keys

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def _char_chunk(text: str, doc_id: str, source: str) -> list["CorpusChunk"]:
        """Character-based chunking via RecursiveCharacterTextSplitter (original method)."""
        raw_chunks = _TEXT_SPLITTER.split_text(text)
        madde_nos = _assign_madde_nos(raw_chunks)
        chunks: list[CorpusChunk] = []
        for i, chunk in enumerate(raw_chunks):
            if len(chunk) < config.MIN_CHUNK_CHARS:
                continue
            chunks.append(CorpusChunk(
                chunk_id=f"{source}_{doc_id}_{i}",
                doc_id=doc_id,
                text=chunk,
                source=source,
                char_len=len(chunk),
                madde_no=madde_nos[i],
            ))
        return chunks

    @staticmethod
    def _article_chunk(text: str, doc_id: str, source: str) -> list["CorpusChunk"]:
        """Article-level chunking: one chunk per article (title + heading +
        body), oversized articles sub-split with overlap.

        Every article is kept whatever its length ("Türkiye Devleti bir
        Cumhuriyettir." is a whole article); short heading-less text (a
        section header, a preamble fragment) is merged into the next article,
        and a short trailing sub-chunk of a long article into the one before.
        """
        texts: list[tuple[str, "str | None"]] = []
        pending = ""
        for part in _article_parts(text):
            part = part.strip()
            if not part:
                continue
            madde_no = _madde_no_from_text(part)
            if madde_no is None and len(part) < config.MIN_CHUNK_CHARS:
                pending = f"{pending}\n\n{part}".strip()
                continue
            if pending:
                part, pending = f"{pending}\n\n{part}", ""
            if len(part) <= config.CHUNK_SIZE:
                texts.append((part, madde_no))
                continue
            # Article is larger than CHUNK_SIZE — sub-split it.
            subs = _TEXT_SPLITTER.split_text(part)
            merged: list[str] = []
            for sub in subs:
                if merged and len(sub) < config.MIN_CHUNK_CHARS:
                    merged[-1] = f"{merged[-1]}\n{sub}"
                else:
                    merged.append(sub)
            # Continuation chunks inherit the article of the heading chunk.
            texts.extend(zip(merged, _assign_madde_nos(merged, carry=madde_no)))
        if pending:
            if texts:
                last, no = texts[-1]
                texts[-1] = (f"{last}\n\n{pending}", no)
            else:
                texts.append((pending, None))

        return [
            CorpusChunk(
                chunk_id=f"{source}_{doc_id}_{i}",
                doc_id=doc_id,
                text=chunk_text,
                source=source,
                char_len=len(chunk_text),
                madde_no=madde_no,
            )
            for i, (chunk_text, madde_no) in enumerate(texts)
        ]

    @staticmethod
    def chunk_text(text: str, doc_id: str, source: str) -> list["CorpusChunk"]:
        """Split text into overlapping chunks.

        When config.ARTICLE_CHUNKING_ENABLED is True, respects Turkish law article
        boundaries (MADDE regex) before falling back to RecursiveCharacterTextSplitter
        for oversized articles.  When False (default), uses plain character-based
        splitting via RecursiveCharacterTextSplitter.

        Returns [] for texts shorter than CORPUS_DOC_MIN_CHARS.
        """
        if len(text) < config.CORPUS_DOC_MIN_CHARS:
            return []

        if config.ARTICLE_CHUNKING_ENABLED:
            return DataProcessor._article_chunk(text, doc_id, source)
        return DataProcessor._char_chunk(text, doc_id, source)

    # ------------------------------------------------------------------
    # Corpus builder (generator)
    # ------------------------------------------------------------------

    def build_corpus_chunks(self, holdout: bool = False) -> Iterator[CorpusChunk]:
        """Generator — yields CorpusChunk objects for every corpus row.

        Deduplicates by text hash so each unique legal passage appears once.
        build_relevant_chunk_map uses context-hash matching to correctly
        resolve the canonical chunk even when the query's row was deduplicated.
        Also loads supplementary law texts from extra_laws.jsonl if present
        (cleaned by :func:`data.extra_laws_cleaner.clean_extra_law_records`).

        Every yielded ``chunk_id`` is unique: scraped records that share a
        ``doc_id`` get a ``__dupN`` suffix so FAISS ids / metadata do not collide.

        Args:
            holdout: Deprecated and ignored.  It removed the kaggle eval
                contexts from the index, which made their gold passages
                unretrievable (with 240 distinct contexts every one was held
                out); see :meth:`kaggle_eval_df` for the leakage control
                that replaced it.
        """
        if holdout:
            import warnings
            warnings.warn("build_corpus_chunks(holdout=True) is ignored: eval "
                          "passages stay in the index", DeprecationWarning, stacklevel=2)
        seen_hashes: set[str] = set()
        seen_ids: set[str] = set()
        kept = 0
        skipped = 0
        renamed = 0

        def _emit(chunk: CorpusChunk) -> CorpusChunk:
            nonlocal renamed
            cid = chunk.chunk_id
            k = 1
            while cid in seen_ids:
                k += 1
                cid = f"{chunk.chunk_id}__dup{k}"
            if cid != chunk.chunk_id:
                renamed += 1
                chunk.chunk_id = cid
            seen_ids.add(cid)
            return chunk

        corpus_df = self.get_corpus_rows()

        for row in corpus_df.itertuples(index=False):
            context = row.context if pd.notna(row.context) else ""
            if not context:
                continue
            for chunk in DataProcessor.chunk_text(str(context), str(row.id), str(row.source)):
                text_hash = hashlib.md5(chunk.text.encode()).hexdigest()
                if text_hash in seen_hashes:
                    skipped += 1
                    continue
                seen_hashes.add(text_hash)
                kept += 1
                yield _emit(chunk)

        # Load supplementary law texts (HMK, TTK, İYUK, İİK, VUK, DMK, …)
        extra_path = pathlib.Path(config.BASE_DIR) / "data" / "extra_laws.jsonl"
        if extra_path.exists():
            from utils import read_jsonl
            from data.extra_laws_cleaner import clean_extra_law_records
            records, clean_stats = clean_extra_law_records(list(read_jsonl(extra_path)))
            extra_kept = 0
            seen_docs: collections.Counter = collections.Counter()
            for entry in records:
                text   = entry.get("text", "")
                source = entry.get("source", "")
                doc_id = entry.get("doc_id", "")
                seen_docs[(source, doc_id)] += 1
                if seen_docs[(source, doc_id)] > 1:
                    doc_id = f"{doc_id}__dup{seen_docs[(source, doc_id)]}"
                for chunk in DataProcessor.chunk_text(text, doc_id, source):
                    text_hash = hashlib.md5(chunk.text.encode()).hexdigest()
                    if text_hash in seen_hashes:
                        skipped += 1
                        continue
                    seen_hashes.add(text_hash)
                    kept += 1
                    extra_kept += 1
                    yield _emit(chunk)
            print(f"[build_corpus_chunks] extra_laws: +{extra_kept} chunks from supplementary laws "
                  f"(cleaning: {clean_stats})")

        print(f"[build_corpus_chunks] kept={kept}, skipped={skipped} duplicate chunks, "
              f"renamed={renamed} colliding chunk_ids")

    # ------------------------------------------------------------------
    # QA set builders
    # ------------------------------------------------------------------

    def _rows_to_qa_examples(self, df: pd.DataFrame) -> list[QAExample]:
        examples: list[QAExample] = []
        for row in df.itertuples(index=False):
            raw = row._asdict()
            context_val = raw.get("context", "")
            context_str = "" if pd.isna(context_val) else str(context_val)

            source_val = raw.get("source", "")
            source_str = "" if pd.isna(source_val) else str(source_val)

            data_type_val = raw.get("data_type", "")
            data_type_str = "" if pd.isna(data_type_val) else str(data_type_val)

            question_val = raw.get("question", "")
            question_str = str(question_val) if pd.notna(question_val) else ""

            answer_val = raw.get("answer", "")
            answer_str = str(answer_val) if pd.notna(answer_val) else ""

            examples.append(QAExample(
                query_id=str(raw.get("id", "")),
                question=question_str,
                answer=answer_str,
                context=context_str,
                source=source_str,
                data_type=data_type_str,
            ))
        return examples

    def build_qa_eval_set(self) -> list[QAExample]:
        """The ``kaggle`` eval set (see :meth:`kaggle_eval_df`)."""
        return self._rows_to_qa_examples(self.kaggle_eval_df())

    def _without_eval_questions(self, df: pd.DataFrame) -> pd.DataFrame:
        keys = self.eval_question_keys()
        mask = df["question"].fillna("").map(normalize_question).isin(keys)
        if mask.any():
            print(f"[train data] dropped {int(mask.sum())} rows whose question is in an eval set")
        return df[~mask]

    def build_qa_train_set(self) -> list[QAExample]:
        """QA train set: the ``train`` split minus any question of an eval set."""
        return self._rows_to_qa_examples(self._without_eval_questions(self.get_qa_split("train")))

    def build_kaggle_train_set(self) -> list[QAExample]:
        """Kaggle rows (with contexts) for supervised retrieval training: every
        kaggle row except the eval rows and any question of an eval set."""
        df = self.get_corpus_rows()
        eval_ids = set(self.kaggle_eval_df()["id"])
        df = df[df["context"].notna() & ~df["id"].isin(eval_ids)]
        return self._rows_to_qa_examples(self._without_eval_questions(df))

    @staticmethod
    def build_gold_eval_set(hmgs_path=None) -> list[QAExample]:
        """Load the HMGS gold test set, filtered to laws present in the corpus.

        Reads the HMGS CSV, maps kaynak names to corpus source names via
        config.HMGS_SOURCE_MAP, and drops rows whose kaynak has no corpus
        counterpart (no chunks to retrieve against).

        Args:
            hmgs_path: Path to the HMGS CSV. Defaults to config.HMGS_DATA_PATH.

        Returns:
            List of QAExample whose source matches a corpus source.
        """
        import logging
        log = logging.getLogger(__name__)

        if hmgs_path is None:
            hmgs_path = config.HMGS_DATA_PATH
        df = pd.read_csv(hmgs_path, encoding="utf-8-sig")

        required = {"soru", "cevap", "kaynak"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"HMGS CSV is missing columns: {missing}")

        import re as _re

        # MC-reference answers reference exam option numbers (e.g. "Yalnız I",
        # "I, II ve III") that are not present in the truncated question text.
        # These cannot be evaluated with automatic metrics — always filtered.
        _MC_RE = _re.compile(
            r'^(Yalnız|Sadece)\s+(I{1,3}|IV|V)'
            r'|^(I{1,3}|IV|V)\s*(,\s*(I{1,3}|IV|V))+'
            r'|^(I{1,3}|IV|V)\s+ve\s+(I{1,3}|IV|V)',
            _re.IGNORECASE,
        )

        # Sources excluded despite corpus coverage (see config.HMGS_DROPPED_SOURCES).
        _DROPPED_SOURCES = config.HMGS_DROPPED_SOURCES

        source_map = config.HMGS_SOURCE_MAP
        examples: list[QAExample] = []
        skipped = 0
        skipped_mc = 0
        skipped_src = 0

        # to_dict keeps column names such as "veri türü" (itertuples renames them)
        for i, raw in enumerate(df.to_dict("records")):

            def _str(val):
                return "" if pd.isna(val) else str(val)

            kaynak = _str(raw.get("kaynak", ""))

            if kaynak in _DROPPED_SOURCES:
                skipped_src += 1
                continue

            mapped_source = source_map.get(kaynak)
            if mapped_source is None:
                skipped += 1
                continue

            answer = _str(raw.get("cevap", ""))
            if _MC_RE.match(answer.strip()):
                skipped_mc += 1
                continue

            examples.append(QAExample(
                query_id=f"hmgs_{i:04d}",
                question=_str(raw.get("soru", "")),
                answer=answer,
                context="",
                source=mapped_source,
                data_type=_str(raw.get("veri türü", "")),
            ))

        log.info(
            "build_gold_eval_set: kept=%d  dropped=no_corpus:%d  mc_ref:%d  noisy_src:%d",
            len(examples), skipped, skipped_mc, skipped_src,
        )
        expected = config.HMGS_EVAL_EXPECTED
        if len(examples) < expected * 0.8:
            log.warning(
                "build_gold_eval_set: only %d examples built, expected ~%d. "
                "Check HMGS CSV filtering or HMGS_SOURCE_MAP.",
                len(examples), expected,
            )
        return examples

    # ------------------------------------------------------------------
    # Ground-truth relevance map
    # ------------------------------------------------------------------

    @staticmethod
    def build_turkish_legal_rag_eval_set(path=None) -> list[QAExample]:
        """Load the committed turkish_legal_rag eval set (see scripts/16, 17).

        Rows keep their explicit ``source`` and ``madde_no`` so that
        :meth:`build_relevant_chunk_map` can label gold chunks directly.
        With ``config.TLR_USE_LABEL_FIXES`` the labels checked against the
        law text by scripts/17 are used, otherwise the original HF labels.
        """
        p = pathlib.Path(path) if path else pathlib.Path(config.TLR_DATA_PATH)
        fixed = config.TLR_USE_LABEL_FIXES
        rows = list(_read_jsonl(p))
        if not fixed:
            # original HF labels and conflict exclusions (rows re-admitted by
            # scripts/17 carry label_conflict_hf=True)
            rows = [{**r, "madde_no": r.get("madde_no_hf", r.get("madde_no")),
                     "label_conflict": r.get("label_conflict_hf", r.get("label_conflict"))}
                    for r in rows]
        return [
            QAExample(
                query_id=r["query_id"],
                question=r["question"],
                answer=r.get("answer", ""),
                context=r.get("context", ""),
                source=r.get("source", ""),
                data_type=r.get("data_type", ""),
                madde_no=r.get("madde_no"),
                hf_row_id=r.get("hf_row_id"),
            )
            for r in rows
            if not r.get("label_conflict")
        ]

    @staticmethod
    def build_relevant_chunk_map(
        corpus_chunks: list,           # list[CorpusChunk]
        qa_examples: list,             # list[QAExample]
        retriever=None,                # kept for API compatibility, ignored
        return_coverage: bool = False, # if True, return (rel_map, coverage_dict)
    ) -> "dict | tuple[dict, dict]":
        """
        Build ground-truth relevance map using source/doc_id join.
        Model-independent: does NOT use embeddings to define relevance.

        Strategy (first one that yields chunks wins; see the ``_label_*``
        functions):
        0. gold_source_ids: exact chunk IDs from evaluator benchmark
        0.5 explicit source + madde_no fields (turkish_legal_rag): chunks of that
           law whose article matches exactly
        1. Context hash match: re-chunk qa.context and match by text hash
        2. doc_id match: chunk.doc_id == qa.query_id
        2.5. Answer substring: chunk.text contains a significant portion of qa.answer
        3. Article-level match: extract madde number from question/answer, match chunks
           whose text starts with "MADDE N" in the gold source law.
        3.5 Silver lexical (optional, config.RELEVANCE_SILVER_LEXICAL):
           Within gold source law only, score chunks by normalized token overlap with
           question+answer; label top-m above threshold.  Tagged as silver.

        Queries with no match from any strategy are left with an empty relevant list
        and are excluded from retrieval metrics (compute_all_metrics skips them).

        Args:
            corpus_chunks:   All CorpusChunk objects in the index.
            qa_examples:     QAExample objects (or plain dicts) to label.
            retriever:       Ignored; kept for API compatibility.
            return_coverage: If True, return a (relevant_map, coverage_dict) tuple
                             instead of just relevant_map.

        Returns:
            relevant_map: dict mapping query_id -> list of relevant chunk_ids.
            coverage_dict (only when return_coverage=True): dict with per-strategy
                counts, label_strategy breakdown and the number of supplied gold
                chunk ids missing from the corpus.
        """
        import logging
        log = logging.getLogger(__name__)

        index = _LabelIndex(corpus_chunks)
        strategies = [
            ("gold", _label_gold_ids),
            ("explicit_madde", _label_explicit_madde),
            ("context_hash", _label_context_hash),
            ("doc_id", _label_doc_id),
            ("answer_substr", _label_answer_substring),
            ("article", _label_article_mention),
        ]
        if config.RELEVANCE_SILVER_LEXICAL:
            strategies.append(("silver_lexical", _label_silver_lexical))

        relevant_map: dict[str, list[str]] = {}
        label_strategy_map: dict[str, str] = {}
        counts = collections.Counter({name: 0 for name in (
            "gold", "explicit_madde", "context_hash", "doc_id",
            "answer_substr", "article", "silver_lexical")})
        missing_gold_ids = 0
        for qa in qa_examples:
            q = _QAFields.of(qa)
            missing_gold_ids += sum(1 for g in q.gold_ids if g not in index.valid_ids)
            relevant: list[str] = []
            for name, label_fn in strategies:
                relevant = label_fn(q, index)
                if relevant:
                    counts[name] += 1
                    label_strategy_map[q.query_id] = name
                    break
            relevant_map[q.query_id] = relevant

        n_total = len(qa_examples)
        unlabeled = sum(1 for v in relevant_map.values() if not v)
        labeled = n_total - unlabeled
        log.info(
            "build_relevant_chunk_map coverage: total=%d  labeled=%d (%.0f%%)  "
            "unlabeled=%d  by_strategy=%s",
            n_total, labeled, 100 * labeled / n_total if n_total else 0,
            unlabeled, dict(counts),
        )
        if unlabeled:
            log.warning(
                "build_relevant_chunk_map: %d/%d queries are unlabeled (no article "
                "number found, no context/answer match, silver disabled or below "
                "threshold).  These queries are excluded from retrieval metrics.",
                unlabeled, n_total,
            )
        if missing_gold_ids:
            # A missing gold id is unreachable gold, not a smaller gold set: the
            # supplied ids do not match this corpus (stale ids / other chunking).
            log.warning(
                "build_relevant_chunk_map: %d supplied gold chunk ids are not in the "
                "corpus; recall over the remaining ids is optimistic.", missing_gold_ids,
            )

        if not return_coverage:
            return relevant_map

        coverage = {
            "total": n_total,
            "labeled": labeled,
            "unlabeled": unlabeled,
            "by_strategy": dict(counts),
            "missing_gold_ids": missing_gold_ids,
            "label_strategy_per_query": label_strategy_map,
        }
        return relevant_map, coverage

    # ------------------------------------------------------------------
    # JSONL I/O
    # ------------------------------------------------------------------

    @staticmethod
    def save_jsonl(items, path) -> int:
        """Write items to a JSONL file.  Creates parent dirs.  Returns count."""
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with p.open("w", encoding="utf-8") as f:
            for item in items:
                record = asdict(item) if hasattr(item, "__dataclass_fields__") else item
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count

    @staticmethod
    def load_jsonl(path) -> list[dict]:
        """Load a JSONL file and return a list of raw dicts.

        Delegates to :func:`utils.read_jsonl` which logs a warning (with file
        name and 1-based line number) and skips any line that cannot be parsed
        as JSON.
        """
        from utils import read_jsonl
        p = pathlib.Path(path)
        MAX_JSONL_BYTES = 2 * 1024 ** 3  # 2 GB
        file_size = p.stat().st_size
        if file_size > MAX_JSONL_BYTES:
            raise ValueError(
                f"JSONL file too large to load: {file_size / 1024**3:.1f} GB > 2 GB limit: {p}"
            )
        return list(read_jsonl(p))
