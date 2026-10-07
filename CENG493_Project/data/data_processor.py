from dataclasses import dataclass, asdict
from typing import Iterator
import collections
import hashlib
import json
import pathlib
import re

import numpy as np
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
    Never goes below *floor* (the previous heading line).
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
        floor = line_start + 1
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

    def _get_kaggle_corpus_eval_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split kaggle rows into (corpus_df, eval_df) holding eval rows out of corpus.

        Uses an article-hash split to prevent data leakage: rows sharing the same
        context text (same article) are always kept together on the same side of the
        split.  A plain row-level sample would allow the same article to appear in
        both the FAISS corpus and the eval set under different doc_ids, leaking the
        gold context into the retrieval index.

        Algorithm:
        1. Compute MD5 of each row's context text (NaN/empty → treated as "" so
           all context-less rows stay in the corpus, not the eval set).
        2. Build a sorted, deduplicated list of unique non-empty context hashes.
        3. Assign the last N unique hashes to the eval set (deterministic, no shuffle
           needed because the list is sorted — equivalent to random_state=42 row
           sampling for uniformly-distributed hashes).
        4. Eval rows = all rows whose context hash is in the eval hash set.
        5. Corpus rows = everything else (including all NaN-context rows).

        Returns:
            corpus_df: rows NOT sampled for eval — used for FAISS index construction.
            eval_df:   rows sampled for eval — used to build the QA eval set.

        Result is cached on the instance so the expensive split is computed only once
        per pipeline run even when both build_corpus_chunks() and build_qa_eval_set()
        call this method.
        """
        if hasattr(self, "_split_cache"):
            return self._split_cache

        df = self.get_corpus_rows()

        # Step 1 — compute per-row context hash (empty string for NaN).
        def _ctx_hash(val):
            text = "" if (val is None or (isinstance(val, float) and pd.isna(val))) else str(val)
            return hashlib.md5(text.encode()).hexdigest() if text else ""

        ctx_hashes = [_ctx_hash(val) for val in df["context"]]

        # Step 2 — unique non-empty hashes, sorted for determinism.
        unique_hashes = sorted({h for h in ctx_hashes if h})
        n = min(config.QA_EVAL_EXPECTED, len(unique_hashes))

        # Step 3 — take the last N unique hashes as the eval set.
        eval_hash_set = set(unique_hashes[-n:])

        # Step 4 — partition rows via boolean mask.
        eval_mask = np.array([h in eval_hash_set for h in ctx_hashes])
        eval_df = df[eval_mask].reset_index(drop=True)
        corpus_df = df[~eval_mask].reset_index(drop=True)

        self._split_cache = (corpus_df, eval_df)
        return corpus_df, eval_df

    def get_eval_only_rows(self) -> pd.DataFrame:
        """Return only the kaggle rows held out for eval (not in the FAISS corpus)."""
        _, eval_df = self._get_kaggle_corpus_eval_split()
        return eval_df

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
            holdout: When True, the kaggle rows held out for the legacy
                ``kaggle`` eval set (see ``_get_kaggle_corpus_eval_split``)
                are NOT indexed.  Leave False (default) for the
                ``turkish_legal_rag`` / ``hmgs`` eval sets: they are labelled
                against the full kaggle law texts, and holding out would drop
                all eval laws from the corpus.
        """
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

        if holdout:
            corpus_df, _ = self._get_kaggle_corpus_eval_split()
        else:
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
              f"renamed={renamed} colliding chunk_ids, holdout={holdout}")

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
        """Build QA eval set from the held-out kaggle split (data leakage fix).

        Eval rows are the same subset held out from the FAISS corpus by
        _get_kaggle_corpus_eval_split(), so retrieval is always evaluated on
        unseen queries.  Uses random_state=42 for reproducibility.
        """
        _, eval_df = self._get_kaggle_corpus_eval_split()
        return self._rows_to_qa_examples(eval_df)

    def build_qa_train_set(self) -> list[QAExample]:
        """Build QA train set (train split)."""
        df = self.get_qa_split("train")
        return self._rows_to_qa_examples(df)

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

        for i, row in enumerate(df.itertuples(index=False)):
            raw = row._asdict()

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
        """Load the committed turkish_legal_rag eval set (see scripts/16).

        Rows keep their explicit ``source`` and ``madde_no`` so that
        :meth:`build_relevant_chunk_map` can label gold chunks directly.
        """
        p = pathlib.Path(path) if path else pathlib.Path(config.TLR_DATA_PATH)
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
            for r in _read_jsonl(p)
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

        Strategy (in order):
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
                counts and label_strategy breakdown.
        """
        import logging
        log = logging.getLogger(__name__)

        # Build lookup structures
        hash_to_chunk_ids: dict[str, list[str]] = {}
        by_source: dict[str, list] = {}
        for chunk in corpus_chunks:
            h = hashlib.md5(chunk.text.encode()).hexdigest()
            hash_to_chunk_ids.setdefault(h, []).append(chunk.chunk_id)
            by_source.setdefault(chunk.source, []).append(chunk)

        # Issue 11 fix: build valid_ids once outside the per-query loop (O(N) not O(N*Q))
        valid_ids = set(c.chunk_id for c in corpus_chunks)

        inherited = _inherited_madde_nos(corpus_chunks)

        relevant_map: dict[str, list[str]] = {}
        # label_strategy tracks how each query was labeled (for coverage reporting)
        label_strategy_map: dict[str, str] = {}
        # Coverage counters — one per labeling strategy
        labeled_s0 = labeled_s05 = labeled_s1 = labeled_s2 = labeled_s25 = labeled_s3 = 0
        labeled_silver = 0
        unlabeled = 0
        # Silver config (read once for performance)
        silver_enabled = config.RELEVANCE_SILVER_LEXICAL
        silver_top_m = config.SILVER_TOP_M
        silver_threshold = config.SILVER_THRESHOLD

        for qa in qa_examples:
            relevant: list[str] = []

            # Unified field accessors: support both QAExample dataclass and plain dict.
            _is_dict = isinstance(qa, dict)
            qa_query_id  = qa["query_id"]  if _is_dict else qa.query_id
            qa_context   = qa.get("context", "") if _is_dict else qa.context
            qa_answer    = qa.get("answer", "")  if _is_dict else qa.answer
            qa_source    = qa.get("source", "")  if _is_dict else qa.source
            qa_question  = qa.get("question", "") if _is_dict else getattr(qa, "question", "")

            # Strategy 0: gold_source_ids — exact chunk IDs supplied by the
            # evaluator's benchmark (gold_benchmark.json / rag_eval.json).
            # These are chunk_ids that exist verbatim in the corpus, so we use
            # them directly without any heuristic matching.
            gold_ids = qa.get("gold_source_ids") if _is_dict else getattr(qa, "gold_source_ids", None)
            if gold_ids:
                relevant = [gid for gid in gold_ids if gid in valid_ids]
                if relevant:
                    labeled_s0 += 1
                    label_strategy_map[qa_query_id] = "gold"
                    relevant_map[qa_query_id] = relevant
                    continue  # Skip remaining strategies — ground truth is exact.

            # Strategy 0.5: explicit source + madde_no fields (turkish_legal_rag).
            # The gold article is given directly, so label the corpus chunks of
            # that law and article without any text heuristics.
            explicit_madde = qa.get("madde_no") if _is_dict else getattr(qa, "madde_no", None)
            if explicit_madde and qa_source:
                relevant = [
                    c.chunk_id for c in by_source.get(qa_source, [])
                    if _chunk_matches_madde_str(c, explicit_madde, inherited.get(c.chunk_id))
                ]
                if relevant:
                    labeled_s05 += 1
                    label_strategy_map[qa_query_id] = "explicit_madde"
                    relevant_map[qa_query_id] = relevant
                    continue

            # Strategy 1: context-hash match — re-chunk qa.context using the same
            # chunking path as the corpus index build (article or char chunking).
            # This ensures hashes match exactly, regardless of ARTICLE_CHUNKING_ENABLED.
            if qa_context:
                for corpus_chunk in DataProcessor.chunk_text(qa_context, qa_query_id, qa_source or ""):
                    h = hashlib.md5(corpus_chunk.text.encode()).hexdigest()
                    relevant.extend(hash_to_chunk_ids.get(h, []))
                # deduplicate while preserving order
                seen: set[str] = set()
                deduped = []
                for cid in relevant:
                    if cid not in seen:
                        seen.add(cid)
                        deduped.append(cid)
                relevant = deduped
            if relevant:
                labeled_s1 += 1
                label_strategy_map[qa_query_id] = "context_hash"

            # Strategy 2: doc_id match — used when context is empty/missing
            if not relevant:
                relevant = [c.chunk_id for c in corpus_chunks if c.doc_id == qa_query_id]
                if relevant:
                    labeled_s2 += 1
                    label_strategy_map[qa_query_id] = "doc_id"

            # Strategy 2.5: answer substring match — for gold sets with known answers (e.g. HMGS)
            # Find chunks that contain a significant portion of the answer text.
            if not relevant and qa_answer and len(qa_answer) >= 40:
                answer_lower = qa_answer.lower().strip()
                search_str = answer_lower[:80] if len(answer_lower) >= 80 else answer_lower
                candidate_chunks = by_source.get(qa_source, corpus_chunks) if qa_source else corpus_chunks
                relevant = [c.chunk_id for c in candidate_chunks if search_str in c.text.lower()]
                if relevant:
                    labeled_s25 += 1
                    label_strategy_map[qa_query_id] = "answer_substr"

            # Strategy 3: article-level match — extract the madde (article) number from
            # the question and answer text.  Match only corpus chunks that belong to that
            # specific article in the correct source law.
            #
            # Design rationale: the previous strategy assigned the first N chunks of the
            # entire source law as relevant, which is arbitrary and inflates
            # Recall/MRR/nDCG for queries that do not cite a specific article.
            # Article-level matching is precise but requires the query to reference an
            # article number explicitly.  Queries where no article can be determined are
            # left with an empty relevant set and are excluded from retrieval metrics by
            # compute_all_metrics (queries with no ground-truth are always skipped).
            # This is the correct behavior: we should not compute retrieval metrics for
            # queries whose ground-truth relevance is unknown.
            if not relevant and qa_source:
                madde_no = _extract_madde_no(qa_question, qa_answer)
                if madde_no is not None:
                    source_chunks = by_source.get(qa_source, [])
                    relevant = [
                        c.chunk_id for c in source_chunks
                        if _chunk_matches_article(c, madde_no)
                    ]
                    if relevant:
                        labeled_s3 += 1
                        label_strategy_map[qa_query_id] = "article"
                # If madde_no is None: leave relevant=[] → query will be unlabeled
                # and excluded from retrieval metrics.

            # Strategy 3.5: silver lexical labeling — optional, off by default.
            # Within the gold source law only, rank chunks by normalized token overlap
            # with (question + answer) and label the top-m above threshold.
            # Uses Turkish-aware case folding (İ→i, I→ı).
            # This is a HEURISTIC: labels are "silver" quality, not gold.
            # Do NOT use silver labels when precise article-level labels are available.
            if not relevant and qa_source and silver_enabled:
                source_chunks = by_source.get(qa_source, [])
                if source_chunks:
                    q_tokens = _turkish_tokenize(f"{qa_question} {qa_answer}")
                    scored = [
                        (c, _silver_lexical_score(q_tokens, c.text))
                        for c in source_chunks
                    ]
                    scored.sort(key=lambda x: -x[1])
                    relevant = [
                        c.chunk_id for c, score in scored[:silver_top_m]
                        if score >= silver_threshold
                    ]
                    if relevant:
                        labeled_silver += 1
                        label_strategy_map[qa_query_id] = "silver_lexical"

            if not relevant:
                unlabeled += 1
                # Keep the key with empty list — retrieval_metrics.py already
                # excludes queries with no ground-truth from metric computation.

            relevant_map[qa_query_id] = relevant

        n_total = len(qa_examples)
        labeled = n_total - unlabeled
        log.info(
            "build_relevant_chunk_map coverage: "
            "total=%d  labeled=%d (%.0f%%)  unlabeled=%d  "
            "[s0(gold)=%d s0.5(explicit_madde)=%d s1(ctx_hash)=%d s2(doc_id)=%d s2.5(ans_substr)=%d "
            "s3(article)=%d s3.5(silver_lexical)=%d]",
            n_total, labeled, 100 * labeled / n_total if n_total else 0,
            unlabeled, labeled_s0, labeled_s05, labeled_s1, labeled_s2, labeled_s25,
            labeled_s3, labeled_silver,
        )
        if unlabeled:
            log.warning(
                "build_relevant_chunk_map: %d/%d queries are unlabeled (no article "
                "number found, no context/answer match, silver disabled or below "
                "threshold).  These queries are excluded from retrieval metrics.",
                unlabeled, n_total,
            )

        if not return_coverage:
            return relevant_map

        coverage = {
            "total": n_total,
            "labeled": labeled,
            "unlabeled": unlabeled,
            "by_strategy": {
                "gold":          labeled_s0,
                "explicit_madde": labeled_s05,
                "context_hash":  labeled_s1,
                "doc_id":        labeled_s2,
                "answer_substr": labeled_s25,
                "article":       labeled_s3,
                "silver_lexical": labeled_silver,
            },
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
