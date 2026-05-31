from dataclasses import dataclass, asdict
from typing import Iterator
import hashlib
import json
import pathlib
import re

import numpy as np
import pandas as pd
import config
from langchain_text_splitters import RecursiveCharacterTextSplitter


_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# Matches the start of a Turkish law article heading (e.g. "MADDE 1", "Madde 12").
_ARTICLE_RE = re.compile(r"(?m)(?=^\s*MADDE\s+\d)", re.IGNORECASE)


@dataclass
class CorpusChunk:
    chunk_id: str   # f"{source}_{doc_id}_{chunk_index}"
    doc_id: str
    text: str
    source: str
    char_len: int


@dataclass
class QAExample:
    query_id: str
    question: str
    answer: str
    context: str    # "" for test/train rows (null in CSV)
    source: str
    data_type: str


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
            ))
        return chunks

    @staticmethod
    def _article_chunk(text: str, doc_id: str, source: str) -> list["CorpusChunk"]:
        """Article-level chunking: split at MADDE boundaries, sub-split oversized articles."""
        parts = _ARTICLE_RE.split(text)
        chunks: list[CorpusChunk] = []
        chunk_index = 0
        for part in parts:
            part = part.strip()
            if not part or len(part) < config.MIN_CHUNK_CHARS:
                continue
            if len(part) <= config.CHUNK_SIZE:
                chunks.append(CorpusChunk(
                    chunk_id=f"{source}_{doc_id}_{chunk_index}",
                    doc_id=doc_id,
                    text=part,
                    source=source,
                    char_len=len(part),
                ))
                chunk_index += 1
            else:
                # Article is larger than CHUNK_SIZE — sub-split it.
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                for sub_chunk in sub_splitter.split_text(part):
                    if len(sub_chunk) < config.MIN_CHUNK_CHARS:
                        continue
                    chunks.append(CorpusChunk(
                        chunk_id=f"{source}_{doc_id}_{chunk_index}",
                        doc_id=doc_id,
                        text=sub_chunk,
                        source=source,
                        char_len=len(sub_chunk),
                    ))
                    chunk_index += 1
        return chunks

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

        if getattr(config, "ARTICLE_CHUNKING_ENABLED", False):
            return DataProcessor._article_chunk(text, doc_id, source)
        return DataProcessor._char_chunk(text, doc_id, source)

    # ------------------------------------------------------------------
    # Corpus builder (generator)
    # ------------------------------------------------------------------

    def build_corpus_chunks(self) -> Iterator[CorpusChunk]:
        """Generator — yields CorpusChunk objects for every corpus row.

        Deduplicates by text hash so each unique legal passage appears once.
        build_relevant_chunk_map uses context-hash matching to correctly
        resolve the canonical chunk even when the query's row was deduplicated.
        Also loads supplementary law texts from extra_laws.jsonl if present.

        Eval rows are held out (data leakage fix): only the complement of the
        QA eval sample is indexed into FAISS.
        """
        seen_hashes: set[str] = set()
        kept = 0
        skipped = 0

        corpus_df, _ = self._get_kaggle_corpus_eval_split()

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
                yield chunk

        # Load supplementary law texts (HMK, TTK, İYUK, İİK, VUK, DMK, …)
        extra_path = pathlib.Path(config.BASE_DIR) / "data" / "extra_laws.jsonl"
        if extra_path.exists():
            extra_kept = 0
            with open(extra_path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    text   = entry.get("text", "")
                    source = entry.get("source", "")
                    doc_id = entry.get("doc_id", "")
                    for chunk in DataProcessor.chunk_text(text, doc_id, source):
                        text_hash = hashlib.md5(chunk.text.encode()).hexdigest()
                        if text_hash in seen_hashes:
                            skipped += 1
                            continue
                        seen_hashes.add(text_hash)
                        kept += 1
                        extra_kept += 1
                        yield chunk
            print(f"[build_corpus_chunks] extra_laws: +{extra_kept} chunks from supplementary laws")

        print(f"[build_corpus_chunks] kept={kept}, skipped={skipped} duplicate chunks")

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

        # VUK rows are misattributed (3/5 questions are actually about HMK /
        # Avukatlık Kanunu) — drop the entire source to avoid noise.
        _DROPPED_SOURCES = {"213 sayılı Vergi Usul Kanunu"}

        source_map = getattr(config, "HMGS_SOURCE_MAP", {})
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
        expected = getattr(config, "HMGS_EVAL_EXPECTED", None)
        if expected and len(examples) < expected * 0.8:
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
    def build_relevant_chunk_map(
        corpus_chunks: list,          # list[CorpusChunk]
        qa_examples: list,            # list[QAExample]
        retriever=None,               # kept for API compatibility, ignored
    ) -> dict:                        # {query_id: [chunk_id, ...]}
        """
        Build ground-truth relevance map using source/doc_id join.
        Model-independent: does NOT use embeddings to define relevance.

        Strategy (in order):
        0. gold_source_ids: exact chunk IDs from evaluator benchmark
        1. Context hash match: re-chunk qa.context and match by text hash
        2. doc_id match: chunk.doc_id == qa.query_id
        2.5. Answer substring: chunk.text contains a significant portion of qa.answer
        3. Source match: all chunks from qa.source (for HMGS gold sets)

        Returns dict mapping query_id -> list of relevant chunk_ids.
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

        relevant_map: dict[str, list[str]] = {}
        no_match_count = 0

        for qa in qa_examples:
            relevant: list[str] = []

            # Unified field accessors: support both QAExample dataclass and plain dict.
            _is_dict = isinstance(qa, dict)
            qa_query_id = qa["query_id"] if _is_dict else qa.query_id
            qa_context  = qa.get("context", "") if _is_dict else qa.context
            qa_answer   = qa.get("answer", "") if _is_dict else qa.answer
            qa_source   = qa.get("source", "") if _is_dict else qa.source

            # Strategy 0: gold_source_ids — exact chunk IDs supplied by the
            # evaluator's benchmark (gold_benchmark.json / rag_eval.json).
            # These are chunk_ids that exist verbatim in the corpus, so we use
            # them directly without any heuristic matching.
            gold_ids = qa.get("gold_source_ids") if _is_dict else getattr(qa, "gold_source_ids", None)
            if gold_ids:
                relevant = [gid for gid in gold_ids if gid in valid_ids]
                if relevant:
                    relevant_map[qa_query_id] = relevant
                    continue  # Skip remaining strategies — ground truth is exact.

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

            # Strategy 2: doc_id match — used when context is empty/missing
            if not relevant:
                relevant = [c.chunk_id for c in corpus_chunks if c.doc_id == qa_query_id]

            # Strategy 2.5: answer substring match — for gold sets with known answers (e.g. HMGS)
            # Find chunks that contain a significant portion of the answer text.
            if not relevant and qa_answer and len(qa_answer) >= 40:
                answer_lower = qa_answer.lower().strip()
                search_str = answer_lower[:80] if len(answer_lower) >= 80 else answer_lower
                candidate_chunks = by_source.get(qa_source, corpus_chunks) if qa_source else corpus_chunks
                relevant = [c.chunk_id for c in candidate_chunks if search_str in c.text.lower()]

            # Strategy 3: source match — for gold sets without context (e.g. HMGS).
            # All chunks from the matching law are considered relevant.
            # Capped at MAX_STRATEGY3_RELEVANT (default 20) to keep metrics meaningful;
            # assigning hundreds of chunks from an entire law inflates Recall/MRR/NDCG.
            if not relevant and qa_source:
                source_chunks = by_source.get(qa_source, [])
                max_s3 = getattr(config, "MAX_STRATEGY3_RELEVANT", 20)
                relevant = [c.chunk_id for c in source_chunks[:max_s3]]

            if not relevant:
                no_match_count += 1

            relevant_map[qa_query_id] = relevant

        if no_match_count:
            log.warning(
                "build_relevant_chunk_map: %d/%d queries have no relevant chunks. "
                "Check that qa.source values match corpus chunk sources.",
                no_match_count, len(qa_examples),
            )
        matched_count = sum(1 for v in relevant_map.values() if v)
        log.info(
            "build_relevant_chunk_map: %d/%d queries have relevant chunks (%.1f%%)",
            matched_count, len(qa_examples),
            100 * matched_count / len(qa_examples) if qa_examples else 0,
        )
        if qa_examples and matched_count < len(qa_examples) * 0.5:
            log.warning(
                "build_relevant_chunk_map: fewer than 50%% of queries matched. "
                "Retrieval metrics may be unreliable. Check corpus/eval split alignment."
            )
        return relevant_map

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
        """Load a JSONL file and return a list of raw dicts."""
        p = pathlib.Path(path)
        MAX_JSONL_BYTES = 2 * 1024 ** 3  # 2 GB
        file_size = p.stat().st_size
        if file_size > MAX_JSONL_BYTES:
            raise ValueError(
                f"JSONL file too large to load: {file_size / 1024**3:.1f} GB > 2 GB limit: {p}"
            )
        results: list[dict] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        return results
