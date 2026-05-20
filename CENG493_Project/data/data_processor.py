from dataclasses import dataclass, asdict
from typing import Iterator
import hashlib
import json
import pathlib

import pandas as pd
import config
from langchain_text_splitters import RecursiveCharacterTextSplitter


_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""],
)


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

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    @staticmethod
    def chunk_text(text: str, doc_id: str, source: str) -> list[CorpusChunk]:
        """Split text into overlapping chunks using sentence-boundary-aware splitting.

        Uses RecursiveCharacterTextSplitter which respects paragraph/sentence
        boundaries before falling back to hard character limits, producing
        cleaner chunks than a pure character-offset sliding window.
        Returns [] for texts shorter than CORPUS_DOC_MIN_CHARS.
        """
        if len(text) < config.CORPUS_DOC_MIN_CHARS:
            return []

        raw_chunks = _TEXT_SPLITTER.split_text(text)
        chunks: list[CorpusChunk] = []
        for i, chunk in enumerate(raw_chunks):
            if len(chunk) < config.CHUNK_OVERLAP:
                continue
            chunks.append(CorpusChunk(
                chunk_id=f"{source}_{doc_id}_{i}",
                doc_id=doc_id,
                text=chunk,
                source=source,
                char_len=len(chunk),
            ))
        return chunks

    # ------------------------------------------------------------------
    # Corpus builder (generator)
    # ------------------------------------------------------------------

    def build_corpus_chunks(self) -> Iterator[CorpusChunk]:
        """Generator — yields CorpusChunk objects for every corpus row.

        Deduplicates by text hash so each unique legal passage appears once.
        build_relevant_chunk_map uses context-hash matching to correctly
        resolve the canonical chunk even when the query's row was deduplicated.
        Also loads supplementary law texts from extra_laws.jsonl if present.
        """
        seen_hashes: set[str] = set()
        kept = 0
        skipped = 0

        for row in self.get_corpus_rows().itertuples(index=False):
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
        """Build QA eval set sampled from kaggle split.

        Kaggle rows have question + answer + context + source, enabling
        model-independent ground-truth relevance mapping via doc_id match.
        Uses random_state=42 for reproducibility.
        """
        df = self.get_corpus_rows()  # split == "kaggle", all rows have source+context
        n = min(config.QA_EVAL_EXPECTED, len(df))
        sampled = df.sample(n=n, random_state=42)
        return self._rows_to_qa_examples(sampled)

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
        1. Exact source match: qa.source == chunk.source
        2. doc_id prefix match: chunk.doc_id starts with qa's row id
        3. Answer substring: chunk.text contains a significant portion of qa.answer
           (at least 80 chars of the answer appears in the chunk)

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

        relevant_map: dict[str, list[str]] = {}
        no_match_count = 0

        for qa in qa_examples:
            relevant: list[str] = []

            # Strategy 1: context-hash match — re-chunk qa.context and find
            # the canonical corpus chunks by text hash.  This correctly handles
            # the deduplicated corpus: the relevant chunk may be stored under a
            # different doc_id than qa.query_id if it was first seen in another row.
            if qa.context:
                for text in _TEXT_SPLITTER.split_text(qa.context):
                    if len(text) >= config.CORPUS_DOC_MIN_CHARS:
                        h = hashlib.md5(text.encode()).hexdigest()
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
                relevant = [c.chunk_id for c in corpus_chunks if c.doc_id == qa.query_id]

            # Strategy 2.5: answer substring match — for gold sets with known answers (e.g. HMGS)
            # Find chunks that contain a significant portion of the answer text.
            if not relevant and qa.answer and len(qa.answer) >= 40:
                answer_lower = qa.answer.lower().strip()
                search_str = answer_lower[:80] if len(answer_lower) >= 80 else answer_lower
                candidate_chunks = by_source.get(qa.source, corpus_chunks) if qa.source else corpus_chunks
                relevant = [c.chunk_id for c in candidate_chunks if search_str in c.text.lower()]

            # Strategy 3: source match — for gold sets without context (e.g. HMGS)
            # All chunks from the matching law are considered relevant.
            if not relevant and qa.source:
                relevant = [c.chunk_id for c in by_source.get(qa.source, [])]

            if not relevant:
                no_match_count += 1

            relevant_map[qa.query_id] = relevant

        if no_match_count:
            log.warning(
                "build_relevant_chunk_map: %d/%d queries have no relevant chunks. "
                "Check that qa.source values match corpus chunk sources.",
                no_match_count, len(qa_examples),
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
