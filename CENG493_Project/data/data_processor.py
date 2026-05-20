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


@dataclass
class CorpusChunk:
    chunk_id: str
    doc_id: str
    text: str
    source: str
    char_len: int


@dataclass
class QAExample:
    query_id: str
    question: str
    answer: str
    context: str
    source: str
    data_type: str


class DataProcessor:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self._df: pd.DataFrame | None = None

    def load_and_validate(self) -> dict:
        """Load CSV, validate columns, return summary dict."""
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

    def _get_kaggle_corpus_eval_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Article-hash based split — an article is EITHER in eval OR in corpus, never both."""
        if not hasattr(self, "_kaggle_split_cache"):
            df = self._df[self._df["split"] == "kaggle"].reset_index(drop=True)

            min_score = getattr(config, "KAGGLE_MIN_SCORE", 6)
            if "score" in df.columns:
                before = len(df)
                df = df[pd.to_numeric(df["score"], errors="coerce").fillna(0) >= min_score]
                df = df.reset_index(drop=True)
                dropped = before - len(df)
                if dropped:
                    import logging
                    logging.getLogger(__name__).info(
                        "_get_kaggle_corpus_eval_split: dropped %d rows with score < %d",
                        dropped, min_score,
                    )

            # 1. Assign every row to an article bucket by hashing the context text
            df = df.copy()
            df["_article_hash"] = df["context"].fillna("").apply(
                lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()
            )

            # 2. Collect unique article hashes and shuffle deterministically
            unique_hashes = sorted(df["_article_hash"].unique())
            rng = np.random.default_rng(seed=42)
            rng.shuffle(unique_hashes)

            # 3. Greedily assign article hashes to the eval bucket until we reach n_holdout rows
            n_holdout = getattr(config, "QA_EVAL_EXPECTED", 300)
            eval_hashes: set[str] = set()
            n_eval = 0
            for h in unique_hashes:
                if n_eval >= n_holdout:
                    break
                eval_hashes.add(h)
                n_eval += int((df["_article_hash"] == h).sum())

            # 4. Split
            eval_mask = df["_article_hash"].isin(eval_hashes)
            eval_df   = df[eval_mask].reset_index(drop=True)
            corpus_df = df[~eval_mask].reset_index(drop=True)

            # Clean up helper column
            eval_df   = eval_df.drop(columns=["_article_hash"])
            corpus_df = corpus_df.drop(columns=["_article_hash"])

            self._kaggle_split_cache = (corpus_df, eval_df)
        return self._kaggle_split_cache

    def get_corpus_rows(self) -> pd.DataFrame:
        """Kaggle rows for FAISS corpus (eval holdout excluded)."""
        self._ensure_loaded()
        corpus_df, _ = self._get_kaggle_corpus_eval_split()
        return corpus_df

    def get_eval_only_rows(self) -> pd.DataFrame:
        """Holdout eval rows (passages not indexed in FAISS)."""
        self._ensure_loaded()
        _, eval_df = self._get_kaggle_corpus_eval_split()
        return eval_df

    def get_qa_split(self, split: str) -> pd.DataFrame:
        self._ensure_loaded()
        return self._df[self._df["split"] == split].reset_index(drop=True)

    @staticmethod
    def chunk_text(text: str, doc_id: str, source: str) -> list[CorpusChunk]:
        """Chunk by article boundaries or RecursiveCharacterTextSplitter."""
        if len(text) < config.CORPUS_DOC_MIN_CHARS:
            return []

        if not getattr(config, 'ARTICLE_CHUNKING_ENABLED', True):
            return DataProcessor._char_chunk(text, doc_id, source)

        article_regex = getattr(config, 'ARTICLE_REGEX', r'(?=(?:MADDE|Madde)\s+\d+)')
        parts = re.split(article_regex, text)
        parts = [p.strip() for p in parts if p.strip()]

        article_parts = [p for p in parts if re.match(r'(?:MADDE|Madde)\s+\d+', p)]
        if len(article_parts) < 2:
            return DataProcessor._char_chunk(text, doc_id, source)

        chunks: list[CorpusChunk] = []

        for part in parts:
            madde_match = re.match(r'(?:MADDE|Madde)\s+(\d+)', part)
            if madde_match:
                madde_no = madde_match.group(1)
                base_id = f"m{madde_no}"
            else:
                base_id = "pre"

            if len(part) < config.CORPUS_DOC_MIN_CHARS:
                continue

            if len(part) <= config.CHUNK_SIZE:
                chunks.append(CorpusChunk(
                    chunk_id=f"{source}_{doc_id}_{base_id}",
                    doc_id=doc_id,
                    text=part,
                    source=source,
                    char_len=len(part),
                ))
            else:
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=0,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                sub_chunks = sub_splitter.split_text(part)
                for sub_idx, sub_text in enumerate(sub_chunks):
                    if len(sub_text) < config.CORPUS_DOC_MIN_CHARS:
                        continue
                    chunks.append(CorpusChunk(
                        chunk_id=f"{source}_{doc_id}_{base_id}_{sub_idx}",
                        doc_id=doc_id,
                        text=sub_text,
                        source=source,
                        char_len=len(sub_text),
                    ))

        return chunks

    @staticmethod
    def _char_chunk(text: str, doc_id: str, source: str) -> list[CorpusChunk]:
        """Fallback: character-window chunking only."""
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

    def build_corpus_chunks(self) -> Iterator[CorpusChunk]:
        """Yield corpus chunks; hash-dedup per text; append extra_laws.jsonl if present."""
        seen_hashes: set[str] = set()
        kept = 0
        skipped = 0

        for row in self.get_corpus_rows().itertuples(index=False):
            context = row.context if pd.notna(row.context) else ""
            if not context:
                continue
            for chunk in self.chunk_text(str(context), str(row.id), str(row.source)):
                text_hash = hashlib.md5(chunk.text.encode()).hexdigest()
                if text_hash in seen_hashes:
                    skipped += 1
                    continue
                seen_hashes.add(text_hash)
                kept += 1
                yield chunk

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
                    for chunk in self.chunk_text(text, doc_id, source):
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
        """Sample eval QA from holdout rows (not in FAISS)."""
        df = self.get_eval_only_rows()
        n = min(config.QA_EVAL_EXPECTED, len(df))
        sampled = df.sample(n=n, random_state=42)
        return self._rows_to_qa_examples(sampled)

    def build_qa_train_set(self) -> list[QAExample]:
        """Train split QA rows."""
        df = self.get_qa_split("train")
        return self._rows_to_qa_examples(df)

    @staticmethod
    def build_gold_eval_set(hmgs_path=None) -> list[QAExample]:
        """HMGS CSV to QAExamples; map kaynak via HMGS_SOURCE_MAP; drop MC refs and noisy VUK."""
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

        _MC_RE = _re.compile(
            r'^(Yalnız|Sadece)\s+(I{1,3}|IV|V)'
            r'|^(I{1,3}|IV|V)\s*(,\s*(I{1,3}|IV|V))+'
            r'|^(I{1,3}|IV|V)\s+ve\s+(I{1,3}|IV|V)',
            _re.IGNORECASE,
        )

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

    @staticmethod
    def build_relevant_chunk_map(corpus_chunks: list, qa_examples: list, retriever=None) -> dict:
        """Oracle relevant chunk ids: context-hash, doc_id, answer substring."""
        import logging
        log = logging.getLogger(__name__)

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

            if qa.context:
                for text in _TEXT_SPLITTER.split_text(qa.context):
                    if len(text) >= config.CORPUS_DOC_MIN_CHARS:
                        h = hashlib.md5(text.encode()).hexdigest()
                        relevant.extend(hash_to_chunk_ids.get(h, []))
                seen: set[str] = set()
                deduped = []
                for cid in relevant:
                    if cid not in seen:
                        seen.add(cid)
                        deduped.append(cid)
                relevant = deduped

            if not relevant:
                relevant = [c.chunk_id for c in corpus_chunks if c.doc_id == qa.query_id]

            if not relevant:
                no_match_count += 1

            relevant_map[qa.query_id] = relevant

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

    @staticmethod
    def save_jsonl(items, path) -> int:
        """Write JSONL; return row count."""
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
        """Read JSONL into list of dicts (max 2GB)."""
        p = pathlib.Path(path)
        MAX_JSONL_BYTES = 2 * 1024 ** 3
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
