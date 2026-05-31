"""Custom corpus resolution: directory of docs or pre-chunked JSONL."""
from __future__ import annotations
import json
import re
import pathlib
from typing import TYPE_CHECKING

import config
from data.data_processor import DataProcessor, CorpusChunk

if TYPE_CHECKING:
    pass


def _normalize_corpus_record(record: dict) -> dict:
    """Normalize evaluator corpus format to the project's expected schema.

    The evaluator's corpus uses a top-level ``id`` field and nests chunk
    metadata inside a ``metadata`` sub-object.  When both ``id`` and
    ``metadata`` are present but ``chunk_id`` is absent at the top level,
    we remap the fields so the rest of the pipeline sees a uniform schema.
    """
    if "chunk_id" in record:
        # Already in the project's native format — nothing to do.
        return record

    # Accept both BEIR-style _id and evaluator-style id
    raw_id = record.get("id") or record.get("_id")
    if not raw_id:
        # Unrecognised format; return as-is and let the caller surface errors.
        return record

    meta = record.get("metadata") or {}

    chunk_id = meta.get("chunk_id") or raw_id

    # Derive doc_id: prefer metadata field, otherwise strip trailing _<digits>
    # from the top-level id (e.g. "oricon_anayasa_000001" -> "oricon_anayasa").
    doc_id = meta.get("doc_id") or re.sub(r"_\d+$", "", raw_id)

    source = meta.get("source") or record.get("title", "")
    char_len = meta.get("chunk_char_count") or len(record.get("text", ""))

    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": record.get("text", ""),
        "source": source,
        "char_len": char_len,
    }


def load_txt_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def load_pdf_text(path: pathlib.Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError("pypdf required for PDF support: pip install pypdf")
    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def chunk_directory(docs_dir: pathlib.Path) -> list[CorpusChunk]:
    """Chunk all .txt/.pdf files in docs_dir into CorpusChunks."""
    chunks: list[CorpusChunk] = []
    seen_hashes: set[str] = set()
    import hashlib

    for ext in config.SUPPORTED_DOC_EXTENSIONS:
        for fpath in sorted(docs_dir.glob(f"*{ext}")):
            doc_id = fpath.stem
            source = fpath.stem
            try:
                if ext == ".pdf":
                    text = load_pdf_text(fpath)
                else:
                    text = load_txt_text(fpath)
            except Exception as exc:
                print(f"  WARNING: Skipping {fpath.name}: {exc}")
                continue
            if not text.strip():
                print(f"  WARNING: Empty content in {fpath.name}, skipping")
                continue
            for chunk in DataProcessor.chunk_text(text, doc_id, source):
                h = hashlib.md5(chunk.text.encode()).hexdigest()
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    chunks.append(chunk)

    return chunks


def load_corpus_jsonl(path: pathlib.Path) -> list[dict]:
    """Load a corpus JSONL and return records normalized to the project schema.

    Applies :func:`_normalize_corpus_record` to every row so callers always
    receive ``chunk_id``, ``doc_id``, ``text``, ``source``, and ``char_len``
    regardless of whether the file came from the evaluator or was produced by
    this project's own tooling.
    """
    records = DataProcessor.load_jsonl(path)
    return [_normalize_corpus_record(r) for r in records]


def _maybe_normalize_corpus_file(path: pathlib.Path) -> None:
    """Rewrite a corpus JSONL to the project schema when it uses the evaluator
    format (top-level ``id`` without ``chunk_id``).

    If the first record already has ``chunk_id`` the file is left untouched so
    we never rewrite a file unnecessarily.
    """
    records = DataProcessor.load_jsonl(path)
    if not records:
        return
    # Peek at the first record to decide whether normalization is needed.
    if "chunk_id" in records[0]:
        return  # Already in native format.

    print(f"  Normalizing evaluator corpus format -> {path}")
    normalized = [_normalize_corpus_record(r) for r in records]
    with open(path, "w", encoding="utf-8") as fh:
        for rec in normalized:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(normalized)} normalized records -> {path}")


def resolve_corpus(
    corpus_arg: str | None,
    docs_arg: str | None,
) -> pathlib.Path:
    """Return path to corpus_chunks JSONL to use for indexing.

    Priority:
    - corpus_arg (pre-chunked JSONL) -> return as-is
    - docs_arg (directory of docs) -> chunk, write custom JSONL, return it
    - neither -> default corpus_chunks.jsonl
    """
    if corpus_arg and docs_arg:
        raise ValueError("Provide --corpus OR --docs-path, not both")

    if corpus_arg:
        p = pathlib.Path(corpus_arg)
        if not p.exists():
            raise FileNotFoundError(f"--corpus file not found: {p}")
        # Normalize the file in-place to the project's schema if needed.
        # We detect foreign format by checking the first non-empty record.
        _maybe_normalize_corpus_file(p)
        return p

    if docs_arg:
        docs_dir = pathlib.Path(docs_arg)
        if not docs_dir.is_dir():
            raise NotADirectoryError(f"--docs-path is not a directory: {docs_dir}")
        print(f"Chunking documents from {docs_dir} ...")
        chunks = chunk_directory(docs_dir)
        if not chunks:
            raise ValueError(f"No usable chunks from {docs_dir}")
        out_path = pathlib.Path(config.PROCESSED_DIR) / config.CUSTOM_CORPUS_FILE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps({
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "source": c.source,
                    "char_len": c.char_len,
                }, ensure_ascii=False) + "\n")
        print(f"  Wrote {len(chunks)} chunks -> {out_path}")
        return out_path

    return pathlib.Path(config.PROCESSED_DIR) / "corpus_chunks.jsonl"
