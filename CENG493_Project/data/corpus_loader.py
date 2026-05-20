"""Custom corpus resolution: directory of docs or pre-chunked JSONL."""
from __future__ import annotations
import json
import pathlib
from typing import TYPE_CHECKING

import config
from data.data_processor import DataProcessor, CorpusChunk

if TYPE_CHECKING:
    pass


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
