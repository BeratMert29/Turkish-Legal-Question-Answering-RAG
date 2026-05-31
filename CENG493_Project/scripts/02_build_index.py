"""Build FAISS index from corpus chunks."""
import os, sys
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
import json
import time
import sys
from pathlib import Path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)
import argparse
import config
from data.data_processor import DataProcessor
from data.corpus_loader import resolve_corpus, load_corpus_jsonl
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from corpus chunks")
    parser.add_argument("--corpus", default=None, help="Path to pre-chunked corpus JSONL")
    parser.add_argument("--docs-path", default=None, dest="docs_path", help="Directory of .txt/.pdf docs to chunk and index")
    args = parser.parse_args()

    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Load corpus chunks
    corpus_path = resolve_corpus(args.corpus, args.docs_path)
    print(f"Loading corpus from {corpus_path}")
    chunks = load_corpus_jsonl(corpus_path)  # auto-normalizes evaluator format
    texts = [c['text'] for c in chunks]
    metadata = [{'chunk_id': c['chunk_id'], 'doc_id': c['doc_id'], 'source': c['source'], 'text': c['text']} for c in chunks]
    print(f"  Loaded {len(chunks)} chunks")

    # Load embedding model
    print(f"\nLoading embedding model: {config.EMBEDDING_MODEL}")
    print("(First run downloads ~2.2 GB — this may take several minutes)")
    t0 = time.time()
    embedder = Embedder()
    embedder.load_model()
    print(f"  Model loaded ({time.time()-t0:.1f}s)")

    # Verify embedding dimension
    test_emb = embedder.encode(["test"], is_query=False, show_progress=False)
    assert test_emb.shape[1] == config.EMBEDDING_DIM, \
        f"Embedding dim mismatch: got {test_emb.shape[1]}, expected {config.EMBEDDING_DIM}"
    print(f"  Embedding dim verified: {test_emb.shape[1]}")

    # Build index
    print(f"\nEmbedding {len(texts)} chunks...")
    t0 = time.time()
    retriever = Retriever(embedder)
    retriever.build_index(texts, metadata)
    print(f"  Embedded in {time.time()-t0:.1f}s")
    print(f"  Index vectors: {retriever.index.ntotal}")

    # Save
    index_path = config.INDEX_DIR / config.INDEX_FILE
    metadata_path = config.INDEX_DIR / config.METADATA_FILE
    retriever.save_index(index_path, metadata_path)
    print(f"\n  Index saved: {index_path}")
    print(f"  Metadata saved: {metadata_path}")
    print("\n✓ Index built")

if __name__ == '__main__':
    main()
