#!/usr/bin/env python3
"""
15_build_graph.py — Build cross-reference graph over corpus chunks.

Reads corpus metadata (FAISS metadata.jsonl) and builds a graph of adjacency,
intra-document, and cross-document edges using retrieval/graph_builder.py.

Usage:
    python scripts/15_build_graph.py
    python scripts/15_build_graph.py --corpus /path/to/corpus.jsonl

Output:
    index/graph.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from retrieval.graph_builder import build_graph_from_metadata, save_graph, graph_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_metadata_from_index() -> list[dict]:
    """Load metadata records from config.INDEX_DIR / config.METADATA_FILE."""
    meta_path = config.INDEX_DIR / config.METADATA_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {meta_path}\n"
            "Run scripts/02_build_index.py first to create the FAISS index."
        )
    records = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_metadata_from_corpus(corpus_path: Path) -> list[dict]:
    """Load metadata from a corpus.jsonl file (evaluator format)."""
    from data.corpus_loader import load_corpus_jsonl

    raw_chunks = load_corpus_jsonl(corpus_path)
    # Ensure each record has the fields graph_builder expects
    records = []
    for r in raw_chunks:
        records.append({
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "text": r.get("text", ""),
            "source": r["source"],
        })
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cross-reference graph for legal corpus.")
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        help="Path to corpus.jsonl (evaluator format). If not given, uses index/metadata.jsonl.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Build Cross-Reference Graph")
    print("=" * 60)

    # Load metadata
    if args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            sys.exit(f"ERROR: Corpus file not found: {corpus_path}")
        print(f"Loading corpus from: {corpus_path}")
        metadata = load_metadata_from_corpus(corpus_path)
    else:
        print(f"Loading metadata from: {config.INDEX_DIR / config.METADATA_FILE}")
        metadata = load_metadata_from_index()

    print(f"  {len(metadata):,} chunks loaded")

    # Build graph
    print("\nBuilding graph (adjacency + intra + cross-reference edges)...")
    t0 = time.time()
    graph = build_graph_from_metadata(metadata)
    elapsed = time.time() - t0
    print(f"  Graph built in {elapsed:.1f}s")

    # Stats
    stats = graph_stats(graph)
    print(f"\nGraph statistics:")
    print(f"  Nodes with edges : {stats['total_nodes']:,}")
    print(f"  Total edges      : {stats['total_edges']:,}")
    print(f"  Edges by type:")
    for kind, count in sorted(stats["by_kind"].items()):
        print(f"    {kind:<8}: {count:,}")

    # Save
    output_path = config.INDEX_DIR / config.GRAPH_FILE
    save_graph(graph, output_path)
    size_mb = output_path.stat().st_size / 1e6
    print(f"\nGraph saved to: {output_path}  ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
