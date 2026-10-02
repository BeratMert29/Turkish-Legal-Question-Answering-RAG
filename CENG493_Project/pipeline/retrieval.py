"""Shared retrieval helpers: dispatch, rerank, graph expansion, auto-build.

All heavy imports (torch, faiss, sentence_transformers) are deferred to
function bodies so this module loads on a CPU-only / light-deps machine.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from retrieval.bm25_retriever import BM25Index
    from retrieval.reranker import Reranker
    from retrieval.retriever import Retriever
    from retrieval.graph_index import GraphIndex


def auto_build_graph(graph_path: Path) -> None:
    """Build graph.json on the fly from the first available metadata file."""
    import config

    candidates = [
        config.INDEX_DIR / config.METADATA_FILE,
        config.BASE_DIR.parent / "results" / "index" / config.METADATA_FILE,
    ]
    for meta_path in candidates:
        if meta_path.exists():
            print(f"  Auto-building graph from {meta_path} …")
            import os as _os
            from retrieval.graph_builder import build_graph_from_metadata, save_graph

            with meta_path.open(encoding="utf-8") as _f:
                meta = [json.loads(line) for line in _f if line.strip()]
            _graph = build_graph_from_metadata(meta)
            _tmp_path = graph_path.with_suffix(".tmp")
            save_graph(_graph, _tmp_path)
            _os.replace(_tmp_path, graph_path)
            print(f"  Graph saved → {graph_path}")
            return


def retrieve(
    retriever: "Retriever",
    questions: list[str],
    *,
    retrieval_mode: str = "dense",
    bm25: "Optional[BM25Index]" = None,
    reranker: "Optional[Reranker]" = None,
    graph_index: "Optional[GraphIndex]" = None,
    use_rerank: bool = False,
    use_graph: bool = False,
    top_k: Optional[int] = None,
    reranker_candidates: Optional[int] = None,
    graph_hops: Optional[int] = None,
    graph_budget: Optional[int] = None,
) -> list[list[dict]]:
    """Unified retrieval dispatch: dense/hybrid/rrf -> rerank -> graph expand.

    Parameters
    ----------
    retriever : Retriever
        FAISS-backed dense retriever.
    questions : list[str]
        Batch of query strings.
    retrieval_mode : str
        One of ``"dense"``, ``"hybrid"``, ``"rrf"``.
    bm25 : BM25Index, optional
        Required when *retrieval_mode* is ``"hybrid"`` or ``"rrf"``.
    reranker : Reranker, optional
        Cross-encoder reranker; used when *use_rerank* is ``True``.
    graph_index : GraphIndex, optional
        Graph neighbor index; used when *use_graph* is ``True``.
    use_rerank, use_graph : bool
        Enable the corresponding post-retrieval step.
    top_k, reranker_candidates, graph_hops, graph_budget : int, optional
        Override the corresponding ``config.*`` default.

    Returns
    -------
    list[list[dict]]
        Per-query list of retrieved chunk dicts.
    """
    import config

    _top_k = top_k or config.TOP_K_RETRIEVAL
    _reranker_k = reranker_candidates or config.RERANKER_CANDIDATES
    initial_k = _reranker_k if use_rerank else _top_k

    t0 = time.time()

    if retrieval_mode == "rrf" and bm25 is not None:
        chunks = retriever.batch_rrf_retrieve(questions, bm25, top_k=initial_k)
    elif retrieval_mode == "hybrid" and bm25 is not None:
        chunks = retriever.batch_hybrid_retrieve(questions, bm25, top_k=initial_k)
    else:
        chunks = retriever.batch_retrieve(questions, top_k=initial_k)

    if use_rerank and reranker is not None:
        chunks = reranker.batch_rerank(questions, chunks, top_k=_top_k)

    if use_graph and graph_index is not None:
        _hops = graph_hops or config.GRAPH_HOPS
        _budget = graph_budget or config.GRAPH_NEIGHBOR_BUDGET
        chunks = graph_index.expand_batch(
            chunks,
            hops=_hops,
            budget=_budget,
            kinds=("adj",),  # adjacency edges only (safest)
            queries=questions,
        )

    print(f"    Retrieval done in {time.time() - t0:.1f}s")
    return chunks
