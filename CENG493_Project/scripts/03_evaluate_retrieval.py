"""Evaluate retrieval quality using context-hash ground truth + BM25 hybrid."""
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

import config
from data.data_processor import DataProcessor
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever
from retrieval.bm25_retriever import BM25Index
from retrieval.reranker import Reranker
from evaluation.retrieval_metrics import compute_all_metrics


def main():
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    index_path    = config.INDEX_DIR / config.INDEX_FILE
    metadata_path = config.INDEX_DIR / config.METADATA_FILE
    print(f"Loading index from {index_path}")
    embedder = Embedder()
    embedder.load_model()
    retriever = Retriever(embedder)
    retriever.load_index(index_path, metadata_path)
    n_corpus = retriever.index.ntotal
    print(f"  Index loaded: {n_corpus} vectors")

    print(f"\nLoading data from {config.RAW_DATA_PATH}")
    processor = DataProcessor(config.RAW_DATA_PATH)
    processor.load_and_validate()
    corpus_chunks = list(processor.build_corpus_chunks())
    qa_examples   = processor.build_qa_eval_set()
    print(f"  Corpus chunks: {len(corpus_chunks)}")
    print(f"  QA examples:   {len(qa_examples)}")

    print("\nBuilding ground-truth relevance map (context-hash)...")
    relevant_map = DataProcessor.build_relevant_chunk_map(corpus_chunks, qa_examples)
    matched = sum(1 for v in relevant_map.values() if v)
    print(f"  Queries with at least one relevant chunk: {matched}/{len(qa_examples)}")

    questions = [qa.question for qa in qa_examples]

    print(f"\nDense retrieval for {len(qa_examples)} queries...")
    t0 = time.time()
    all_dense = retriever.batch_retrieve(questions, top_k=config.TOP_K_RETRIEVAL)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nBuilding BM25 index over {len(corpus_chunks)} chunks...")
    t0 = time.time()
    bm25_index = BM25Index()
    bm25_index.build([{"text": c.text, "chunk_id": c.chunk_id} for c in corpus_chunks])
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nHybrid retrieval for {len(qa_examples)} queries (alpha=0.7)...")
    t0 = time.time()
    all_hybrid = retriever.batch_hybrid_retrieve(
        questions, bm25_index, alpha=0.7, top_k=config.TOP_K_RETRIEVAL
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nRRF retrieval for {len(qa_examples)} queries...")
    t0 = time.time()
    all_rrf = retriever.batch_rrf_retrieve(questions, bm25_index, top_k=config.TOP_K_RETRIEVAL)
    print(f"  Done in {time.time()-t0:.1f}s")

    print("\nLoading cross-encoder reranker...")
    reranker = Reranker()
    reranker.load_model()

    print(f"\nDense+Rerank retrieval for {len(qa_examples)} queries...")
    t0 = time.time()
    dense_cands = retriever.batch_retrieve(questions, top_k=config.RERANKER_CANDIDATES)
    all_dense_rerank = reranker.batch_rerank(questions, dense_cands, top_k=config.TOP_K_RETRIEVAL)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nHybrid+Rerank retrieval for {len(qa_examples)} queries...")
    t0 = time.time()
    hybrid_cands = retriever.batch_hybrid_retrieve(questions, bm25_index, top_k=config.RERANKER_CANDIDATES)
    all_hybrid_rerank = reranker.batch_rerank(questions, hybrid_cands, top_k=config.TOP_K_RETRIEVAL)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nRRF+Rerank retrieval for {len(qa_examples)} queries...")
    t0 = time.time()
    rrf_cands = retriever.batch_rrf_retrieve(questions, bm25_index, top_k=config.RERANKER_CANDIDATES)
    all_rrf_rerank = reranker.batch_rerank(questions, rrf_cands, top_k=config.TOP_K_RETRIEVAL)
    print(f"  Done in {time.time()-t0:.1f}s")

    def build_results(all_retrieved):
        metric_results = []
        full_results   = {}
        for qa, chunks in zip(qa_examples, all_retrieved):
            retrieved_chunk_ids = []
            seen: set[str] = set()
            for c in chunks:
                cid = c["chunk_id"]
                if cid not in seen:
                    seen.add(cid)
                    retrieved_chunk_ids.append(cid)
            metric_results.append({
                "query_id":  qa.query_id,
                "relevant":  relevant_map.get(qa.query_id, []),
                "retrieved": retrieved_chunk_ids,
            })
            full_results[qa.query_id] = chunks
        return metric_results, full_results

    dense_results,  dense_full  = build_results(all_dense)
    hybrid_results, hybrid_full = build_results(all_hybrid)

    dense_metrics  = compute_all_metrics(dense_results)
    hybrid_metrics = compute_all_metrics(hybrid_results)

    rrf_results,           rrf_full           = build_results(all_rrf)
    dense_rerank_results,  dense_rerank_full  = build_results(all_dense_rerank)
    hybrid_rerank_results, hybrid_rerank_full = build_results(all_hybrid_rerank)
    rrf_rerank_results,    rrf_rerank_full    = build_results(all_rrf_rerank)

    rrf_metrics           = compute_all_metrics(rrf_results)
    dense_rerank_metrics  = compute_all_metrics(dense_rerank_results)
    hybrid_rerank_metrics = compute_all_metrics(hybrid_rerank_results)
    rrf_rerank_metrics    = compute_all_metrics(rrf_rerank_results)

    all_modes = {
        "dense":          dense_metrics,
        "hybrid":         hybrid_metrics,
        "rrf":            rrf_metrics,
        "dense_rerank":   dense_rerank_metrics,
        "hybrid_rerank":  hybrid_rerank_metrics,
        "rrf_rerank":     rrf_rerank_metrics,
    }

    print(f"\n{'Mode':20s}  {'R@5':>8s}  {'R@10':>8s}  {'MRR':>8s}  {'nDCG@10':>8s}")
    print("-" * 66)
    for mode_name, m in all_modes.items():
        print(f"  {mode_name:18s}  {m.get('recall_at_5',0):8.4f}  "
              f"{m.get('recall_at_10',0):8.4f}  {m.get('mrr',0):8.4f}  "
              f"{m.get('ndcg_at_10',0):8.4f}")

    out = {
        "dense_metrics":          dense_metrics,
        "hybrid_metrics":         hybrid_metrics,
        "rrf_metrics":            rrf_metrics,
        "dense_rerank_metrics":   dense_rerank_metrics,
        "hybrid_rerank_metrics":  hybrid_rerank_metrics,
        "rrf_rerank_metrics":     rrf_rerank_metrics,
        "per_query_results": {
            "dense":          dense_results,
            "hybrid":         hybrid_results,
            "rrf":            rrf_results,
            "dense_rerank":   dense_rerank_results,
            "hybrid_rerank":  hybrid_rerank_results,
            "rrf_rerank":     rrf_rerank_results,
        },
    }
    results_path = config.RESULTS_DIR / "retrieval_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {results_path}")

    print("\nRetrieval evaluation complete")


if __name__ == '__main__':
    main()
