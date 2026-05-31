#!/usr/bin/env python3
"""scripts/11_build_embedding_triplets.py -- Build contrastive training triplets for BGE-M3 fine-tuning.

For each question in qa_train.jsonl:
  - positive: top-1 corpus chunk by cosine similarity (skipped if score < 0.3)
  - hard negatives: ranks 5-30, filtered to exclude chunks from the same source document
  - padded with random negatives to reach NUM_HARD_NEGATIVES total

Output format (FlagEmbedding-compatible JSONL):
  {"query": "...", "pos": ["positive chunk text"], "neg": ["neg1", ..., "neg7"]}
"""
import json
import random
import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import config
from retrieval.embedder import Embedder

# ── Config ───────────────────────────────────────────────────────────────────
NUM_HARD_NEGATIVES = 7        # target negatives per query
HARD_NEG_TOP_K = 30           # search pool size for hard negative mining
MIN_POSITIVE_SCORE = 0.3      # skip query if best chunk similarity is below this
HARD_NEG_START_RANK = 4       # skip ranks 0-3 (too close to positive) — 0-indexed
MIN_GT_OVERLAP = 0.3          # minimum Jaccard overlap to accept a ground-truth positive
RANDOM_SEED = 42

QA_PATH = config.PROCESSED_DIR / "qa_train.jsonl"
CORPUS_PATH = config.PROCESSED_DIR / "corpus_chunks.jsonl"
OUTPUT_PATH = config.PROCESSED_DIR / "embedding_triplets.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _jaccard_tokens(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    tokens_a = set(a.lower().split())
    tokens_b = set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _find_gt_positive(context: str, chunk_texts: list[str]) -> tuple[int | None, float]:
    """Find corpus chunk with highest Jaccard overlap to ground-truth context."""
    best_idx = None
    best_score = 0.0
    for idx, text in enumerate(chunk_texts):
        score = _jaccard_tokens(context, text)
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


def main() -> None:
    random.seed(RANDOM_SEED)

    # ── Load data ────────────────────────────────────────────────────────────
    print(f"Loading QA pairs from {QA_PATH} ...")
    qa_pairs = load_jsonl(QA_PATH)
    print(f"  {len(qa_pairs)} QA pairs loaded.")

    print(f"Loading corpus chunks from {CORPUS_PATH} ...")
    corpus = load_jsonl(CORPUS_PATH)
    chunk_texts = [c["text"] for c in corpus]
    chunk_ids = [c["chunk_id"] for c in corpus]
    print(f"  {len(corpus)} corpus chunks loaded.")

    # ── Embed corpus ─────────────────────────────────────────────────────────
    print(f"\nLoading embedder ({config.EMBEDDING_MODEL}) ...")
    embedder = Embedder()
    embedder.load_model()

    print(f"Embedding {len(chunk_texts)} corpus chunks ...")
    corpus_embs = embedder.encode(chunk_texts, is_query=False)  # (N, 1024) float32

    # ── Build FAISS index ────────────────────────────────────────────────────
    import faiss

    index = faiss.IndexFlatIP(config.EMBEDDING_DIM)
    index.add(corpus_embs.astype(np.float32))
    print(f"  FAISS index built with {index.ntotal} vectors.")

    # ── Embed all questions in one batch ─────────────────────────────────────
    questions = [qa["question"] for qa in qa_pairs]
    print(f"\nEncoding {len(questions)} questions ...")
    q_embs = embedder.encode(questions, is_query=True)  # (M, 1024) float32

    # ── Batch search: retrieve top-HARD_NEG_TOP_K chunks for every question ──
    print(f"Running FAISS search (top_k={HARD_NEG_TOP_K}) ...")
    scores_all, indices_all = index.search(q_embs.astype(np.float32), HARD_NEG_TOP_K)
    print("  Search complete.")

    # ── Build triplets ───────────────────────────────────────────────────────
    all_indices = list(range(len(chunk_texts)))
    triplets: list[dict] = []
    skipped_low_score = 0

    for i, qa in enumerate(qa_pairs):
        # Try ground-truth context first (avoids self-distillation errors)
        gt_context = qa.get("context", "")
        pos_idx = None
        pos_text = None
        if gt_context:
            gt_idx, gt_overlap = _find_gt_positive(gt_context, chunk_texts)
            if gt_idx is not None and gt_overlap >= MIN_GT_OVERLAP:
                pos_idx = gt_idx
                pos_text = chunk_texts[pos_idx]

        # Fall back to top-1 FAISS result
        if pos_idx is None:
            pos_idx = int(indices_all[i][0])
            pos_score = float(scores_all[i][0])
            if pos_score < MIN_POSITIVE_SCORE:
                skipped_low_score += 1
                continue
            pos_text = chunk_texts[pos_idx]

        pos_source = corpus[pos_idx].get("source", "")

        # Hard negatives: ranks HARD_NEG_START_RANK to HARD_NEG_TOP_K-1
        hard_negs: list[str] = []
        for rank in range(HARD_NEG_START_RANK, HARD_NEG_TOP_K):
            neg_idx = int(indices_all[i][rank])

            # Skip if this is the positive chunk itself
            if neg_idx == pos_idx:
                continue

            # Skip if the chunk comes from the same source document (too easy / near-duplicate)
            neg_source = corpus[neg_idx].get("source", "")
            if neg_source and neg_source == pos_source:
                continue

            hard_negs.append(chunk_texts[neg_idx])
            if len(hard_negs) >= NUM_HARD_NEGATIVES:
                break

        # Pad with random negatives if the hard negative pool was exhausted
        # Dedup: don't repeat any already-selected negative or the positive
        selected_texts = set(hard_negs)
        selected_texts.add(pos_text)
        pad_attempts = 0
        while len(hard_negs) < NUM_HARD_NEGATIVES and pad_attempts < 1000:
            rand_idx = random.choice(all_indices)
            rand_text = chunk_texts[rand_idx]
            if rand_idx != pos_idx and rand_text not in selected_texts:
                hard_negs.append(rand_text)
                selected_texts.add(rand_text)
            pad_attempts += 1

        triplets.append({
            "query": qa["question"],
            "pos": [pos_text],
            "neg": hard_negs,
        })

    # ── Save output ───────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"\nDone.")
    print(f"  Triplets saved : {len(triplets):,}  ->  {OUTPUT_PATH}")
    print(f"  Skipped        : {skipped_low_score:,}  (positive score < {MIN_POSITIVE_SCORE})")
    print(f"  Negatives/query: {NUM_HARD_NEGATIVES} (hard) + random padding as needed")


if __name__ == "__main__":
    main()
