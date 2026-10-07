import argparse
import random
import sys
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from data.data_processor import DataProcessor
from retrieval.embedder import Embedder
from retrieval.retriever import Retriever


ADAPTER_DIR = config.BASE_DIR / "models" / "bge_reranker_ft"
HARD_NEG_PER_QUERY = 3
TRAIN_EPOCHS = 3
BATCH_SIZE = 4
RANDOM_SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune bge-reranker-v2-m3 on in-domain legal data")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build dataset, print stats, exit without training.",
    )
    p.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        metavar="FRAC",
        help="Fraction of pairs held out for AP evaluation (default: 0.1).",
    )
    return p.parse_args()


def build_pairs(
    corpus_chunks: list,
    qa_examples: list,
    retriever: "Retriever",
    hard_neg_per_query: int,
) -> list[tuple[str, str, float]]:
    relevant_map = DataProcessor.build_relevant_chunk_map(corpus_chunks, qa_examples)

    # chunk_id → text lookup
    chunk_text: dict[str, str] = {c.chunk_id: c.text for c in corpus_chunks}

    questions = [qa.question for qa in qa_examples]
    # Dense pool for hard negatives — need more candidates than top_k
    all_dense = retriever.batch_retrieve(questions, top_k=20)

    pairs: list[tuple[str, str, float]] = []
    skipped_noisy = 0

    for qa, dense_chunks in zip(qa_examples, all_dense):
        relevant_ids = set(relevant_map.get(qa.query_id, []))

        # Guard against source-level fallbacks that flood positives
        if len(relevant_ids) > 15:
            skipped_noisy += 1
            continue

        # Positives
        for cid in relevant_ids:
            text = chunk_text.get(cid, "")
            if text:
                pairs.append((qa.question, text, 1.0))

        # Hard negatives from dense pool that are not in the relevant set
        negatives = [
            c for c in dense_chunks
            if c["chunk_id"] not in relevant_ids
        ][:hard_neg_per_query]
        for chunk in negatives:
            pairs.append((qa.question, chunk["text"], 0.0))

    print(f"  Skipped noisy queries (relevant_ids > 15): {skipped_noisy}")
    return pairs


def compute_ap(model, pairs: list[tuple[str, str, float]]) -> float:
    """Average Precision over held-out pairs grouped by query."""
    import numpy as np

    # Group by query
    by_query: dict[str, list[tuple[str, float]]] = {}
    for q, passage, label in pairs:
        by_query.setdefault(q, []).append((passage, label))

    ap_scores = []
    for query, items in by_query.items():
        passages = [p for p, _ in items]
        labels = [l for _, l in items]
        if not any(labels):
            continue
        scores = model.predict([(query, p) for p in passages])
        order = np.argsort(scores)[::-1]
        hits = 0
        prec_sum = 0.0
        for rank, idx in enumerate(order, start=1):
            if labels[idx] == 1.0:
                hits += 1
                prec_sum += hits / rank
        ap_scores.append(prec_sum / sum(labels))

    return float(np.mean(ap_scores)) if ap_scores else 0.0


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)

    print(f"Loading corpus from {config.RAW_DATA_PATH} ...")
    processor = DataProcessor(config.RAW_DATA_PATH)
    processor.load_and_validate()
    corpus_chunks = list(processor.build_corpus_chunks())
    print(f"  Corpus chunks: {len(corpus_chunks)}")

    # Combine Kaggle 300 eval + HMGS gold as annotation source
    kaggle_examples = processor.build_qa_eval_set()
    try:
        hmgs_examples = DataProcessor.build_gold_eval_set()
    except Exception as e:
        print(f"  HMGS load failed ({e}), using Kaggle only.")
        hmgs_examples = []

    qa_examples = kaggle_examples + hmgs_examples
    print(f"  QA examples  : {len(qa_examples)} "
          f"(kaggle={len(kaggle_examples)}, hmgs={len(hmgs_examples)})")

    index_path = config.INDEX_DIR / config.INDEX_FILE
    metadata_path = config.INDEX_DIR / config.METADATA_FILE
    print(f"\nLoading FAISS index ...")
    embedder = Embedder()
    embedder.load_model()
    retriever = Retriever(embedder)
    retriever.load_index(index_path, metadata_path)
    print(f"  Index: {retriever.index.ntotal} vectors")

    print(f"\nBuilding training pairs (hard_neg_per_query={HARD_NEG_PER_QUERY}) ...")
    pairs = build_pairs(corpus_chunks, kaggle_examples, retriever, HARD_NEG_PER_QUERY)

    pos = sum(1 for _, _, l in pairs if l == 1.0)
    neg = sum(1 for _, _, l in pairs if l == 0.0)
    ratio = neg / pos if pos else float("inf")
    print(f"  Total pairs  : {len(pairs)}")
    print(f"  Positives    : {pos}")
    print(f"  Negatives    : {neg}")
    print(f"  Neg/Pos ratio: {ratio:.2f}")

    if args.dry_run:
        print("\n[dry-run] Exiting without training.")
        return

    # Train/eval split at the query level to prevent the same query from
    # appearing in both train and eval sets.
    n_eval = max(1, int(len(pairs) * args.eval_split))
    all_queries = list({qa.question for qa in kaggle_examples})
    random.shuffle(all_queries)
    n_eval_q = max(20, len(all_queries) // 10)
    eval_query_set = set(all_queries[:n_eval_q])
    train_qa = [qa for qa in kaggle_examples if qa.question not in eval_query_set]
    eval_qa  = [qa for qa in kaggle_examples if qa.question in eval_query_set]

    train_pairs = build_pairs(corpus_chunks, train_qa, retriever, HARD_NEG_PER_QUERY)
    eval_pairs  = build_pairs(corpus_chunks, eval_qa,  retriever, HARD_NEG_PER_QUERY)
    random.shuffle(train_pairs)
    random.shuffle(eval_pairs)
    eval_pairs = eval_pairs[:n_eval]
    print(f"\n  Train pairs  : {len(train_pairs)}")
    print(f"  Eval pairs   : {len(eval_pairs)}")

    import torch
    from sentence_transformers import CrossEncoder, InputExample
    from torch.utils.data import DataLoader

    train_examples = [InputExample(texts=[q, p], label=l) for q, p, l in train_pairs]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading CrossEncoder {config.RERANKER_MODEL} on {device} ...")
    model = CrossEncoder(config.RERANKER_MODEL, num_labels=1, device=device)

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)

    warmup_steps = int(len(train_dataloader) * TRAIN_EPOCHS * 0.1)

    print(f"\nFine-tuning for {TRAIN_EPOCHS} epochs ...")
    model.fit(
        train_dataloader=train_dataloader,
        epochs=TRAIN_EPOCHS,
        warmup_steps=warmup_steps,
        output_path=str(ADAPTER_DIR),
        show_progress_bar=True,
    )

    print(f"\nSaved fine-tuned reranker to {ADAPTER_DIR}")

    if eval_pairs:
        print("\nEvaluating on held-out split ...")
        ap = compute_ap(model, eval_pairs)
        print(f"  Average Precision (AP): {ap:.4f}")


if __name__ == "__main__":
    main()
