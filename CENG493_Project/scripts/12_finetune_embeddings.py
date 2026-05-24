#!/usr/bin/env python3
"""scripts/12_finetune_embeddings.py -- Fine-tune BGE-M3 on Turkish legal triplets.

Uses sentence-transformers SentenceTransformerTrainer with MultipleNegativesRankingLoss
(InfoNCE-style contrastive loss). Hard negatives from the triplet file are passed as
additional dataset columns (negative_0 ... negative_6) so the loss can treat them as
in-batch hard negatives in addition to the other anchors in the batch.

Prerequisites:
  - Run scripts/11_build_embedding_triplets.py first to generate the triplet file.
  - pip install "sentence-transformers>=3.0" datasets torch

Output:
  models/bge-m3-turkish-legal/   -- saved SentenceTransformer (safe-tensors + tokenizer)
  models/bge-m3-turkish-legal/training_config.json  -- hyperparameters for reproducibility
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

# ── Paths & hyperparameters ──────────────────────────────────────────────────
MODEL_NAME = "BAAI/bge-m3"
OUTPUT_DIR = config.BASE_DIR / "models" / "bge-m3-turkish-legal"
TRIPLET_FILE = config.PROCESSED_DIR / "embedding_triplets.jsonl"

TRAINING_CONFIG = {
    "model_name": MODEL_NAME,
    "learning_rate": 1e-5,
    "epochs": 2,
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.1,
    "fp16": True,
    "gradient_checkpointing": True,
    "eval_split": 0.05,
    "loss": "MultipleNegativesRankingLoss",
    "notes": (
        "anchor+positive+ALL hard negatives; up to 7 hard negatives per triplet are passed as "
        "negative, negative_1, ..., negative_6 columns. MNRL uses these plus in-batch negatives."
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    # Lazy imports so the script fails fast if deps are missing
    try:
        from sentence_transformers import (
            SentenceTransformer,
            SentenceTransformerTrainer,
            SentenceTransformerTrainingArguments,
            losses,
        )
        from datasets import Dataset
    except ImportError as e:
        print(f"ERROR: missing dependency -- {e}")
        print("Install with:  pip install 'sentence-transformers>=3.0' datasets")
        sys.exit(1)

    # ── Load triplets ────────────────────────────────────────────────────────
    if not TRIPLET_FILE.exists():
        print(f"ERROR: triplet file not found at {TRIPLET_FILE}")
        print("Run scripts/11_build_embedding_triplets.py first.")
        sys.exit(1)

    print(f"Loading triplets from {TRIPLET_FILE} ...")
    raw = load_jsonl(TRIPLET_FILE)
    print(f"  {len(raw):,} triplets loaded.")

    # ── Build HuggingFace Dataset ────────────────────────────────────────────
    # Build triplets: anchor + positive + ALL hard negatives.
    # sentence-transformers v3+ MultipleNegativesRankingLoss supports multiple
    # negative columns: "negative", "negative_1", "negative_2", ...
    records = []
    for t in raw:
        record = {
            "anchor": t["query"],
            "positive": t["pos"][0],
        }
        negs = t.get("neg", [])
        for neg_i, neg_text in enumerate(negs):
            col_name = "negative" if neg_i == 0 else f"negative_{neg_i}"
            record[col_name] = neg_text
        records.append(record)

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(
        test_size=TRAINING_CONFIG["eval_split"], seed=42
    )
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"  Train: {len(train_ds):,}  |  Eval: {len(eval_ds):,}")

    # ── Load model ───────────────────────────────────────────────────────────
    print(f"\nLoading {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # ── Loss ─────────────────────────────────────────────────────────────────
    # MultipleNegativesRankingLoss uses in-batch negatives (InfoNCE).
    # The extra negative_* columns are treated as additional negatives for each anchor.
    loss = losses.MultipleNegativesRankingLoss(model)

    # ── Training arguments ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=TRAINING_CONFIG["epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        warmup_ratio=TRAINING_CONFIG["warmup_ratio"],
        fp16=TRAINING_CONFIG["fp16"],
        gradient_checkpointing=TRAINING_CONFIG["gradient_checkpointing"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        loss=loss,
    )

    print("\nStarting training ...")
    print(f"  Model      : {MODEL_NAME}")
    print(f"  Epochs     : {TRAINING_CONFIG['epochs']}")
    print(f"  Batch size : {TRAINING_CONFIG['batch_size']}")
    print(f"  LR         : {TRAINING_CONFIG['learning_rate']}")
    print(f"  Output dir : {OUTPUT_DIR}")

    trainer.train()

    # ── Save ─────────────────────────────────────────────────────────────────
    print(f"\nSaving fine-tuned model to {OUTPUT_DIR} ...")
    model.save(str(OUTPUT_DIR))

    config_path = OUTPUT_DIR / "training_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(TRAINING_CONFIG, f, indent=2, ensure_ascii=False)

    print(f"  Model saved          : {OUTPUT_DIR}")
    print(f"  Training config saved: {config_path}")
    print("\nDone.")
    print("\nNext steps:")
    print("  1. Update config.py: set EMBEDDING_MODEL = FINETUNED_EMBEDDING_MODEL")
    print("  2. Rebuild FAISS index: python scripts/02_build_index.py")
    print("  3. Evaluate: python scripts/03_evaluate_retrieval.py")


if __name__ == "__main__":
    main()
