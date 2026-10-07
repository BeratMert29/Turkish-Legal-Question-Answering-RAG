"""Prepare processed data files from raw CSV."""
import time
import sys
from pathlib import Path
_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.append(_project_root)
import config
from data.data_processor import DataProcessor

def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dp = DataProcessor(config.RAW_DATA_PATH)
    dp.load_and_validate()

    # Build corpus chunks
    print("Building corpus chunks...")
    t0 = time.time()
    chunks = list(dp.build_corpus_chunks())
    corpus_path = config.PROCESSED_DIR / "corpus_chunks.jsonl"
    n_chunks = DataProcessor.save_jsonl(chunks, corpus_path)
    print(f"  → {n_chunks} chunks written to {corpus_path} ({time.time()-t0:.1f}s)")

    # Build QA eval set
    print("Building QA eval set...")
    t0 = time.time()
    eval_set = dp.build_qa_eval_set()
    eval_path = config.PROCESSED_DIR / "qa_eval.jsonl"
    DataProcessor.save_jsonl(eval_set, eval_path)
    print(f"  → {len(eval_set)} examples written to {eval_path} ({time.time()-t0:.1f}s)")

    # Build HMGS gold eval set
    if config.HMGS_DATA_PATH.exists():
        hmgs_set = dp.build_gold_eval_set()
        hmgs_path = config.PROCESSED_DIR / config.HMGS_GOLD_FILE
        DataProcessor.save_jsonl(hmgs_set, hmgs_path)
        print(f"  HMGS gold: {len(hmgs_set)} examples → {hmgs_path}")
    else:
        print(f"  SKIPPED HMGS: {config.HMGS_DATA_PATH} not found")

    # Build QA train set
    print("Building QA train set...")
    t0 = time.time()
    train_set = dp.build_qa_train_set()
    train_path = config.PROCESSED_DIR / "qa_train.jsonl"
    DataProcessor.save_jsonl(train_set, train_path)
    print(f"  → {len(train_set)} examples written to {train_path} ({time.time()-t0:.1f}s)")

    print("\n✓ Data preparation complete")

if __name__ == '__main__':
    main()
