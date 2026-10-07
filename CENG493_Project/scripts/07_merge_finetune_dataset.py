"""Merge ipproo/Turkish-law HuggingFace dataset with qa_train.jsonl."""
import json
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

IPPROO_SOURCE = "ipproo/Turkish-law"
OUTPUT_FILE = config.PROCESSED_DIR / "qa_train_merged.jsonl"
MERGE_CONFIG_FILE = config.PROCESSED_DIR / "merge_config.json"
QA_TRAIN_FILE = config.PROCESSED_DIR / "qa_train.jsonl"


def _normalize(text: str) -> str:
    """Lowercase + strip + collapse whitespace + NFC for dedup."""
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.lower().split())


def _map_ipproo_row(row: dict, idx: int) -> dict:
    """
    Map a raw ipproo/Turkish-law row to the project schema.

    The dataset publishes one of two schemas — check both:
      Schema A (instruction-tuning style): instruction / input / output
      Schema B (QA style):                 question / answer

    'context' and 'source' are populated from 'input' when present;
    otherwise left empty to match the qa_train.jsonl convention.
    """
    if "instruction" in row and "output" in row:
        # Schema A
        question = (row.get("instruction") or "").strip()
        context = (row.get("input") or "").strip()
        answer = (row.get("output") or "").strip()
    elif "question" in row and "answer" in row:
        # Schema B
        question = (row.get("question") or "").strip()
        context = (row.get("context") or "").strip()
        answer = (row.get("answer") or "").strip()
    else:
        # Fallback: grab whatever text fields are available
        fields = list(row.keys())
        question = str(row.get(fields[0], "")).strip()
        answer = str(row.get(fields[-1], "")).strip()
        context = ""

    return {
        "query_id": f"ipproo_{idx}",
        "question": question,
        "answer": answer,
        "context": context,
        "source": IPPROO_SOURCE,
        "data_type": "train",
    }


def load_existing(path: Path) -> tuple[list[dict], set[str]]:
    from utils import read_jsonl
    records = []
    seen_questions: set[str] = set()
    for rec in read_jsonl(path):
        records.append(rec)
        seen_questions.add(_normalize(rec["question"]))
    return records, seen_questions


def main() -> None:
    from datasets import load_dataset  # local import — heavy dependency

    print(f"Loading existing training data from {QA_TRAIN_FILE} ...")
    existing_records, seen_questions = load_existing(QA_TRAIN_FILE)
    original_count = len(existing_records)
    print(f"  Loaded {original_count} existing examples.")

    print(f"\nDownloading {IPPROO_SOURCE} from HuggingFace ...")
    ds = load_dataset(IPPROO_SOURCE)

    all_ipproo_rows: list[dict] = []
    for split_name, split_ds in ds.items():
        print(f"  Split '{split_name}': {len(split_ds)} rows")
        all_ipproo_rows.extend(split_ds)

    print(f"  Total ipproo rows: {len(all_ipproo_rows)}")

    added = 0
    duplicates = 0
    merged = list(existing_records)

    for idx, row in enumerate(all_ipproo_rows, start=1):
        mapped = _map_ipproo_row(row, idx)
        if not mapped["question"] or not mapped["answer"]:
            continue
        key = _normalize(mapped["question"])
        if key in seen_questions:
            duplicates += 1
            continue
        seen_questions.add(key)
        merged.append(mapped)
        added += 1

    final_count = len(merged)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    merge_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_datasets": [str(QA_TRAIN_FILE), IPPROO_SOURCE],
        "counts": {
            "original_qa_train": original_count,
            "ipproo_rows_fetched": len(all_ipproo_rows),
            "ipproo_added": added,
            "duplicates_removed": duplicates,
            "final_merged": final_count,
        },
        "output_file": str(OUTPUT_FILE),
        "dedup_method": "normalized_question_text",
    }
    with open(MERGE_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(merge_config, f, ensure_ascii=False, indent=2)

    print("\n--- Merge Stats ---")
    print(f"  Original qa_train examples : {original_count}")
    print(f"  ipproo rows fetched        : {len(all_ipproo_rows)}")
    print(f"  ipproo rows added          : {added}")
    print(f"  Duplicates removed         : {duplicates}")
    print(f"  Final merged count         : {final_count}")
    print(f"\nSaved merged dataset to  : {OUTPUT_FILE}")
    print(f"Saved merge config to    : {MERGE_CONFIG_FILE}")


if __name__ == "__main__":
    main()
