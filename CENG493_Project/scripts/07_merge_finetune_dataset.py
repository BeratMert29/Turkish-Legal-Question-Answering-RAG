"""Merge ipproo/Turkish-law HuggingFace dataset with qa_train.jsonl."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from utils import resolve_processed_file

IPPROO_SOURCE = "ipproo/Turkish-law"


def _normalize_q(text: str) -> str:
    """Canonical question key: delegate to data.data_processor.normalize_question
    (Turkish-aware: İ→i, I→ı before lowercase, then collapse non-word chars).
    Import is deferred so utils/ is on sys.path when called."""
    from data.data_processor import normalize_question
    return normalize_question(text)


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
        seen_questions.add(_normalize_q(rec["question"]))
    return records, seen_questions


def main(argv=None) -> None:
    import argparse
    from datasets import load_dataset  # local import — heavy dependency
    from data.data_processor import DataProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-missing-eval",
        action="store_true",
        help=(
            "Bypass the leakage check when no eval files exist.  "
            "Records leakage_checked=False in merge_config.json."
        ),
    )
    args = parser.parse_args(argv)

    # Resolve file paths via the canonical helper.  required=True raises
    # FileNotFoundError immediately if the input is absent — no silent fallback
    # to any other directory.  Output files use required=False (don't exist yet).
    qa_train_file = resolve_processed_file("qa_train.jsonl", required=True)
    output_file = resolve_processed_file("qa_train_merged.jsonl", required=False)
    merge_config_file = resolve_processed_file("merge_config.json", required=False)

    print(f"Loading existing training data from {qa_train_file} ...")
    existing_records, seen_questions = load_existing(qa_train_file)
    original_count = len(existing_records)
    print(f"  Loaded {original_count} existing examples.")

    # --- Eval-set deduplication -------------------------------------------
    # Exclude any train record whose (normalised) question also appears in an
    # eval set.  eval_keys comes from DataProcessor.saved_eval_question_keys()
    # which covers qa_eval, qa_hmgs, qa_turkish_legal_rag (and label-conflict
    # rows).  An empty result means the eval files haven't been built yet; that
    # would make this check a silent no-op — so we hard-fail unless the caller
    # explicitly passes --allow-missing-eval.
    print("\nLoading eval-set question keys for leakage check ...")

    # Track which expected eval files are present (reported in metadata).
    # TLR eval file lives in config.TLR_PROCESSED_DIR (a separate committed
    # directory), read explicitly from config — no fallback logic.
    _expected_eval_files: dict[str, Path] = {
        "qa_eval": config.PROCESSED_DIR / config.QA_GOLD_FILE,
        "qa_hmgs": config.PROCESSED_DIR / config.HMGS_GOLD_FILE,
        "qa_turkish_legal_rag": config.TLR_PROCESSED_DIR / config.TLR_GOLD_FILE,
    }
    found_eval_files = {k: str(v) for k, v in _expected_eval_files.items() if v.exists()}
    missing_eval_files = [k for k, v in _expected_eval_files.items() if not v.exists()]

    eval_keys = DataProcessor.saved_eval_question_keys()
    eval_keys_count = len(eval_keys)
    print(f"  {eval_keys_count} unique eval questions loaded.")

    leakage_checked: bool
    leakage_skip_reason: str | None = None

    if not eval_keys:
        if args.allow_missing_eval:
            leakage_skip_reason = (
                f"--allow-missing-eval: eval_keys is empty; "
                f"missing files: {missing_eval_files}"
            )
            print(f"  WARNING: {leakage_skip_reason}")
            leakage_checked = False
        else:
            missing_str = (
                ", ".join(missing_eval_files) if missing_eval_files
                else "(all expected files present but returned no keys)"
            )
            raise SystemExit(
                f"\nERROR: Leakage check cannot proceed — no eval questions were loaded.\n"
                f"  Missing eval files: {missing_str}\n"
                f"  Run the appropriate upstream scripts to build eval sets first,\n"
                f"  or pass --allow-missing-eval to bypass (leakage_checked will be False).\n"
                f"  Expected locations:\n"
                + "".join(f"    {k}: {v}\n" for k, v in _expected_eval_files.items())
            )
    else:
        leakage_checked = True

    clean_existing: list[dict] = []
    leaked_existing = 0
    for rec in existing_records:
        key = _normalize_q(rec["question"])
        if key in eval_keys:
            leaked_existing += 1
            # Record excluded from training entirely.  The eval_keys check in
            # the ipproo loop below also prevents re-adding this question from
            # the HuggingFace dataset.
            print(f"  [LEAKAGE] Removed from existing: {rec.get('question','')[:100]!r}")
        else:
            clean_existing.append(rec)

    print(f"  Removed {leaked_existing} existing record(s) that matched eval questions.")

    print(f"\nDownloading {IPPROO_SOURCE} from HuggingFace ...")
    ds = load_dataset(IPPROO_SOURCE)

    all_ipproo_rows: list[dict] = []
    for split_name, split_ds in ds.items():
        print(f"  Split '{split_name}': {len(split_ds)} rows")
        all_ipproo_rows.extend(split_ds)

    print(f"  Total ipproo rows: {len(all_ipproo_rows)}")

    added = 0
    duplicates = 0
    eval_skipped_ipproo = 0
    merged = list(clean_existing)

    for idx, row in enumerate(all_ipproo_rows, start=1):
        mapped = _map_ipproo_row(row, idx)
        if not mapped["question"] or not mapped["answer"]:
            continue
        key = _normalize_q(mapped["question"])
        if key in eval_keys:
            eval_skipped_ipproo += 1
            continue
        if key in seen_questions:
            duplicates += 1
            continue
        seen_questions.add(key)
        merged.append(mapped)
        added += 1

    final_count = len(merged)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in merged:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    merge_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_datasets": [str(qa_train_file), IPPROO_SOURCE],
        "leakage_check": {
            "leakage_checked": leakage_checked,
            "eval_keys_loaded": eval_keys_count if leakage_checked else 0,
            "eval_files_found": found_eval_files,
            "eval_files_missing": missing_eval_files,
            "skip_reason": leakage_skip_reason,
        },
        "counts": {
            "original_qa_train": original_count,
            "existing_leaked_removed": leaked_existing,
            "ipproo_rows_fetched": len(all_ipproo_rows),
            "ipproo_eval_skipped": eval_skipped_ipproo,
            "ipproo_added": added,
            "duplicates_removed": duplicates,
            "final_merged": final_count,
        },
        "output_file": str(output_file),
        "dedup_method": "normalize_question_turkish_aware",
    }
    with open(merge_config_file, "w", encoding="utf-8") as f:
        json.dump(merge_config, f, ensure_ascii=False, indent=2)

    print("\n--- Merge Stats ---")
    print(f"  Original qa_train examples       : {original_count}")
    print(f"  Existing records leaked (removed): {leaked_existing}")
    print(f"  ipproo rows fetched              : {len(all_ipproo_rows)}")
    print(f"  ipproo rows in eval (skipped)    : {eval_skipped_ipproo}")
    print(f"  ipproo rows added                : {added}")
    print(f"  Duplicates removed               : {duplicates}")
    print(f"  Final merged count               : {final_count}")
    print(f"\nSaved merged dataset to  : {output_file}")
    print(f"Saved merge config to    : {merge_config_file}")


if __name__ == "__main__":
    main()
