"""Custom benchmark Q&A set resolution."""
from __future__ import annotations
import json
import pathlib

import config
from data.data_processor import DataProcessor


def load_custom_qa(qa_file_path: str | pathlib.Path) -> list[dict]:
    """Load and normalize a custom JSONL benchmark file into list of dicts
    compatible with the eval pipeline's qa_predictions schema."""
    p = pathlib.Path(qa_file_path)
    if not p.exists():
        raise FileNotFoundError(f"--qa-file not found: {p}")

    rows = DataProcessor.load_jsonl(p)
    normalized = []
    for i, row in enumerate(rows):
        question = row.get("question", "")
        answer = row.get("answer", "")
        if not question:
            raise ValueError(f"Row {i}: missing 'question' field")
        if not answer:
            raise ValueError(f"Row {i}: missing 'answer' field")
        query_id = row.get("query_id") or row.get("id") or f"custom_{i:04d}"
        normalized.append({
            "query_id": str(query_id),
            "question": question,
            "answer": answer,
            "context": row.get("context", ""),
            "source": row.get("source", ""),
            "data_type": row.get("data_type", ""),
        })
    return normalized


def resolve_qa_set(
    dataset: str,
    qa_file: str | None,
) -> tuple[list[dict], bool, str]:
    """Return (eval_set_as_dicts, short_answer_mode, suffix).

    - qa_file provided or dataset=='custom' -> custom JSONL
    - dataset=='hmgs' -> HMGS eval set
    - dataset=='kaggle' -> Kaggle eval set
    """
    if qa_file or dataset == "custom":
        if not qa_file:
            raise ValueError("--dataset custom requires --qa-file <path>")
        rows = load_custom_qa(qa_file)
        # Convert to list of dicts matching QAExample fields
        return rows, False, "_custom"

    if dataset == "hmgs":
        eval_path = pathlib.Path(config.PROCESSED_DIR) / config.HMGS_GOLD_FILE
        rows = DataProcessor.load_jsonl(eval_path)
        return rows, True, "_hmgs"

    # kaggle (default)
    eval_path = pathlib.Path(config.PROCESSED_DIR) / config.QA_GOLD_FILE
    rows = DataProcessor.load_jsonl(eval_path)
    return rows, False, ""
