"""Custom benchmark Q&A set resolution."""
from __future__ import annotations
import json
import pathlib

import config
from data.data_processor import DataProcessor


def _load_qa_file(path: pathlib.Path) -> list[dict]:
    """Load a QA file that is a CSV, JSON array, or JSONL file.

    Detection strategy:
    1. If the file extension is .csv, parse as CSV using pandas.
    2. Otherwise, try to parse the entire file as a JSON array first.
    3. If that fails (or the result is not a list), fall back to line-by-line
       JSONL parsing.  This covers:
    - gold_benchmark.json  (JSON array)
    - rag_eval.json        (JSON array)
    - custom JSONL files   (one record per line)
    - CSV files with question/answer columns
    """
    if path.suffix.lower() == ".csv":
        import pandas as pd  # type: ignore[import-untyped]

        df = pd.read_csv(path)
        columns = set(df.columns.tolist())

        # Resolve question column
        question_col: str | None = None
        for candidate in ("question", "query"):
            if candidate in columns:
                question_col = candidate
                break
        if question_col is None:
            raise ValueError(
                f"CSV file {path} must contain a 'question' or 'query' column. "
                f"Found columns: {sorted(columns)}"
            )

        # Resolve answer column
        answer_col: str | None = None
        for candidate in ("answer", "expected_answer", "ground_truth"):
            if candidate in columns:
                answer_col = candidate
                break
        if answer_col is None:
            raise ValueError(
                f"CSV file {path} must contain an 'answer', 'expected_answer', "
                f"or 'ground_truth' column. Found columns: {sorted(columns)}"
            )

        # Resolve optional query_id column
        query_id_col: str | None = None
        for candidate in ("id", "question_id", "query_id"):
            if candidate in columns:
                query_id_col = candidate
                break

        # Resolve optional source column
        source_col: str | None = "source" if "source" in columns else None

        records: list[dict] = []
        for _, row in df.iterrows():
            record: dict = {
                "question": row[question_col],
                "answer": row[answer_col],
            }
            if query_id_col is not None:
                record["query_id"] = row[query_id_col]
            if source_col is not None:
                record["source"] = row[source_col]
            records.append(record)
        return records

    raw = path.read_text(encoding="utf-8").strip()
    if raw.startswith("["):
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass
    # Fall back to JSONL (one JSON object per line)
    return DataProcessor.load_jsonl(path)


def _normalize_qa_record(row: dict, index: int) -> dict:
    """Normalize a single QA record to the project's eval schema.

    Handles both the project's own format and the evaluator's formats
    (gold_benchmark.json and rag_eval.json).

    New field ``gold_source_ids`` carries the exact corpus chunk IDs from
    ``gold_sources`` / ``gold_chunk_ids`` for precise retrieval evaluation.
    """
    # --- question field ---
    question = row.get("question") or row.get("query", "")

    # --- answer field (evaluator uses 'verified_answer' or 'gold_answer_extract') ---
    answer = (
        row.get("answer")
        or row.get("verified_answer")
        or row.get("gold_answer_extract")
        or row.get("expected_response")   # TruLens
        or row.get("ground_truth")        # RAGAS
        or ""
    )

    # --- query_id ---
    query_id = (
        row.get("query_id")
        or row.get("question_id")
        or row.get("id")
        or row.get("_id")         # BEIR queries
        or f"custom_{index:04d}"
    )

    # --- gold_source_ids: list of exact chunk IDs for precision matching ---
    gold_source_ids: list[str] = []

    # gold_benchmark.json style: gold_sources[].source_id
    gold_sources = row.get("gold_sources")
    if gold_sources and isinstance(gold_sources, list):
        gold_source_ids = [
            gs["source_id"] for gs in gold_sources if gs.get("source_id")
        ]

    # rag_eval.json style: gold_chunk_ids[]
    elif row.get("gold_chunk_ids"):
        gold_source_ids = list(row["gold_chunk_ids"])

    # --- source: prefer first gold source id, then plain source string ---
    source = (
        gold_source_ids[0]
        if gold_source_ids
        else row.get("source", "")
    )

    return {
        "query_id": str(query_id),
        "question": question,
        "answer": answer,
        "context": row.get("context", ""),
        "source": source,
        "data_type": row.get("data_type", row.get("answer_type", "")),
        # Exact corpus chunk IDs; empty list for records without ground-truth citations.
        "gold_source_ids": gold_source_ids,
    }


def load_custom_qa(qa_file_path: str | pathlib.Path) -> list[dict]:
    """Load and normalize a custom QA benchmark file into list of dicts
    compatible with the eval pipeline's qa_predictions schema.

    Accepts JSON arrays (.json), JSONL files (.jsonl), and CSV files (.csv).
    For CSV files, the file must contain at minimum a question/query column
    and an answer/expected_answer/ground_truth column.  Optional columns
    include id/question_id/query_id (mapped to query_id) and source.
    Handles the project's native format as well as the evaluator's
    gold_benchmark.json and rag_eval.json formats, TruLens
    (expected_response), and RAGAS (ground_truth) formats.
    """
    p = pathlib.Path(qa_file_path)
    if not p.exists():
        raise FileNotFoundError(f"--qa-file not found: {p}")

    rows = _load_qa_file(p)
    normalized = []
    for i, row in enumerate(rows):
        rec = _normalize_qa_record(row, i)
        if not rec["question"]:
            raise ValueError(f"Row {i}: missing 'question'/'query' field in {p}")
        # For the project's own format 'answer' is required; for evaluator
        # benchmark files 'verified_answer'/'gold_answer_extract' fill in.
        if not rec["answer"]:
            raise ValueError(
                f"Row {i}: missing 'answer'/'verified_answer'/'gold_answer_extract'/"
                f"'expected_response'/'ground_truth' field in {p}"
            )
        normalized.append(rec)
    return normalized


def resolve_qa_set(
    dataset: str,
    qa_file: str | None,
) -> tuple[list[dict], bool, str]:
    """Return (eval_set_as_dicts, short_answer_mode, suffix).

    - qa_file provided or dataset=='custom' -> custom file (JSON array or JSONL)
    - dataset=='hmgs' -> HMGS eval set
    - dataset=='kaggle' -> Kaggle eval set
    """
    if qa_file or dataset == "custom":
        if not qa_file:
            raise ValueError("--dataset custom requires --qa-file <path>")
        rows = load_custom_qa(qa_file)
        return rows, False, "_custom"

    if dataset == "hmgs":
        eval_path = pathlib.Path(config.PROCESSED_DIR) / config.HMGS_GOLD_FILE
        rows = DataProcessor.load_jsonl(eval_path)
        return rows, True, "_hmgs"

    # kaggle (default)
    eval_path = pathlib.Path(config.PROCESSED_DIR) / config.QA_GOLD_FILE
    rows = DataProcessor.load_jsonl(eval_path)
    return rows, False, ""
