"""Shared data-loading helpers for external corpus and QA files.

All project-internal imports are deferred so the module loads with only
stdlib available.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_external_corpus(path: Path) -> list:
    """Load corpus from evaluator-format corpus.jsonl -> list[CorpusChunk]."""
    from data.corpus_loader import load_corpus_jsonl
    from data.data_processor import CorpusChunk

    raw_chunks = load_corpus_jsonl(path)
    return [
        CorpusChunk(
            **{k: r[k] for k in ("chunk_id", "doc_id", "text", "source", "char_len")}
        )
        for r in raw_chunks
    ]


def load_external_qa(path: Path) -> tuple[list, bool]:
    """Load QA examples from rag_eval.json or gold_benchmark.json.

    Auto-detects format from first item keys.
    Attaches ``gold_source_ids`` as a dynamic attribute so
    ``build_relevant_chunk_map`` Strategy 0 (exact chunk ID match) works.

    Returns
    -------
    tuple[list[QAExample], bool]
        ``(qa_examples, short_answer_mode)``
    """
    from data.data_processor import QAExample

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError(f"Empty QA file: {path}")

    # run_baseline accepted dicts; keep that tolerance
    if isinstance(data, dict):
        data = list(data.values())

    first = data[0]
    examples: list = []

    if "query_id" in first and "query" in first:
        # rag_eval.json format -- open-ended answers, no short-answer mode
        short_answer_mode = False
        for item in data:
            qa = QAExample(
                query_id=item["query_id"],
                question=item["query"],
                answer=item.get("gold_answer_extract", ""),
                context="",
                source=item.get("source", ""),
                data_type="",
            )
            qa.gold_source_ids = item.get("gold_chunk_ids", [])
            examples.append(qa)

    elif "question_id" in first and "question" in first:
        # gold_benchmark.json format -- exam-style, short answers
        short_answer_mode = True
        for item in data:
            gold_sources = item.get("gold_sources", [])
            qa = QAExample(
                query_id=item["question_id"],
                question=item["question"],
                answer=item.get("verified_answer", ""),
                context="",
                source=(
                    gold_sources[0].get("source", "") if gold_sources else ""
                ),
                data_type="",
            )
            qa.gold_source_ids = [s["source_id"] for s in gold_sources]
            examples.append(qa)

    else:
        raise ValueError(
            f"Unrecognised QA file format in {path}. "
            "Expected rag_eval.json (query_id+query) or "
            "gold_benchmark.json (question_id+question)."
        )

    return examples, short_answer_mode
