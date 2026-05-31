"""Generate a report-friendly Stage 1 markdown summary from baseline metrics."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config


def pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def fixed(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def ascii_bar(value: float | None, width: int = 24) -> str:
    if value is None:
        return "N/A"
    filled = max(0, min(width, round(value * width)))
    return f"{'#' * filled}{'-' * (width - filled)} {value * 100:.1f}%"


def read_gpu_snapshot() -> dict[str, str] | None:
    if not shutil.which("nvidia-smi"):
        return None

    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except Exception:
        return None

    line = result.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 5:
        return None

    return {
        "name": parts[0],
        "memory_total_mib": parts[1],
        "memory_used_mib": parts[2],
        "utilization_gpu_pct": parts[3],
        "temperature_c": parts[4],
    }


def build_report(metrics: dict, gpu: dict[str, str] | None) -> str:
    hp = metrics.get("hyperparameters", {})
    retrieval = metrics.get("retrieval_metrics", {})
    qa = metrics.get("qa_metrics", {})
    hall = metrics.get("hallucination_summary", {})
    by_category = hall.get("by_category", {})
    hits = by_category.get("hits", {})
    partial = by_category.get("partial", {})
    misses = by_category.get("misses", {})

    num_queries = retrieval.get("num_queries")
    retrieval_time_s = retrieval.get("retrieval_time_s")
    per_query_ms = None
    if num_queries and retrieval_time_s is not None:
        per_query_ms = (retrieval_time_s / num_queries) * 1000

    gpu_lines = []
    if gpu is None:
        gpu_lines.append("- Current GPU snapshot: unavailable (`nvidia-smi` missing or failed)")
    else:
        gpu_lines.extend(
            [
                f"- Current GPU snapshot: {gpu['name']}",
                f"- Current VRAM usage: {gpu['memory_used_mib']} / {gpu['memory_total_mib']} MiB",
                f"- Current GPU utilization: {gpu['utilization_gpu_pct']}%",
                f"- Current GPU temperature: {gpu['temperature_c']} C",
            ]
        )

    return f"""# Step 1 Progress Report

Source metrics: [baseline_metrics.json](./baseline_metrics.json)

## Experiment Setup

- Dataset: HMGS gold evaluation set
- Number of evaluation questions: {num_queries}
- Retrieval mode: `{hp.get("retrieval_mode", "N/A")}`
- Embedding model: `{hp.get("embedding_model", "N/A")}`
- LLM model: `{hp.get("llm_model", "N/A")}`
- Device used by the run: `{hp.get("device", "N/A")}`
- Chunk size / overlap: `{hp.get("chunk_size", "N/A")}` / `{hp.get("chunk_overlap", "N/A")}`

## Retrieval Results

| Metric | Value |
| --- | ---: |
| Recall@5 | {pct(retrieval.get("recall_at_5"))} |
| Recall@10 | {pct(retrieval.get("recall_at_10"))} |
| MRR | {fixed(retrieval.get("mrr"))} |
| nDCG@10 | {fixed(retrieval.get("ndcg_at_10"))} |
| Retrieval time | {fixed(retrieval_time_s, 2)} s |
| Avg. retrieval time / query | {fixed(per_query_ms, 2)} ms |

### Retrieval Bars

```text
Recall@5   {ascii_bar(retrieval.get("recall_at_5"))}
Recall@10  {ascii_bar(retrieval.get("recall_at_10"))}
MRR        {ascii_bar(retrieval.get("mrr"))}
nDCG@10    {ascii_bar(retrieval.get("ndcg_at_10"))}
```

## QA Results

| Metric | Value |
| --- | ---: |
| Exact Match | {pct(qa.get("em"))} |
| F1 | {pct(qa.get("f1"))} |
| ROUGE-L | {pct(qa.get("rouge_l"))} |
| BLEU | {pct(qa.get("bleu"))} |
| Citation Accuracy | {pct(qa.get("citation_accuracy"))} |
| Samples | {qa.get("num_samples", "N/A")} |

### QA Bars

```text
EM         {ascii_bar(qa.get("em"))}
F1         {ascii_bar(qa.get("f1"))}
ROUGE-L    {ascii_bar(qa.get("rouge_l"))}
BLEU       {ascii_bar(qa.get("bleu"))}
Citation   {ascii_bar(qa.get("citation_accuracy"))}
```

## Faithfulness

| Metric | Value |
| --- | ---: |
| Context-grounded answers | {hall.get("context_grounding_count", "N/A")} / {hall.get("total", "N/A")} |
| Context-grounding rate | {pct(hall.get("context_grounding_rate"))} |

### By Retrieval Category

| Category | Context-Grounded | Total | Rate |
| --- | ---: | ---: | ---: |
| Hits | {hits.get("context_grounded", "N/A")} | {hits.get("total", "N/A")} | {pct((hits.get("context_grounded", 0) / hits.get("total", 1)) if hits.get("total") else None)} |
| Partial | {partial.get("context_grounded", "N/A")} | {partial.get("total", "N/A")} | {pct((partial.get("context_grounded", 0) / partial.get("total", 1)) if partial.get("total") else None)} |
| Misses | {misses.get("context_grounded", "N/A")} | {misses.get("total", "N/A")} | {pct((misses.get("context_grounded", 0) / misses.get("total", 1)) if misses.get("total") else None)} |

```text
Context-grounding  {ascii_bar(hall.get("context_grounding_rate"))}
```

## Hardware Snapshot

{chr(10).join(gpu_lines)}
- Historical GPU utilization during the original run: not logged in `baseline_metrics.json`
- Index build time during the original run: {hp.get("index_build_time_s") if hp.get("index_build_time_s") is not None else "not recorded"}

## Reporting Notes

- This file summarizes the saved Step 1 baseline artifact already present in the repo.
- Retrieval quality is mixed: `MRR` and `nDCG@10` are strong, while `Recall@5/10` is low under the project's strict chunk-level relevance definition.
- Answer quality is still limited (`EM` and `F1` are low), but citation accuracy ({pct(qa.get("citation_accuracy"))}) and context-grounding rate ({pct(hall.get("context_grounding_rate"))}) are strong enough to report that grounding behavior is working.
- For future runs, log `nvidia-smi` samples during evaluation if you need report-grade GPU utilization curves.
"""


def main() -> None:
    metrics_path = config.RESULTS_DIR / "baseline_metrics.json"
    output_path = config.RESULTS_DIR / "step1_progress_report.md"

    with metrics_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    report = build_report(metrics, read_gpu_snapshot())
    output_path.write_text(report, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
