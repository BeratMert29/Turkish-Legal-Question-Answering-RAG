"""Stage configuration and registry for the ablation pipeline.

No heavy dependencies -- only stdlib + config (pure pathlib).
"""

from dataclasses import dataclass
from pathlib import Path

import config


@dataclass
class StageConfig:
    name: str                        # human-readable label for the ablation table
    embedding: str                   # "base" | "finetuned"
    retrieval: str                   # "dense" | "hybrid" | "rrf"
    use_rerank: bool                 # apply cross-encoder reranker
    llm: str                         # "base" | "finetuned"
    results_dir: Path
    inject_citations: bool = False   # post-hoc citation injection (for ft LLM)
    requires_emb_ft: bool = False    # skip automatically if emb model dir is empty
    use_graph: bool = False          # apply graph expansion after reranking
    requires_graph: bool = False     # skip automatically if graph.json is missing


STAGE_REGISTRY: dict[str, StageConfig] = {
    "base": StageConfig(
        name="Stage 1 — Base RAG",
        embedding="base",
        retrieval="dense",
        use_rerank=False,
        llm="base",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_BASE,
    ),
    "hybrid": StageConfig(
        name="Stage 1b — Hybrid BM25+Dense",
        embedding="base",
        retrieval="hybrid",
        use_rerank=False,
        llm="base",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_BASE / "hybrid",
    ),
    "rrf": StageConfig(
        name="Stage 1c — RRF",
        embedding="base",
        retrieval="rrf",
        use_rerank=False,
        llm="base",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_BASE / "rrf",
    ),
    "rrf_rerank": StageConfig(
        name="Stage 3 — RRF + Rerank",
        embedding="base",
        retrieval="rrf",
        use_rerank=True,
        llm="base",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_RERANK,
    ),
    "graph": StageConfig(
        name="Stage 3b — RRF+Rerank+Graph",
        embedding="base",
        retrieval="rrf",
        use_rerank=True,
        use_graph=True,
        requires_graph=True,
        llm="base",
        inject_citations=True,
        results_dir=config.BASE_DIR / "results" / "stage_graph",
    ),
    "llm_ft": StageConfig(
        name="Stage 4 — Fine-tuned LLM",
        embedding="base",
        retrieval="dense",
        use_rerank=False,
        llm="finetuned",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_LLM_FT,
    ),
    "emb_ft": StageConfig(
        name="Stage 2 — Fine-tuned Embedding",
        embedding="finetuned",
        retrieval="rrf",
        use_rerank=True,
        llm="base",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_EMB_FT,
        requires_emb_ft=True,
    ),
    "full": StageConfig(
        name="Stage 5 — Full Optimized",
        embedding="finetuned",
        retrieval="rrf",
        use_rerank=True,
        llm="finetuned",
        inject_citations=True,
        results_dir=config.RESULTS_DIR_FULL,
        requires_emb_ft=True,
    ),
}

# Ordered list for display / default run
DEFAULT_STAGE_ORDER = [
    "base", "hybrid", "rrf", "rrf_rerank", "graph",
    "llm_ft", "emb_ft", "full",
]
