"""pipeline -- shared RAG evaluation stages for Turkish Legal QA.

Only lightweight imports at module level so the package loads on CPU
without torch / faiss / sentence_transformers.
"""

from pipeline.stages import StageConfig, STAGE_REGISTRY, DEFAULT_STAGE_ORDER

__all__ = [
    "StageConfig",
    "STAGE_REGISTRY",
    "DEFAULT_STAGE_ORDER",
]
