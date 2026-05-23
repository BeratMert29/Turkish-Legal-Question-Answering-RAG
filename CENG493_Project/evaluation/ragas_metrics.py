"""RAGAS evaluation metrics using Ollama as LLM + embedding backend."""
import logging
from typing import Optional

log = logging.getLogger(__name__)


def compute_ragas_metrics(
    predictions: list[dict],
    llm_model: str,
    embedding_model: str = "nomic-embed-text",
    ollama_base: str = "http://localhost:11434",
    sample_size: int = 50,
) -> Optional[dict]:
    """
    Compute RAGAS metrics: faithfulness, answer_relevancy,
    context_precision, context_recall.

    predictions: list of dicts with keys:
        question, predicted, expected, retrieved_chunks

    Returns dict with ragas_* keys, or None if ragas not installed.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            answer_correctness,
            context_precision,
            context_recall,
        )
        from langchain_ollama import ChatOllama, OllamaEmbeddings
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as e:
        log.warning("RAGAS dependencies not installed (%s). Run: pip install ragas langchain-ollama", e)
        return None

    import random as _random
    _rng = _random.Random(42)
    pool = [p for p in predictions if p.get("predicted")]
    sampled = _rng.sample(pool, min(sample_size, len(pool)))

    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    for pred in sampled:
        chunks = pred.get("retrieved_chunks", [])
        contexts = [c.get("text", "") for c in chunks[:5] if c.get("text")]
        if not contexts:
            continue
        data["question"].append(pred.get("question", ""))
        data["answer"].append(pred.get("predicted", ""))
        data["contexts"].append(contexts)
        data["ground_truth"].append(pred.get("expected", ""))

    if not data["question"]:
        log.warning("No valid predictions for RAGAS evaluation.")
        return None

    dataset = Dataset.from_dict(data)

    try:
        llm = ChatOllama(model=llm_model, base_url=ollama_base, temperature=0)
        embeddings = OllamaEmbeddings(model=embedding_model, base_url=ollama_base)

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall],
            llm=LangchainLLMWrapper(llm),
            embeddings=LangchainEmbeddingsWrapper(embeddings),
            raise_exceptions=False,
        )

        def _to_float(val):
            """Convert a scalar or per-sample list to a rounded float, or None."""
            import math
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                finite = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
                if not finite:
                    return None
                val = sum(finite) / len(finite)
            try:
                f = float(val)
            except (TypeError, ValueError):
                return None
            if math.isnan(f):
                return None
            return round(f, 4)

        return {
            "ragas_faithfulness": _to_float(result["faithfulness"]),
            "ragas_answer_relevancy": _to_float(result["answer_relevancy"]),
            "ragas_answer_correctness": _to_float(result["answer_correctness"]),
            "ragas_context_precision": _to_float(result["context_precision"]),
            "ragas_context_recall": _to_float(result["context_recall"]),
        }

    except Exception as exc:
        log.warning("RAGAS evaluation failed: %s", exc)
        return None
