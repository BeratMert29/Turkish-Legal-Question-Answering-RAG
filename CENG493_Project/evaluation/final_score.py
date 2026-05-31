"""
evaluation/final_score.py — Rubric-based final score computation

Three evaluation scenarios from the teacher's rubric:

  Scenario 1 (Gold Q+A+Doc):
    Final = 0.35 * R + 0.40 * A + 0.25 * G
    R = MRR  |  A = F1  |  G = faithfulness_score

  Scenario 2 (Gold Q+A):
    Final = 0.70 * A + 0.30 * Sim
    A = F1  |  Sim = semantic_similarity

  Scenario 3 (No Gold Data):
    Final = avg(relevancy, faithfulness, coherence)
"""

from __future__ import annotations


def compute_scenario1_score(
    retrieval_metrics: dict,
    qa_metrics: dict,
    faithfulness_score: float,
) -> float:
    """
    Scenario 1: Gold Q+A+Doc
    Final = 0.35*R + 0.40*A + 0.25*G
    R = MRR, A = F1, G = faithfulness_score
    """
    r = float(retrieval_metrics.get("mrr", 0.0))
    a = float(qa_metrics.get("f1", 0.0))
    g = float(faithfulness_score) if faithfulness_score is not None else 0.0
    return 0.35 * r + 0.40 * a + 0.25 * g


def compute_scenario2_score(
    qa_metrics: dict,
    semantic_similarity: float,
) -> float:
    """
    Scenario 2: Gold Q+A
    Final = 0.70*F1 + 0.30*Sim
    """
    a   = float(qa_metrics.get("f1", 0.0))
    sim = float(semantic_similarity) if semantic_similarity is not None else 0.0
    return 0.70 * a + 0.30 * sim


def compute_scenario3_score(
    relevancy_score: float,
    faithfulness_score: float,
    coherence_score: float,
) -> float:
    """
    Scenario 3: No Gold Data
    Final = avg(relevancy, faithfulness, coherence)
    Only averages over non-None metrics so a missing metric does not unfairly
    zero out the denominator-averaged score.
    """
    scores = [float(s) for s in (relevancy_score, faithfulness_score, coherence_score) if s is not None]
    return sum(scores) / len(scores) if scores else 0.0


def compute_all_scenario_scores(
    retrieval_metrics: dict,
    qa_metrics: dict,
    faithfulness_score: float,
    semantic_similarity: float | None = None,
    llm_scores: dict | None = None,
) -> dict:
    """
    Compute all three scenario final scores.

    Args:
        retrieval_metrics:  dict with at least "mrr"
        qa_metrics:         dict with at least "f1"
        faithfulness_score: NLI or LLM faithfulness (float 0-1)
        semantic_similarity: mean cosine sim between predicted/expected (float 0-1)
        llm_scores:         dict with keys "relevancy", "faithfulness", "coherence"
                            (each a float 0-1; falls back to available scores)

    Returns:
        {
            "scenario1": float,   # 0.35R + 0.40A + 0.25G
            "scenario2": float,   # 0.70F1 + 0.30Sim
            "scenario3": float,   # avg(relevancy, faithfulness, coherence)
        }
    """
    llm = llm_scores or {}

    # Resolve faithfulness: prefer LLM-based if available, else NLI
    llm_faith = llm.get("faithfulness")
    g = float(llm_faith) if llm_faith is not None else float(faithfulness_score or 0.0)

    # Scenario 1
    s1 = compute_scenario1_score(retrieval_metrics, qa_metrics, g)

    # Scenario 2 — use semantic_similarity if provided, else F1 proxy
    scenario2_used_f1_fallback = semantic_similarity is None
    sim = float(semantic_similarity) if semantic_similarity is not None else float(qa_metrics.get("f1", 0.0))
    s2 = compute_scenario2_score(qa_metrics, sim)

    # Scenario 3 — use LLM scores; fall back to available proxies
    scenario3_used_f1_fallback = "relevancy" not in llm
    relevancy  = float(llm.get("relevancy",  qa_metrics.get("f1", 0.0)))
    coherence  = float(llm.get("coherence",  0.0))
    s3 = compute_scenario3_score(relevancy, g, coherence)

    result = {
        "scenario1": round(s1, 6),
        "scenario2": round(s2, 6),
        "scenario3": round(s3, 6),
    }
    if scenario2_used_f1_fallback:
        result["scenario2_used_f1_fallback"] = True
    if scenario3_used_f1_fallback:
        result["scenario3_used_f1_fallback"] = True
    return result
