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

Missing components are None (never defaulted to 0.0 or swapped for a proxy).
Each scenario is computed on the available components with weights
renormalised to sum to 1, and the output records which components were used,
which were missing, the effective weights, and (optionally) each component's n.
The faithfulness source ("nli" or "judge") is chosen explicitly by the caller.
"""

from __future__ import annotations


_S1_WEIGHTS = {"retrieval_mrr": 0.35, "f1": 0.40, "faithfulness": 0.25}
_S2_WEIGHTS = {"f1": 0.70, "semantic_similarity": 0.30}
_S3_WEIGHTS = {"relevancy": 1.0, "faithfulness": 1.0, "coherence": 1.0}


def _num(x):
    return None if x is None else float(x)


def _weighted(values: dict, weights: dict, n_by_component: dict | None = None) -> tuple:
    """Weighted mean over non-None values with renormalised weights.

    Returns (score|None, record) where record lists used/missing components.
    """
    used = {k: float(v) for k, v in values.items() if v is not None}
    missing = [k for k in weights if k not in used]
    if not used:
        score, eff = None, {}
    else:
        tot = sum(weights[k] for k in used)
        eff = {k: weights[k] / tot for k in used}
        score = sum(eff[k] * used[k] for k in used)
    rec = {"used": used, "missing": missing, "weights": eff,
           "complete": not missing}
    if n_by_component:
        rec["n"] = {k: n_by_component.get(k) for k in used}
    return score, rec


def compute_scenario1_score(retrieval_metrics: dict, qa_metrics: dict,
                            faithfulness_score: float | None) -> float | None:
    """Scenario 1: 0.35*MRR + 0.40*F1 + 0.25*faithfulness (renormalised if any is None)."""
    return _weighted({
        "retrieval_mrr": _num(retrieval_metrics.get("mrr")),
        "f1": _num(qa_metrics.get("f1")),
        "faithfulness": _num(faithfulness_score),
    }, _S1_WEIGHTS)[0]


def compute_scenario2_score(qa_metrics: dict, semantic_similarity: float | None) -> float | None:
    """Scenario 2: 0.70*F1 + 0.30*Sim (F1 only if Sim is None)."""
    return _weighted({
        "f1": _num(qa_metrics.get("f1")),
        "semantic_similarity": _num(semantic_similarity),
    }, _S2_WEIGHTS)[0]


def compute_scenario3_score(relevancy_score, faithfulness_score, coherence_score) -> float | None:
    """Scenario 3: mean of the available components; None if none available."""
    return _weighted({
        "relevancy": _num(relevancy_score),
        "faithfulness": _num(faithfulness_score),
        "coherence": _num(coherence_score),
    }, _S3_WEIGHTS)[0]


def compute_all_scenario_scores(
    retrieval_metrics: dict,
    qa_metrics: dict,
    faithfulness_score: float | None = None,
    semantic_similarity: float | None = None,
    llm_scores: dict | None = None,
    faithfulness_source: str = "nli",
    n_by_component: dict | None = None,
) -> dict:
    """Compute the three scenario scores with explicit component accounting.

    Args:
        faithfulness_score: NLI-vs-context faithfulness (used when
            faithfulness_source == "nli").
        llm_scores: {"relevancy", "faithfulness", "coherence"} judge scores
            (None/absent = missing). Judge faithfulness is used only when
            faithfulness_source == "judge"; there is no silent switching.
        faithfulness_source: "nli" or "judge".
        n_by_component: optional {"retrieval_mrr": n, "f1": n, "faithfulness": n,
            "semantic_similarity": n, "relevancy": n, "coherence": n} recorded
            in the output so mixed-n comparisons are visible.

    Returns:
        {"scenario1": float|None, "scenario2": float|None, "scenario3": float|None,
         "faithfulness_source": str,
         "components": {"scenario1": {used, missing, weights, complete[, n]}, ...}}
    """
    if faithfulness_source not in ("nli", "judge"):
        raise ValueError(f"faithfulness_source must be 'nli' or 'judge', got {faithfulness_source!r}")
    llm = llm_scores or {}
    faith = _num(faithfulness_score) if faithfulness_source == "nli" else _num(llm.get("faithfulness"))

    s1, r1 = _weighted({
        "retrieval_mrr": _num(retrieval_metrics.get("mrr")),
        "f1": _num(qa_metrics.get("f1")),
        "faithfulness": faith,
    }, _S1_WEIGHTS, n_by_component)
    s2, r2 = _weighted({
        "f1": _num(qa_metrics.get("f1")),
        "semantic_similarity": _num(semantic_similarity),
    }, _S2_WEIGHTS, n_by_component)
    s3, r3 = _weighted({
        "relevancy": _num(llm.get("relevancy")),
        "faithfulness": faith,
        "coherence": _num(llm.get("coherence")),
    }, _S3_WEIGHTS, n_by_component)

    return {
        "scenario1": None if s1 is None else round(s1, 6),
        "scenario2": None if s2 is None else round(s2, 6),
        "scenario3": None if s3 is None else round(s3, 6),
        "faithfulness_source": faithfulness_source,
        "components": {"scenario1": r1, "scenario2": r2, "scenario3": r3},
    }
