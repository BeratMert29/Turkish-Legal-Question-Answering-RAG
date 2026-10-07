import numpy as np
import pytest

from evaluation.hallucination import (
    _classify_result, gold_rank, stratified_sample, run_hallucination_analysis,
)
from evaluation.nli import entailment_index, nli_context_faithfulness, split_sentences


class _Cfg:
    def __init__(self, id2label):
        self.id2label = id2label


class FakeNLI:
    """Entailment when hypothesis text appears in premise. Label order configurable."""

    def __init__(self, id2label):
        self.config = _Cfg(id2label)
        self.ent = [i for i, l in id2label.items() if l.lower() == "entailment"][0]
        self.calls = []

    def predict(self, pairs, batch_size=8):
        self.calls.append(list(pairs))
        out = np.full((len(pairs), 3), -5.0)
        for i, (prem, hyp) in enumerate(pairs):
            ok = hyp.strip(" .") in prem
            out[i, self.ent if ok else (self.ent + 1) % 3] = 5.0
        return out


ORDER_A = {0: "entailment", 1: "neutral", 2: "contradiction"}
ORDER_B = {0: "contradiction", 1: "neutral", 2: "entailment"}


@pytest.mark.parametrize("order", [ORDER_A, ORDER_B])
def test_entailment_index_from_id2label(order):
    m = FakeNLI(order)
    assert entailment_index(m) == m.ent


def test_entailment_index_missing_raises():
    with pytest.raises(ValueError):
        entailment_index(FakeNLI.__new__(FakeNLI))


def test_split_sentences_strips_citations():
    assert split_sentences("Birinci cümle [Kaynak 1]. İkinci cümle!") == [
        "Birinci cümle.", "İkinci cümle!"]


@pytest.mark.parametrize("order", [ORDER_A, ORDER_B])
def test_context_faithfulness_scores_against_context(order):
    m = FakeNLI(order)
    preds = [
        {"query_id": "q1", "predicted": "Madde yirmi. Ceza beş yıl.",
         "retrieved_chunks": [{"text": "x"}, {"text": "Madde yirmi. Ceza beş yıl."}]},
        {"query_id": "q2", "predicted": "Uydurma bir cümle.",
         "retrieved_chunks": [{"text": "alakasız"}]},
        {"query_id": "q3", "predicted": "", "retrieved_chunks": [{"text": "a"}]},
    ]
    r = nli_context_faithfulness(preds, m)
    by = {s["query_id"]: s for s in r["per_sample"]}
    assert by["q1"]["score"] > 0.99 and by["q2"]["score"] < 0.01
    assert by["q3"]["score"] is None and r["n"] == 2 and r["n_skipped"] == 1
    # premise is chunk text, never the gold answer
    assert all(p[0] in ("x", "Madde yirmi. Ceza beş yıl.", "alakasız")
               for p in m.calls[0])


def test_context_faithfulness_query_ids_filter():
    m = FakeNLI(ORDER_A)
    preds = [{"query_id": f"q{i}", "predicted": "a.", "retrieved_chunks": [{"text": "a."}]}
             for i in range(3)]
    r = nli_context_faithfulness(preds, m, query_ids=["q1"])
    assert r["n"] == 1 and r["per_sample"][0]["query_id"] == "q1"


def _res(qid, exp, retrieved):
    return {"query_id": qid, "expected_source": exp, "retrieved_sources": retrieved}


def test_gold_rank_and_strata():
    assert gold_rank(_res("a", "TCK", ["TCK", "x"])) == 1
    assert _classify_result(_res("a", "TCK", ["TCK"])) == "hit"
    assert _classify_result(_res("a", "TCK", ["x", "tck"])) == "partial"
    assert _classify_result(_res("a", "TCK", ["x", "y"])) == "miss"
    assert _classify_result({"query_id": "z"}) is None
    assert gold_rank({"relevant": ["c3"], "retrieved": ["c1", "c3"]}) == 2


def test_stratified_sample_independent_of_scores():
    rs = []
    for i in range(30):
        # scores all identical/huge: old threshold logic would put all in "hit"
        r = _res(f"q{i}", "TCK", ["TCK"] if i % 3 == 0 else (["x", "TCK"] if i % 3 == 1 else ["x"]))
        r["retrieved_chunks"] = [{"score": 99.0}]
        rs.append(r)
    s = stratified_sample(rs, 9)
    assert len(s["hits"]) == len(s["partial"]) == len(s["misses"]) == 3
    assert all(_classify_result(r) == "miss" for r in s["misses"])
    # unlabeled results excluded
    assert sum(len(v) for v in stratified_sample([{"query_id": "u"}], 9).values()) == 0


@pytest.mark.parametrize("order", [ORDER_A, ORDER_B])
def test_hallucination_analysis_uses_context(order):
    m = FakeNLI(order)
    sample = {"hits": [{"query_id": "q1", "predicted": "Doğru cümle.", "expected": "gold"}],
              "partial": [], "misses": []}
    retrieved = {"q1": [{"text": "Doğru cümle."}]}
    out = run_hallucination_analysis(sample, retrieved, m)
    assert out["per_sample"][0]["context_grounded"] is True
    assert out["summary"]["context_grounding_rate"] == 1.0
