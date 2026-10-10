"""data/tlr_labels.py: gold article labels checked against the law text."""

from data.data_processor import CorpusChunk
from data.tlr_labels import (
    answer_coverage, apply_label_checks, check_label, named_articles, neighbour_keys,
)

LAW = "İş Kanunu"
CHUNKS = [
    CorpusChunk("c67", "d", "Günlük çalışma\nMadde 67 - Başlama ve bitiş saatleri işçilere duyurulur.",
                LAW, 60, "67"),
    CorpusChunk("c68", "d", "Ara dinlenmesi\nMadde 68 - Dört saat veya daha kısa süreli işlerde "
                "onbeş dakika ara dinlenmesi verilir.", LAW, 90, "68"),
    CorpusChunk("c70", "d", "Gece çalışmaları\nMadde 70 - Gece süresi yedibuçuk saati geçemez; "
                "ara dinlenmesi süreleri saklıdır.", LAW, 80, "70"),
]


def _row(qid, madde, answer, question="Soru?", conflict=False):
    return {"query_id": qid, "source": LAW, "madde_no": madde, "question": question,
            "answer": answer, "label_conflict": conflict}


def test_coverage_uses_word_stems():
    assert answer_coverage("ara dinlenmesi onbeş dakikadır", CHUNKS[1].text) == 1.0
    assert answer_coverage("tamamen alakasız kelimeler", CHUNKS[1].text) == 0.0


def test_neighbour_keys_keep_article_kind():
    assert [k for _, k in neighbour_keys("2", 2)] == ["3", "1", "4"]
    assert [k for _, k in neighbour_keys("gecici-1", 1)] == ["gecici-2"]
    assert neighbour_keys("183-a") == []


def test_named_articles():
    assert named_articles("TCK 101. maddesi", "madde 5 ve 12 inci madde") == ["101", "12", "5"]


def test_off_by_one_label_moves_to_the_article_holding_the_answer():
    eval_rows, conflicts, report = apply_label_checks(
        [_row("q1", "67", "Dört saatten kısa işlerde onbeş dakika ara dinlenmesi verilir.")],
        CHUNKS)
    r = eval_rows[0]
    assert r["madde_no"] == "68" and r["madde_no_hf"] == "67"
    assert r["label_check"]["status"] == "relabelled" and r["label_check"]["offset"] == 1
    assert report["counts"]["main_relabelled"] == 1 and not conflicts


def test_correct_label_is_kept():
    r = apply_label_checks([_row("q", "68", "Ara dinlenmesi onbeş dakika verilir.")], CHUNKS)[0][0]
    assert r["madde_no"] == "68" and r["label_check"]["status"] == "ok"


def test_named_article_beats_neighbour_only_when_it_covers_better():
    texts = {(LAW, "68"): CHUNKS[1].text, (LAW, "70"): CHUNKS[2].text,
             (LAW, "67"): CHUNKS[0].text}
    # the answer cites "70. madde" and its content is in 70
    d = check_label(_row("q", "67", "70. madde uyarınca gece süresi yedibuçuk saati geçemez."), texts)
    assert d["madde_no"] == "70" and d["reason"] == "named"


def test_unresolved_conflict_stays_out_and_rerun_is_idempotent():
    rows = [_row("qc", "67", "tamamen alakasız bir cevap metni", conflict=True),
            _row("qr", "67", "Onbeş dakika ara dinlenmesi verilir.")]
    eval1, conf1, _ = apply_label_checks(rows, CHUNKS)
    assert [r["query_id"] for r in conf1] == ["qc"]
    eval2, conf2, _ = apply_label_checks(eval1 + conf1, CHUNKS)
    assert [r["madde_no"] for r in eval2] == [r["madde_no"] for r in eval1]
    assert eval2[0]["madde_no_hf"] == "67"


def test_loader_switches_between_checked_and_hf_labels(tmp_path, monkeypatch):
    import json
    import config
    from data.data_processor import DataProcessor
    path = tmp_path / "tlr.jsonl"
    rows = [
        {"query_id": "a", "question": "q", "answer": "x", "source": LAW,
         "madde_no": "68", "madde_no_hf": "67", "label_conflict": False,
         "label_conflict_hf": False},
        {"query_id": "b", "question": "q", "answer": "x", "source": LAW,
         "madde_no": "70", "madde_no_hf": "69", "label_conflict": False,
         "label_conflict_hf": True},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(config, "TLR_USE_LABEL_FIXES", True)
    fixed = DataProcessor.build_turkish_legal_rag_eval_set(path)
    assert [(q.query_id, q.madde_no) for q in fixed] == [("a", "68"), ("b", "70")]
    monkeypatch.setattr(config, "TLR_USE_LABEL_FIXES", False)
    hf = DataProcessor.build_turkish_legal_rag_eval_set(path)
    assert [(q.query_id, q.madde_no) for q in hf] == [("a", "67")]
