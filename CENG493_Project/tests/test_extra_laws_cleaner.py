"""Tests for data.extra_laws_cleaner and unique chunk_ids in build_corpus_chunks."""
import json

import pandas as pd

import config
from data.data_processor import DataProcessor
from data.extra_laws_cleaner import clean_extra_law_records

SRC = "Test Kanunu"
BODY = " Hüküm metni burada yer alır." * 8


def _rec(n, head="MADDE", sep="-", extra="", doc=None):
    return {"source": SRC, "doc_id": doc or f"{SRC}_madde_{n}",
            "text": f"{head} {n}{sep} (1){BODY}{extra}"}


def test_stranded_gecici_marker_relabels_following_records():
    recs = [_rec(1), _rec(2, extra="\nGEÇİCİ"), _rec(1, doc=f"{SRC}_madde_1")]
    out, st = clean_extra_law_records(recs)
    assert [r["doc_id"] for r in out] == [
        f"{SRC}_madde_1", f"{SRC}_madde_2", f"{SRC}_gecici_madde_1"]
    assert out[2]["text"].startswith("GEÇİCİ MADDE 1")
    assert not out[1]["text"].rstrip().endswith("GEÇİCİ")
    assert st["relabelled_gecici"] == 1


def test_ek_and_ek_gecici_markers():
    recs = [_rec(1), _rec(2, extra="\nEk"), _rec(1), _rec(2, extra="\nEk Geçici"), _rec(1)]
    out, _ = clean_extra_law_records(recs)
    assert [r["doc_id"].split(SRC + "_")[1] for r in out] == [
        "madde_1", "madde_2", "ek_madde_1", "ek_madde_2", "ekgecici_madde_1"]


def test_letter_article_gets_distinct_doc_id():
    recs = [_rec(5), {"source": SRC, "doc_id": f"{SRC}_madde_5",
                      "text": f"MADDE 5/A- (Ek: 1/1/2020-1/1 md.){BODY}"},
            {"source": SRC, "doc_id": f"{SRC}_madde_309",
             "text": f"Madde 309/ç- (Ek: 1/1/2020-1/1 md.){BODY}"}]
    out, _ = clean_extra_law_records(recs)
    assert [r["doc_id"] for r in out] == [
        f"{SRC}_madde_5", f"{SRC}_madde_5-a", f"{SRC}_madde_309-ç"]


def test_pdf_table_numbers_and_table_rows_dropped():
    recs = [_rec(1), _rec(32313), 
            {"source": SRC, "doc_id": f"{SRC}_madde_2",
             "text": "Madde 2 12/7/2012\n1/1/2013 tarihinden geçerli " + BODY},
            {"source": SRC, "doc_id": f"{SRC}_madde_3",
             "text": "Madde 3\n40, 184, 189, 212, 215, 335 " + BODY}]
    out, st = clean_extra_law_records(recs)
    assert [r["doc_id"] for r in out] == [f"{SRC}_madde_1"]
    assert st["dropped_pdf_table_number"] == 1 and st["dropped_amendment_table"] == 2


def test_amending_law_section_after_restart_dropped():
    body = [_rec(n) for n in range(1, 60)]
    amend = {"source": SRC, "doc_id": f"{SRC}_madde_1",
             "text": "MADDE 1 – 12/1/2011 tarihli ve 6100 sayılı Kanunun 373 üncü maddesi" + BODY}
    out, st = clean_extra_law_records(body + [amend])
    assert len(out) == 59 and st.get("dropped_amending_law", 0) + st.get("dropped_amendment_table", 0) == 1


def _processor(tmp_path, monkeypatch, records):
    (tmp_path / "data").mkdir()
    with open(tmp_path / "data" / "extra_laws.jsonl", "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    csv = tmp_path / "d.csv"
    pd.DataFrame([{"id": "k0", "question": "q", "answer": "a", "context": "",
                   "source": "S", "data_type": "", "score": 1, "split": "kaggle"}]).to_csv(csv, index=False)
    monkeypatch.setattr(config, "BASE_DIR", tmp_path)
    return DataProcessor(csv)


def test_build_corpus_chunk_ids_unique_with_colliding_records(tmp_path, monkeypatch):
    # Same doc_id twice with different text (versions of one article) + a clean repeat.
    recs = [_rec(1), _rec(1, extra=" farklı sürüm metni"), _rec(2)]
    dp = _processor(tmp_path, monkeypatch, recs)
    ids = [c.chunk_id for c in dp.build_corpus_chunks()]
    assert len(ids) == 3 and len(set(ids)) == 3


def test_build_corpus_chunk_ids_unique_when_chunk_ids_collide(tmp_path, monkeypatch):
    # Different doc_ids that still yield the same chunk_id string.
    recs = [{"source": SRC, "doc_id": "x_0", "text": f"MADDE 1-{BODY} a"},
            {"source": SRC, "doc_id": "x", "text": f"MADDE 1-{BODY} bb"}]
    dp = _processor(tmp_path, monkeypatch, recs)
    ids = [c.chunk_id for c in dp.build_corpus_chunks()]
    assert len(ids) == len(set(ids)) == 2


def test_stranded_titles_move_to_the_article_they_name():
    recs = [
        {"source": "HMK", "doc_id": "HMK_madde_4",
         "text": "MADDE 4- (1) Sulh hukuk mahkemeleri\nbakar.\nİKİNCİ AYIRIM\nYetki\nGenel kural"},
        {"source": "HMK", "doc_id": "HMK_madde_5",
         "text": "MADDE 5- (1) Yetki kurallara tabidir.\nGenel yetkili mahkeme"},
        {"source": "HMK", "doc_id": "HMK_madde_6", "text": "MADDE 6- (1) Yerleşim yeri mahkemesi."},
    ]
    out, stats = clean_extra_law_records(recs)
    assert out[0]["text"].endswith("bakar.")
    assert out[1]["text"].startswith("İKİNCİ AYIRIM\nYetki\nGenel kural\nMADDE 5-")
    assert out[1]["text"].endswith("tabidir.")
    assert out[2]["text"].startswith("Genel yetkili mahkeme\nMADDE 6-")
    assert stats["moved_title_lines"] == 4
