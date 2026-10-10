"""
tests/test_turkish_legal_rag.py — turkish_legal_rag adapter, gold labeling,
eval-set default, and headline-metric switch.  Offline: no network, no models.
"""

import collections
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from data.data_processor import CorpusChunk, DataProcessor, QAExample


def _load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(
        name, _PROJECT_ROOT / "scripts" / filename,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


adapter = _load_script("_tlr_adapter", "16_prepare_turkish_legal_rag.py")

KNOWN = {"Türk Medeni Kanunu", "Türk Ceza Kanunu"}


def _row(row_id, q, kaynak="Türk Medeni Kanunu", madde="3-", origin="kaggle_batuhankalem"):
    return {"row_id": float(row_id), "soru": q, "cevap": "a", "kaynak": kaynak,
            "madde_no": madde, "source_origin": origin}


class TestAdapterFilters:
    def test_kaggle_only(self):
        rows = [_row(1, "soru bir"), _row(2, "soru iki", origin=None)]
        ex, rep = adapter.build_examples(rows, KNOWN, set())
        assert [e["hf_row_id"] for e in ex] == ["1"]
        assert rep["dropped"]["source_origin"] == 1

    def test_leakage_drop_uses_normalized_text(self):
        train = {adapter.normalize_question("Soru, bir?")}
        rows = [_row(1, "SORU bir"), _row(2, "başka soru")]
        ex, rep = adapter.build_examples(rows, KNOWN, train)
        assert [e["hf_row_id"] for e in ex] == ["2"]
        assert rep["dropped"]["train_leakage"] == 1

    def test_normalize_question_turkish_dotted_i(self):
        """İ (U+0130) must not produce a spurious space after lowercasing.

        Python's str.lower() expands İ to i + combining-dot (two code-points).
        The old local normalize_question used plain .lower() which made
        ``re.sub(r'\\W+', ' ', ...)`` split on the combining dot, turning
        "TAKİP" into "taki p".  The imported normalize_question from
        data.data_processor uses normalize_turkish() which handles this.
        """
        from data.data_processor import normalize_question as nq
        # normalize_turkish maps İ->i and I->ı before lowercasing, so
        # "İSRARLI" -> "israrlı" (dotless ı preserved) and "TAKİP" -> "takip"
        # (no spurious space from combining dot).
        assert nq("İSRARLI TAKİP") == "israrlı takip"
        assert nq("İSTANBUL") == "istanbul"
        assert " " not in nq("TAKİP"), "combining-dot must not create a space"
        # Ensure adapter.normalize_question is the same function (task 2 fix).
        assert adapter.normalize_question("İSRARLI TAKİP") == "israrlı takip"

    def test_unknown_law_drop(self):
        rows = [_row(1, "q1", kaynak="Olmayan Kanun"), _row(2, "q2")]
        ex, rep = adapter.build_examples(rows, KNOWN, set())
        assert len(ex) == 1 and rep["dropped"]["unknown_law"] == 1
        assert rep["kept_per_law"] == {"Türk Medeni Kanunu": 1}

    def test_schema_matches_hmgs_plus_extras(self):
        ex, _ = adapter.build_examples([_row(7, "q")], KNOWN, set())
        assert set(ex[0]) == {"query_id", "question", "answer", "context", "source",
                              "data_type", "madde_no", "hf_row_id", "label_conflict"}
        assert ex[0]["query_id"] == "tlr_7" and ex[0]["madde_no"] == "3"

    def test_gecici_section_recovered_from_text(self):
        rows = [_row(1, "Geçici Madde 5 neyi düzenler?", madde="5-"),
                _row(2, "Madde 5 neyi düzenler?", madde="5-")]
        ex, _ = adapter.build_examples(rows, KNOWN, set())
        assert [e["madde_no"] for e in ex] == ["gecici-5", "5"]

    def test_label_conflict_flagged_and_dropped_from_default_set(self):
        row = _row(1, "Hangi madde?", madde="150-")
        row["cevap"] = "Türk Ceza Kanunu'nun 151. maddesinde belirtilmiştir."
        ok = _row(2, "Hangi madde?", madde="150-")
        ok["cevap"] = "150. maddede belirtilmiştir."
        ex, rep = adapter.build_examples([row, ok], KNOWN, set())
        assert [e["hf_row_id"] for e in ex] == ["2"]
        assert rep["label_conflict"] == 1 and rep["kept"] == 1
        assert rep["label_conflict_rows"][0]["label_conflict"] is True

    def test_conflict_rows_skipped_by_loader(self, tmp_path):
        p = tmp_path / "x.jsonl"
        base = {"question": "q", "answer": "a", "context": "",
                "source": "Türk Medeni Kanunu", "data_type": "", "madde_no": "3"}
        p.write_text(
            json.dumps({**base, "query_id": "a", "label_conflict": True}) + "\n"
            + json.dumps({**base, "query_id": "b", "label_conflict": False}) + "\n",
            encoding="utf-8")
        assert [e.query_id for e in DataProcessor.build_turkish_legal_rag_eval_set(p)] == ["b"]

    def test_madde_normalization(self):
        n = adapter.normalize_madde_no
        assert n("3-") == "3"
        assert n("255-") == "255"
        assert n(" 12 ") == "12"
        assert n("183/A") == "183-a"
        assert n("Ek Madde 3") == "ek-3"
        assert n("Geçici 2-") == "gecici-2"
        assert n(None) is None
        assert n("abc") is None
        # upper-case Turkish İ must not turn GEÇİCİ into something else
        assert n("GEÇİCİ MADDE 5") == "gecici-5"
        assert n("EK GEÇİCİ MADDE 1") == "ekgecici-1"
        assert adapter.refine_madde_no("5", "GEÇİCİ MADDE 5 nedir?", "") == "gecici-5"


def _chunk(cid, source, text, madde_no=None):
    return CorpusChunk(cid, cid, text, source, len(text), madde_no)


class TestExplicitGoldLabels:
    def test_labels_from_source_and_madde_no(self):
        chunks = [
            _chunk("a", "Türk Medeni Kanunu", "Madde 3- İyi niyet", madde_no="3"),
            _chunk("b", "Türk Medeni Kanunu", "Madde 4- Hakim", madde_no="4"),
            _chunk("c", "Türk Ceza Kanunu", "Madde 3- Başka", madde_no="3"),
        ]
        qa = [QAExample("q", "soru", "cevap", "", "Türk Medeni Kanunu", "", madde_no="3")]
        assert DataProcessor.build_relevant_chunk_map(chunks, qa)["q"] == ["a"]

    def test_heading_found_mid_chunk_and_suffix_forms(self):
        chunks = [
            _chunk("a", "Türk Medeni Kanunu", "BİRİNCİ KISIM\nGiriş\n\nMadde 12- Metin"),
            _chunk("b", "Türk Medeni Kanunu", "Madde 183/A- Metin"),
            _chunk("c", "Türk Medeni Kanunu", "Geçici Madde 2- Metin"),
        ]
        qa = [
            QAExample("q1", "", "", "", "Türk Medeni Kanunu", "", madde_no="12"),
            QAExample("q2", "", "", "", "Türk Medeni Kanunu", "", madde_no="183-a"),
            QAExample("q3", "", "", "", "Türk Medeni Kanunu", "", madde_no="gecici-2"),
        ]
        m, cov = DataProcessor.build_relevant_chunk_map(chunks, qa, return_coverage=True)
        assert (m["q1"], m["q2"], m["q3"]) == (["a"], ["b"], ["c"])
        assert cov["by_strategy"]["explicit_madde"] == 3

    def test_dict_example_and_unknown_article_unlabeled(self):
        chunks = [_chunk("a", "Türk Medeni Kanunu", "Madde 3- x", madde_no="3")]
        qa = [{"query_id": "q", "question": "", "answer": "", "context": "",
               "source": "Türk Medeni Kanunu", "madde_no": "99"}]
        assert DataProcessor.build_relevant_chunk_map(chunks, qa)["q"] == []

    def test_eval_set_loader_keeps_fields(self, tmp_path):
        p = tmp_path / "x.jsonl"
        p.write_text(json.dumps({
            "query_id": "tlr_1", "question": "q", "answer": "a", "context": "",
            "source": "Türk Medeni Kanunu", "data_type": "", "madde_no": "3",
            "hf_row_id": "1"}) + "\n", encoding="utf-8")
        ex = DataProcessor.build_turkish_legal_rag_eval_set(p)
        assert ex[0].madde_no == "3" and ex[0].source == "Türk Medeni Kanunu"


class TestCorpusHoldout:
    """build_corpus_chunks must index the eval laws unless holdout is requested."""

    LAWS = ["Türk Ceza Kanunu", "Türk Medeni Kanunu", "Ceza Muhakemesi Kanunu",
            "Türk Borçlar Kanunu", "Türkiye Cumhuriyeti Anayasası",
            "Türkiye Cumhuriyeti İş Kanunu", "Türk Bayrağı Tüzüğü", "Bilgi Edinme Kanunu"]

    def _processor(self, tmp_path, monkeypatch):
        import pandas as pd
        rows = []
        for i, law in enumerate(self.LAWS):
            ctx = (f"MADDE {i + 1}- {law} birinci hüküm " + "metin " * 60 +
                   f"\nMADDE {i + 2}- {law} ikinci hüküm " + "metin " * 60)
            rows.append({"id": f"k{i}", "question": "q", "answer": "a", "context": ctx,
                         "source": law, "data_type": "", "score": 1, "split": "kaggle"})
        csv = tmp_path / "d.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        monkeypatch.setattr(config, "BASE_DIR", tmp_path)  # no extra_laws.jsonl
        return DataProcessor(csv)

    def test_default_corpus_contains_all_eval_laws(self, tmp_path, monkeypatch):
        dp = self._processor(tmp_path, monkeypatch)
        sources = {c.source for c in dp.build_corpus_chunks()}
        assert sources == set(self.LAWS)

    def test_holdout_is_ignored_eval_passages_stay_indexed(self, tmp_path, monkeypatch):
        dp = self._processor(tmp_path, monkeypatch)
        with pytest.warns(DeprecationWarning):
            held = list(dp.build_corpus_chunks(holdout=True))
        assert {c.source for c in held} == set(self.LAWS)

    def test_gold_labels_found_for_every_eval_law(self, tmp_path, monkeypatch):
        dp = self._processor(tmp_path, monkeypatch)
        chunks = list(dp.build_corpus_chunks())
        qa = [QAExample(f"q{i}", "", "", "", law, "", madde_no=str(i + 1))
              for i, law in enumerate(self.LAWS)]
        m = DataProcessor.build_relevant_chunk_map(chunks, qa)
        assert all(m[q.query_id] for q in qa)


class TestEvalSetDefaultAndHeadline:
    def test_default_and_choices(self):
        assert config.DEFAULT_EVAL_SET == "turkish_legal_rag"
        assert {"turkish_legal_rag", "hmgs", "kaggle"} <= set(config.EVAL_SET_CHOICES)
        eval_cli = _load_script("_tlr_eval_cli", "14_eval_all_stages.py")
        assert eval_cli._parse_args([]).eval_set == "turkish_legal_rag"
        assert eval_cli._parse_args(["--eval-set", "hmgs"]).eval_set == "hmgs"
        spec = importlib.util.spec_from_file_location(
            "_tlr_baseline", _PROJECT_ROOT / "run_baseline.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod._parse_args([]).eval_set == "turkish_legal_rag"

    def test_stages_accept_space_separated(self):
        eval_cli = _load_script("_tlr_eval_cli2", "14_eval_all_stages.py")
        args = eval_cli._parse_args(["--stages", "base", "rrf_rerank", "graph"])
        assert args.stages == "base,rrf_rerank,graph"

    def test_headline_mode_rule(self):
        from pipeline.evaluation import headline_mode
        assert headline_mode(100, 200) == "chunk"      # exactly 50%
        assert headline_mode(195, 195) == "chunk"
        assert headline_mode(99, 200) == "source"
        assert headline_mode(0, 0) == "source"
        assert headline_mode(40, 100, min_fraction=0.4) == "chunk"

    def test_ablation_table_switches_primary(self, capsys):
        from pipeline.evaluation import print_ablation_table

        def res(labeled):
            return {
                "hyperparameters": {"stage_name": "Base"},
                "labeling_coverage": {"labeled": labeled, "total": 10},
                "source_hit_metrics": {"source_hit_at_5_all": 0.5,
                                       "source_labeled_queries": 10},
                "retrieval_metrics": {"recall_at_5": 0.25, "num_queries": labeled},
            }
        print_ablation_table({"base": res(8)}, stage_order=["base"])
        out = capsys.readouterr().out
        assert "PRIMARY ABLATION TABLE  (chunk-level" in out
        print_ablation_table({"base": res(2)}, stage_order=["base"])
        out = capsys.readouterr().out
        assert "PRIMARY ABLATION TABLE  (source-level" in out


class TestKaggleEvalSplit:
    """kaggle eval rows: round-robin over contexts, no train questions; the
    train sets never contain an eval question."""

    def _processor(self, tmp_path, monkeypatch, n_ctx=3, per_ctx=3):
        import pandas as pd
        rows = []
        for c in range(n_ctx):
            for j in range(per_ctx):
                rows.append({"id": f"k{c}_{j}", "question": f"Soru {c}-{j}?", "answer": "a",
                             "context": f"MADDE {c + 1}- bağlam {c} " + "metin " * 40,
                             "source": "L", "data_type": "", "score": 1, "split": "kaggle"})
        rows.append({"id": "t1", "question": "SORU 0-0", "answer": "a", "context": None,
                     "source": "", "data_type": "", "score": 1, "split": "train"})
        rows.append({"id": "t2", "question": "Başka soru", "answer": "a", "context": None,
                     "source": "", "data_type": "", "score": 1, "split": "train"})
        csv = tmp_path / "d.csv"
        pd.DataFrame(rows).to_csv(csv, index=False)
        monkeypatch.setattr(config, "BASE_DIR", tmp_path)
        monkeypatch.setattr(config, "TLR_PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(config, "TLR_DATA_PATH", tmp_path / "none.jsonl")
        monkeypatch.setattr(config, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(config, "HMGS_DATA_PATH", tmp_path / "none.csv")
        return DataProcessor(csv)

    def test_one_question_per_context_first(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "QA_EVAL_EXPECTED", 3)
        dp = self._processor(tmp_path, monkeypatch)
        ev = dp.build_qa_eval_set()
        assert len(ev) == 3 and len({e.context for e in ev}) == 3

    def test_round_robin_fills_up_to_the_cap(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "QA_EVAL_EXPECTED", 5)
        dp = self._processor(tmp_path, monkeypatch)
        ev = dp.build_qa_eval_set()
        per_ctx = sorted(collections.Counter(e.context for e in ev).values())
        assert len(ev) == 5 and per_ctx == [1, 2, 2]
        assert [e.query_id for e in ev] == [e.query_id for e in dp.build_qa_eval_set()]

    def test_question_also_in_train_split_is_not_an_eval_question(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "QA_EVAL_EXPECTED", 100)
        dp = self._processor(tmp_path, monkeypatch)
        assert "k0_0" not in {e.query_id for e in dp.build_qa_eval_set()}

    def test_training_sets_exclude_eval_questions(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "QA_EVAL_EXPECTED", 3)
        dp = self._processor(tmp_path, monkeypatch)
        eval_q = {e.question for e in dp.build_qa_eval_set()}
        kt = dp.build_kaggle_train_set()
        assert kt and not ({q.question for q in kt} & eval_q)
        assert {q.query_id for q in dp.build_qa_train_set()} == {"t1", "t2"}
