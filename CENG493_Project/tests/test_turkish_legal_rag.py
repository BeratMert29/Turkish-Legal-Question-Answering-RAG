"""
tests/test_turkish_legal_rag.py — turkish_legal_rag adapter, gold labeling,
eval-set default, and headline-metric switch.  Offline: no network, no models.
"""

import importlib.util
import json
import sys
from pathlib import Path

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

    def test_unknown_law_drop(self):
        rows = [_row(1, "q1", kaynak="Olmayan Kanun"), _row(2, "q2")]
        ex, rep = adapter.build_examples(rows, KNOWN, set())
        assert len(ex) == 1 and rep["dropped"]["unknown_law"] == 1
        assert rep["kept_per_law"] == {"Türk Medeni Kanunu": 1}

    def test_schema_matches_hmgs_plus_extras(self):
        ex, _ = adapter.build_examples([_row(7, "q")], KNOWN, set())
        assert set(ex[0]) == {"query_id", "question", "answer", "context", "source",
                              "data_type", "madde_no", "hf_row_id"}
        assert ex[0]["query_id"] == "tlr_7" and ex[0]["madde_no"] == "3"

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
