"""
test_pipeline.py -- tests for the shared pipeline package.

Covers:
  - Package imports (light env, no heavy deps)
  - StageConfig, STAGE_REGISTRY, DEFAULT_STAGE_ORDER consistency
  - CLI arg parsing for both scripts (defaults, --help)
  - End-to-end dry run of retrieve -> evaluate with fake objects
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

_PROJECT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Package import tests
# ---------------------------------------------------------------------------

class TestPipelineImports:
    """Verify the pipeline package is importable under the light test env."""

    def test_import_pipeline_package(self):
        import pipeline
        assert hasattr(pipeline, "StageConfig")
        assert hasattr(pipeline, "STAGE_REGISTRY")
        assert hasattr(pipeline, "DEFAULT_STAGE_ORDER")

    def test_import_stages(self):
        from pipeline.stages import StageConfig, STAGE_REGISTRY, DEFAULT_STAGE_ORDER
        assert isinstance(STAGE_REGISTRY, dict)
        assert isinstance(DEFAULT_STAGE_ORDER, list)
        assert len(STAGE_REGISTRY) > 0

    def test_import_retrieval(self):
        from pipeline.retrieval import retrieve, auto_build_graph
        assert callable(retrieve)
        assert callable(auto_build_graph)

    def test_import_data_loading(self):
        from pipeline.data_loading import load_external_corpus, load_external_qa
        assert callable(load_external_corpus)
        assert callable(load_external_qa)

    def test_import_evaluation(self):
        from pipeline.evaluation import (
            prepare_metric_input,
            run_generation_loop,
            save_stage_results,
            print_ablation_table,
            run_stage,
        )
        assert callable(prepare_metric_input)
        assert callable(run_generation_loop)
        assert callable(save_stage_results)
        assert callable(print_ablation_table)
        assert callable(run_stage)


# ---------------------------------------------------------------------------
# Stage registry tests
# ---------------------------------------------------------------------------

_EXPECTED_ORDER = [
    "base", "hybrid", "rrf", "rrf_rerank", "graph",
    "llm_ft", "emb_ft", "full",
]


class TestStageRegistry:
    """Stage registry matches the old DEFAULT_STAGE_ORDER."""

    def test_default_stage_order_matches(self):
        from pipeline.stages import DEFAULT_STAGE_ORDER
        assert DEFAULT_STAGE_ORDER == _EXPECTED_ORDER

    def test_all_stages_in_registry(self):
        from pipeline.stages import STAGE_REGISTRY, DEFAULT_STAGE_ORDER
        for key in DEFAULT_STAGE_ORDER:
            assert key in STAGE_REGISTRY, f"Stage '{key}' missing from registry"

    def test_stage_config_fields(self):
        from pipeline.stages import STAGE_REGISTRY
        for key, cfg in STAGE_REGISTRY.items():
            assert hasattr(cfg, "name")
            assert hasattr(cfg, "embedding")
            assert hasattr(cfg, "retrieval")
            assert hasattr(cfg, "use_rerank")
            assert hasattr(cfg, "llm")
            assert hasattr(cfg, "results_dir")
            assert isinstance(cfg.results_dir, Path)

    def test_registry_keys_match_order(self):
        from pipeline.stages import STAGE_REGISTRY, DEFAULT_STAGE_ORDER
        assert set(DEFAULT_STAGE_ORDER) == set(STAGE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# CLI arg-parsing tests
# ---------------------------------------------------------------------------

class TestEvalAllStagesCLI:
    """Argument parsing for 14_eval_all_stages.py."""

    def test_parse_defaults(self):
        sys.path.insert(0, str(_PROJECT / "scripts"))
        try:
            # Import the parse function
            import importlib
            spec = importlib.util.spec_from_file_location(
                "_eval_cli",
                _PROJECT / "scripts" / "14_eval_all_stages.py",
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            args = mod._parse_args([])
        finally:
            sys.path.pop(0)

        # Default eval set is hmgs
        assert args.eval_set == "hmgs"
        assert args.limit is None
        assert args.corpus is None
        assert args.eval_data is None
        assert args.docs_path is None
        assert not args.list_stages
        # Default stages should contain all stage keys
        from pipeline.stages import DEFAULT_STAGE_ORDER
        assert args.stages == ",".join(DEFAULT_STAGE_ORDER)

    def test_parse_custom(self):
        import importlib
        spec = importlib.util.spec_from_file_location(
            "_eval_cli2",
            _PROJECT / "scripts" / "14_eval_all_stages.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        args = mod._parse_args([
            "--stages", "base,rrf",
            "--eval-set", "kaggle",
            "--limit", "10",
        ])
        assert args.stages == "base,rrf"
        assert args.eval_set == "kaggle"
        assert args.limit == 10


class TestRunBaselineCLI:
    """Argument parsing for run_baseline.py."""

    def test_parse_defaults(self):
        import importlib
        spec = importlib.util.spec_from_file_location(
            "_baseline_cli",
            _PROJECT / "run_baseline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        args = mod._parse_args([])

        assert not args.build_index
        assert not args.eval
        assert not args.retrieval_only
        assert not args.hybrid
        assert not args.rerank
        assert not args.rrf
        assert not args.graph
        assert not args.hmgs
        assert args.corpus is None
        assert args.eval_data is None
        assert args.docs_path is None

    def test_parse_flags(self):
        import importlib
        spec = importlib.util.spec_from_file_location(
            "_baseline_cli2",
            _PROJECT / "run_baseline.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        args = mod._parse_args([
            "--build-index", "--eval", "--rerank", "--rrf", "--hmgs",
        ])
        assert args.build_index
        assert args.eval
        assert args.rerank
        assert args.rrf
        assert args.hmgs


# ---------------------------------------------------------------------------
# End-to-end dry run with fakes
# ---------------------------------------------------------------------------

@dataclass
class _FakeQA:
    query_id: str
    question: str
    answer: str
    source: str


class _FakeRetriever:
    """Minimal retriever that returns canned chunks."""

    def batch_retrieve(self, questions, top_k=10):
        return [
            [{"chunk_id": f"c_{i}_0", "text": "fake text", "source": "FakeLaw"}]
            for i in range(len(questions))
        ]

    def batch_rrf_retrieve(self, questions, bm25, top_k=10):
        return self.batch_retrieve(questions, top_k)

    def batch_hybrid_retrieve(self, questions, bm25, top_k=10):
        return self.batch_retrieve(questions, top_k)


class _FakePipeline:
    """Minimal RAG pipeline that echoes the question."""

    def __init__(self):
        self.retriever = _FakeRetriever()

    def assemble_context(self, chunks):
        ctx = " ".join(c["text"] for c in chunks)
        return ctx, chunks

    def generate(self, question, context):
        return f"Answer to: {question}"


class TestDryRun:
    """End-to-end retrieve -> evaluate flow with fake objects (no models)."""

    def test_retrieve_returns_chunks(self):
        from pipeline.retrieval import retrieve

        retriever = _FakeRetriever()
        questions = ["What is article 1?", "What is article 2?"]
        results = retrieve(retriever, questions, retrieval_mode="dense")
        assert len(results) == 2
        assert results[0][0]["chunk_id"] == "c_0_0"

    def test_retrieve_rrf(self):
        from pipeline.retrieval import retrieve

        retriever = _FakeRetriever()
        questions = ["Q1"]
        results = retrieve(
            retriever, questions,
            retrieval_mode="rrf", bm25=MagicMock(),
        )
        assert len(results) == 1

    def test_prepare_metric_input(self):
        from pipeline.evaluation import prepare_metric_input

        qa = [_FakeQA("q1", "What?", "A1", "LawA")]
        retrieved = [[
            {"chunk_id": "c1", "text": "t1", "source": "LawA"},
            {"chunk_id": "c2", "text": "t2", "source": "LawB"},
        ]]
        relevant_map = {"q1": ["c1"]}

        metric_input, full_retrieved = prepare_metric_input(
            qa, retrieved, relevant_map,
        )
        assert len(metric_input) == 1
        assert metric_input[0]["query_id"] == "q1"
        assert metric_input[0]["retrieved"] == ["c1", "c2"]
        assert metric_input[0]["source_law"] == "LawA"
        assert "q1" in full_retrieved

    def test_generation_loop(self):
        from pipeline.evaluation import run_generation_loop

        qa = [_FakeQA("q1", "What is X?", "X is Y", "LawA")]
        retrieved = [[
            {"chunk_id": "c1", "text": "X context", "source": "LawA"},
        ]]
        pipeline = _FakePipeline()

        preds = run_generation_loop(
            pipeline, qa, retrieved, stage_key="test",
        )
        assert len(preds) == 1
        assert preds[0]["query_id"] == "q1"
        assert "Answer to:" in preds[0]["predicted"]
        assert preds[0]["expected"] == "X is Y"

    def test_generation_loop_with_injection(self):
        from pipeline.evaluation import run_generation_loop

        qa = [_FakeQA("q1", "Q?", "A", "Law")]
        retrieved = [[{"chunk_id": "c1", "text": "ctx", "source": "Law"}]]
        pipeline = _FakePipeline()

        def fake_inject(answer, chunks):
            return answer + " [Kaynak 1]"

        preds = run_generation_loop(
            pipeline, qa, retrieved,
            stage_key="test", inject_citations_fn=fake_inject,
        )
        assert "[Kaynak 1]" in preds[0]["predicted"]

    def test_generation_loop_handles_error(self):
        from pipeline.evaluation import run_generation_loop

        qa = [_FakeQA("q1", "Q?", "A", "Law")]
        retrieved = [[{"chunk_id": "c1", "text": "ctx", "source": "Law"}]]

        class _FailPipeline:
            def assemble_context(self, chunks):
                raise RuntimeError("boom")

        preds = run_generation_loop(
            _FailPipeline(), qa, retrieved, stage_key="test",
        )
        assert len(preds) == 1
        assert preds[0]["predicted"] == ""

    def test_save_stage_results(self, tmp_path):
        from pipeline.evaluation import save_stage_results

        final = {"test": True}
        preds = [{"query_id": "q1", "predicted": "ans"}]
        out = save_stage_results(final, preds, tmp_path / "out")

        assert out.exists()
        with open(out) as f:
            loaded = __import__("json").load(f)
        assert loaded["test"] is True

        pred_path = tmp_path / "out" / "predictions.jsonl"
        assert pred_path.exists()

    def test_print_ablation_table_no_crash(self, capsys):
        from pipeline.evaluation import print_ablation_table

        results = {
            "base": {
                "hyperparameters": {"stage_name": "Base"},
                "source_hit_metrics": {
                    "source_hit_at_5_all": 0.5,
                    "source_hit_at_10_all": 0.6,
                    "source_mrr_all": 0.4,
                    "source_precision_at_5_all": 0.3,
                    "source_labeled_queries": 10,
                },
                "qa_metrics": {"f1": 0.5, "rouge_l": 0.4, "citation_accuracy": 0.3},
                "retrieval_metrics": {
                    "recall_at_5": 0.5, "recall_at_10": 0.6,
                    "mrr": 0.4, "ndcg_at_10": 0.5, "num_queries": 10,
                },
                "faithfulness_rate": 0.8,
                "llm_judge_score": 0.7,
                "semantic_similarity": 0.6,
                "scenario1_score": 0.5,
                "scenario2_score": 0.6,
                "scenario3_score": 0.7,
            },
        }
        print_ablation_table(results)
        captured = capsys.readouterr()
        assert "PRIMARY ABLATION TABLE" in captured.out
        assert "SECONDARY TABLE" in captured.out

    def test_full_retrieve_to_metrics_flow(self):
        """Retrieve -> prepare_metric_input -> compute metrics."""
        from pipeline.retrieval import retrieve
        from pipeline.evaluation import prepare_metric_input
        from evaluation.retrieval_metrics import compute_all_metrics

        retriever = _FakeRetriever()
        qa = [
            _FakeQA("q1", "Question 1?", "Answer 1", "Law1"),
            _FakeQA("q2", "Question 2?", "Answer 2", "Law2"),
        ]
        questions = [q.question for q in qa]
        all_retrieved = retrieve(retriever, questions, retrieval_mode="dense")

        relevant_map = {"q1": ["c_0_0"], "q2": []}
        metric_input, full_retrieved = prepare_metric_input(
            qa, all_retrieved, relevant_map,
        )
        metrics = compute_all_metrics(metric_input)

        assert "recall_at_5" in metrics
        assert "mrr" in metrics
        assert metrics["num_queries"] >= 0
