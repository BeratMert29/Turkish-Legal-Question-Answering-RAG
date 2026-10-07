"""
test_pipeline.py -- tests for the shared pipeline package.

Covers:
  - Package imports (light env, no heavy deps)
  - StageConfig, STAGE_REGISTRY, DEFAULT_STAGE_ORDER consistency
  - CLI arg parsing for both scripts (defaults, --help)
  - End-to-end dry run of retrieve -> evaluate with fake objects
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock


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
        from pipeline.stages import STAGE_REGISTRY, DEFAULT_STAGE_ORDER
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

        # Default eval set is turkish_legal_rag (hmgs stays selectable)
        assert args.eval_set == "turkish_legal_rag"
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


# ---------------------------------------------------------------------------
# New helper / fix tests
# ---------------------------------------------------------------------------

class TestEvalHelpers:
    """Unit tests for helpers added in the improvements/2026-10 branch."""

    def test_evict_model_cache_clears(self):
        """evict_model_cache() must empty _model_cache without raising."""
        from pipeline.evaluation import _model_cache, evict_model_cache

        _model_cache["nli"] = object()
        evict_model_cache()
        assert "nli" not in _model_cache

    def test_build_stage_components_importable(self):
        from pipeline.evaluation import _build_stage_components
        assert callable(_build_stage_components)

    def test_run_generation_and_qa_importable(self):
        from pipeline.evaluation import _run_generation_and_qa
        assert callable(_run_generation_and_qa)

    def test_run_stage_phase_helpers_importable(self):
        """All phase helpers extracted from run_stage must be importable."""
        from pipeline.evaluation import (
            _run_retrieval_phase,
            _run_supplemental_metrics,
            _run_hallucination_phase,
            _run_judge_phase,
            _run_semantic_sim_phase,
            _assemble_final_result,
        )
        for fn in (
            _run_retrieval_phase, _run_supplemental_metrics,
            _run_hallucination_phase, _run_judge_phase,
            _run_semantic_sim_phase, _assemble_final_result,
        ):
            assert callable(fn)

    def test_run_stage_slimmed(self):
        """run_stage body must be shorter than the old 295-line monolith."""
        import inspect
        from pipeline.evaluation import run_stage

        src = inspect.getsource(run_stage)
        lines = [l for l in src.splitlines() if l.strip()]
        assert len(lines) < 90, (
            f"run_stage is {len(lines)} non-blank lines; expected < 90 after extraction"
        )

    def test_hallucination_phase_uses_model_cache(self):
        """_run_hallucination_phase must store nli_model in _model_cache['nli']."""
        from unittest.mock import patch, MagicMock
        from pipeline import evaluation as _eval

        fake_hall = {"summary": {"context_grounding_rate": 0.9}}
        fake_nli = MagicMock()

        with patch.object(_eval, "evict_model_cache") as mock_evict, \
             patch.object(_eval, "run_hallucination_eval",
                          return_value=(fake_hall, 0.9, fake_nli)):
            _eval._model_cache.clear()
            hall, rate = _eval._run_hallucination_phase([], "mock-model")

        mock_evict.assert_called_once()
        assert _eval._model_cache.get("nli") is fake_nli
        assert rate == 0.9
        assert hall is fake_hall

    def test_hallucination_phase_scores_generation_context(self):
        """NLI premises are each prediction's own retrieved_chunks (the
        generator's context), and a cached NLI model survives eviction."""
        from unittest.mock import patch, MagicMock
        from pipeline import evaluation as _eval

        cached = MagicMock()
        preds = [{"query_id": "q1", "predicted": "a",
                  "retrieved_chunks": [{"chunk_id": "ctx", "text": "t"}]}]
        with patch.object(_eval, "run_hallucination_eval",
                          return_value=({"summary": {}}, 0.5, cached)) as mock_rhe:
            _eval._model_cache.clear()
            _eval._model_cache["nli"] = cached
            _eval._run_hallucination_phase(preds, "m")
        args, kwargs = mock_rhe.call_args
        assert args[1] == {"q1": [{"chunk_id": "ctx", "text": "t"}]}
        assert kwargs["nli_model"] is cached

    def test_no_run_stage_nli_attribute(self):
        """run_stage must not cache NLI on a function attribute."""
        from pipeline.evaluation import run_stage
        assert not hasattr(run_stage, "_nli_model"), (
            "run_stage._nli_model found; NLI must go through _model_cache"
        )

    def test_save_stage_results_atomic(self, tmp_path):
        """save_stage_results must produce an intact JSON even if called twice."""
        from pipeline.evaluation import save_stage_results

        out = save_stage_results({"v": 1}, [], tmp_path / "r")
        assert out.exists()
        import json
        assert json.loads(out.read_text())["v"] == 1
        # Second call overwrites atomically — must not raise or leave .tmp files.
        save_stage_results({"v": 2}, [], tmp_path / "r")
        assert json.loads(out.read_text())["v"] == 2
        assert not (tmp_path / "r" / "baseline_metrics.tmp").exists()

    def test_run_llm_judge_eval_shared_sample(self):
        """run_llm_judge_eval must call each judge function with the same query IDs."""
        from pipeline.evaluation import run_llm_judge_eval

        calls: dict[str, list] = {}

        def _fake_judge(preds, base_url, model, sample_size=20, results_dir=None,
                        query_ids=None):
            calls[model + str(len(calls))] = [p["query_id"] for p in preds]
            return {"score": 0.5, "per_sample": [], "parse_fail_count": 0, "sample_size": len(preds)}

        qa = [_FakeQA(f"q{i}", f"Q{i}", f"A{i}", "Law") for i in range(30)]
        preds = [
            {
                "query_id": qa[i].query_id,
                "predicted": f"ans{i}",
                "expected": qa[i].answer,
                "retrieved_chunks": [],
            }
            for i in range(30)
        ]

        from unittest.mock import patch
        with patch("evaluation.llm_judge.llm_judge_answer", side_effect=_fake_judge), \
             patch("evaluation.llm_judge.llm_judge_faithfulness", side_effect=_fake_judge), \
             patch("evaluation.llm_judge.llm_judge_relevancy", side_effect=_fake_judge), \
             patch("evaluation.llm_judge.llm_judge_coherence", side_effect=_fake_judge):
            run_llm_judge_eval(
                preds, qa,
                base_url="http://localhost:11434/v1",
                judge_model="mock",
                sample_size=10,
            )

        id_sets = [set(v) for v in calls.values()]
        assert len(id_sets) == 4
        # All four metrics must have received the same set of query IDs.
        assert id_sets[0] == id_sets[1] == id_sets[2] == id_sets[3]

    def test_run_judge_phase_samples_once_via_sample_judge_query_ids(self):
        """_run_judge_phase must call sample_judge_query_ids once and pass
        the returned IDs as query_ids= to run_llm_judge_eval."""
        from pipeline import evaluation as _eval
        from unittest.mock import patch, MagicMock

        sampled = [f"q{i}" for i in range(20)]
        fake_stage = MagicMock()
        fake_stage.results_dir = None

        preds = [
            {"query_id": f"q{i}", "predicted": f"a{i}",
             "expected": f"e{i}", "retrieved_chunks": []}
            for i in range(50)
        ]
        qa = [_FakeQA(f"q{i}", f"Q{i}", f"A{i}", "Law") for i in range(50)]

        with patch("evaluation.llm_judge.sample_judge_query_ids",
                   return_value=sampled) as mock_sji, \
             patch.object(_eval, "run_llm_judge_eval",
                          return_value={
                              "score": 0.5, "faithfulness": 0.5,
                              "relevancy": 0.5, "coherence": 0.5,
                              "parse_failures": {}, "failure_count": 0,
                              "call_count": len(sampled),
                          }) as mock_rje:
            _eval._run_judge_phase(preds, qa, fake_stage, "test", 0.2)

        # sample_judge_query_ids called exactly once
        mock_sji.assert_called_once()
        # run_llm_judge_eval called with the sampled IDs
        _, kwargs = mock_rje.call_args
        assert kwargs.get("query_ids") == sampled

    def test_retrieval_pipeline_file_no_leak(self, tmp_path):
        """auto_build_graph must not leak a file handle (use with)."""
        # Smoke test: call with a non-existent candidate so it exits early.
        import config as _cfg
        from unittest.mock import patch
        from pipeline.retrieval import auto_build_graph

        fake_path = tmp_path / "graph.json"
        # No metadata candidates exist → function returns without opening files.
        with patch.object(_cfg, "INDEX_DIR", tmp_path), \
             patch.object(_cfg, "BASE_DIR", tmp_path):
            auto_build_graph(fake_path)
        assert not fake_path.exists()

    def test_real_metadata_path_resolves(self, repo_root):
        """results/index/metadata.jsonl is committed and resolves from repo root."""
        meta = repo_root / "results" / "index" / "metadata.jsonl"
        assert meta.exists(), (
            f"metadata.jsonl not committed at {meta}; "
            "CI will silently skip data-dependent tests without it."
        )

    def test_run_hallucination_eval_annotation_returns_three(self):
        """run_hallucination_eval type annotation must match its 3-element return."""
        import inspect
        from pipeline.evaluation import run_hallucination_eval

        hints = inspect.get_annotations(run_hallucination_eval, eval_str=False)
        ret = hints.get("return")
        # The annotation must be tuple[dict, float, Any], not the old tuple[dict, float].
        # We check the string representation is not the old 2-arg form.
        ret_str = str(ret)
        assert "Any" in ret_str, (
            f"run_hallucination_eval return annotation missing 'Any' (nli_model): {ret_str}"
        )

    def test_run_llm_judge_eval_uses_config_default(self):
        """run_llm_judge_eval must fall back to config.LLM_JUDGE_SAMPLE_SIZE when
        sample_size is not supplied."""
        import config
        from pipeline.evaluation import run_llm_judge_eval
        from unittest.mock import patch

        captured: list[int] = []

        def _fake_judge(preds, base_url, model, sample_size=20, results_dir=None,
                        query_ids=None):
            captured.append(len(preds))
            return {"score": 0.5, "per_sample": [], "parse_fail_count": 0, "sample_size": len(preds)}

        qa = [_FakeQA(f"q{i}", f"Q{i}", f"A{i}", "Law") for i in range(50)]
        preds = [
            {"query_id": qa[i].query_id, "predicted": f"ans{i}",
             "expected": qa[i].answer, "retrieved_chunks": []}
            for i in range(50)
        ]

        with patch("evaluation.llm_judge.llm_judge_answer", side_effect=_fake_judge), \
             patch("evaluation.llm_judge.llm_judge_faithfulness", side_effect=_fake_judge), \
             patch("evaluation.llm_judge.llm_judge_relevancy", side_effect=_fake_judge), \
             patch("evaluation.llm_judge.llm_judge_coherence", side_effect=_fake_judge):
            # Call WITHOUT sample_size so the config default is used.
            run_llm_judge_eval(
                preds, qa,
                base_url="http://localhost:11434/v1",
                judge_model="mock",
            )

        expected_n = (
            50 if config.LLM_JUDGE_SAMPLE_SIZE is None
            else min(config.LLM_JUDGE_SAMPLE_SIZE, 50)
        )
        for n in captured:
            assert n == expected_n, (
                f"judge called with {n} samples; expected {expected_n} "
                f"(config.LLM_JUDGE_SAMPLE_SIZE={config.LLM_JUDGE_SAMPLE_SIZE})"
            )


class TestGraphGenerationContext:
    """Graph neighbours must reach the generation context; metrics ignore them."""

    @staticmethod
    def _chunks():
        reg = [{"chunk_id": f"r{i}", "text": "t", "source": "S", "score": 1.0 - i / 10}
               for i in range(6)]
        nb = [
            {"chunk_id": "n0", "text": "t", "source": "S", "score": .5,
             "graph_neighbor": True, "graph_parent": "r0", "graph_root": "r0"},
            {"chunk_id": "n1", "text": "t", "source": "S", "score": .4,
             "graph_neighbor": True, "graph_parent": "r1", "graph_root": "r1"},
        ]
        # neighbours spliced right after their parents
        return [reg[0], nb[0], reg[1], nb[1], *reg[2:]]

    def _pipe(self, budget):
        from generation.rag_pipeline import RAGPipeline
        p = object.__new__(RAGPipeline)
        p.top_k_for_generation = 5
        p.graph_neighbor_budget = budget
        return p

    def test_reserved_slots_include_neighbours(self):
        sel = self._pipe(2)._select_for_generation(self._chunks())
        ids = [c["chunk_id"] for c in sel]
        assert len(ids) == 5
        assert ids == ["r0", "n0", "r1", "n1", "r2"]

    def test_neighbours_of_unkept_parents_do_not_displace_top_chunks(self):
        reg = [{"chunk_id": f"r{i}", "text": "t", "source": "S", "score": 1.0 - i / 10}
               for i in range(10)]
        nb = [{"chunk_id": cid, "text": "t", "source": "S", "score": .1,
               "graph_neighbor": True, "graph_parent": root, "graph_root": root}
              for cid, root in (("n7a", "r7"), ("n7b", "r7"), ("n8a", "r8"))]
        chunks = [*reg[:8], nb[0], nb[1], reg[8], nb[2], reg[9]]
        sel = self._pipe(3)._select_for_generation(chunks)
        assert [c["chunk_id"] for c in sel] == ["r0", "r1", "r2", "r3", "r4"]

    def test_reservation_shrinks_to_available_neighbours(self):
        # only r0 has a neighbour: one slot is reserved, not the full budget of 3
        sel = self._pipe(3)._select_for_generation(
            [c for c in self._chunks() if c["chunk_id"] != "n1"])
        assert [c["chunk_id"] for c in sel] == ["r0", "n0", "r1", "r2", "r3"]

    def test_budget_zero_just_cuts_to_top_k(self):
        sel = self._pipe(0)._select_for_generation(self._chunks())
        assert [c["chunk_id"] for c in sel] == ["r0", "n0", "r1", "n1", "r2"]

    def test_metric_input_ignores_neighbours(self):
        from pipeline.evaluation import prepare_metric_input
        qa = [_FakeQA("q1", "Q", "A", "S")]
        mi, full = prepare_metric_input(qa, [self._chunks()], {})
        assert mi[0]["retrieved"] == [f"r{i}" for i in range(6)]
        assert len(full["q1"]) == 8

    def test_expand_puts_neighbour_after_parent(self):
        from retrieval.graph_index import GraphIndex
        gi = object.__new__(GraphIndex)
        gi._graph = {"r0": [("n0", "adj")]}
        gi._chunk_meta = {"n0": {"text": "x", "doc_id": "d", "source": "S"}}
        gi._source_madde_lookup = {}
        chunks = [{"chunk_id": "r0", "score": 1.0}, {"chunk_id": "r1", "score": .9}]
        out = gi.expand(chunks, budget=1, kinds=("adj",))
        assert [c["chunk_id"] for c in out] == ["r0", "n0", "r1"]
        assert out[1]["graph_neighbor"] and out[1]["graph_parent"] == "r0"


def test_failed_generations_stay_in_qa_denominator():
    from unittest.mock import patch, MagicMock
    from pipeline import evaluation as ev
    preds = [
        {"query_id": "a", "predicted": "ok", "expected": "ok", "retrieved_chunks": []},
        {"query_id": "b", "predicted": "", "expected": "x", "retrieved_chunks": [],
         "generation_error": True},
    ]
    seen = {}

    def _fake_qa(p):
        seen["n"] = len(p)
        return {"f1": 0.5}

    stage = MagicMock(llm="base", use_graph=False)
    with patch.object(ev, "run_generation_loop", return_value=preds), \
         patch("generation.rag_pipeline.RAGPipeline", MagicMock()), \
         patch("evaluation.qa_metrics.compute_all_qa_metrics_with_citation",
               side_effect=_fake_qa):
        out = ev._run_generation_and_qa(
            "base", stage, [], [], None, "m", False, None, 0.9)
    ok_preds, n_total, n_failed, _, _, qa, all_preds = out
    assert seen["n"] == 2 and n_total == 2 and n_failed == 1
    assert len(ok_preds) == 1 and len(all_preds) == 2
    assert qa["n_generation_failed_scored_zero"] == 1


def test_per_query_records_and_ci():
    from pipeline.evaluation import build_per_query, compute_confidence_intervals
    preds = [
        {"query_id": "a", "predicted": "ok cevap", "expected": "ok cevap"},
        {"query_id": "b", "predicted": "", "expected": "x", "generation_error": True},
    ]
    mi = [
        {"query_id": "a", "relevant": ["c1"], "retrieved": ["c1", "c2"],
         "source_law": "TCK", "retrieved_sources": ["TCK"]},
        {"query_id": "b", "relevant": [], "retrieved": ["c9"],
         "source_law": "", "retrieved_sources": ["X"]},
    ]
    rows = build_per_query(preds, mi, {"per_sample": []},
                           [{"query_id": "a", "similarity": 0.9}], {"per_sample": {}})
    assert rows[0]["recall_at_5"] == 1.0 and rows[0]["semantic_similarity"] == 0.9
    assert rows[1]["generation_failed"] and rows[1]["f1"] == 0.0
    assert rows[1]["recall_at_5"] is None
    ci = compute_confidence_intervals(rows)
    assert ci["f1"]["n"] == 2 and "recall_at_5" in ci
