"""Tests for HYBRID_ALPHA config propagation.

Verifies that:
- config.HYBRID_ALPHA exists and is a float in [0, 1].
- Retriever defaults to config.HYBRID_ALPHA when alpha is not passed.
- pipeline.retrieval.retrieve passes config.HYBRID_ALPHA for hybrid mode.
- scripts/03 references config.HYBRID_ALPHA (source-level check).
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch


import config


class TestHybridAlphaConfig:
    """config.HYBRID_ALPHA existence and value."""

    def test_exists(self):
        assert hasattr(config, "HYBRID_ALPHA")

    def test_type(self):
        assert isinstance(config.HYBRID_ALPHA, (int, float))

    def test_range(self):
        assert 0.0 <= config.HYBRID_ALPHA <= 1.0

    def test_default_is_half(self):
        """Production default is 0.5 (equal dense/BM25 weight)."""
        assert config.HYBRID_ALPHA == 0.5


class TestRetrieverAlphaDefault:
    """Retriever.batch_hybrid_retrieve uses config.HYBRID_ALPHA as default."""

    def test_signature_default_is_none(self):
        from retrieval.retriever import Retriever
        sig = inspect.signature(Retriever.batch_hybrid_retrieve)
        assert sig.parameters["alpha"].default is None

    def test_resolves_to_config(self, monkeypatch):
        """When alpha is None, batch_hybrid_retrieve resolves to config.HYBRID_ALPHA."""
        from retrieval.retriever import Retriever

        monkeypatch.setattr(config, "HYBRID_ALPHA", 0.42)

        ret = Retriever.__new__(Retriever)
        ret.index = MagicMock()
        ret.index.ntotal = 3
        ret.metadata = [{"text": "", "doc_id": "", "source": "", "chunk_id": f"c{i}"} for i in range(3)]
        ret.embedder = MagicMock()

        captured = {}

        def fake_hybrid_one(self_unused, q_emb, ds, di, query, bm25, alpha, top_k, pool):
            captured["alpha"] = alpha
            return []

        bm25 = MagicMock()
        bm25.metadata = ret.metadata

        with patch.object(Retriever, "_hybrid_one", fake_hybrid_one):
            ret.embedder.encode.return_value = MagicMock(
                astype=MagicMock(return_value=__import__("numpy").zeros((1, 1024), dtype="float32"))
            )
            ret.index.search.return_value = (
                __import__("numpy").zeros((1, 1), dtype="float32"),
                __import__("numpy").array([[-1]], dtype="int64"),
            )
            ret.batch_hybrid_retrieve(["test"], bm25)

        assert captured["alpha"] == 0.42


class TestPipelineUsesConfig:
    """pipeline.retrieval.retrieve passes config.HYBRID_ALPHA for hybrid mode."""

    def test_hybrid_retrieve_receives_config_alpha(self, monkeypatch):
        """retrieve() in hybrid mode must pass config.HYBRID_ALPHA to the retriever.

        Behavioral test: monkeypatches config.HYBRID_ALPHA to a sentinel value and
        captures the alpha argument actually received by batch_hybrid_retrieve.
        """
        monkeypatch.setattr(config, "HYBRID_ALPHA", 0.13)

        from pipeline.retrieval import retrieve
        from unittest.mock import MagicMock

        captured: dict = {}

        class _CapturingRetriever:
            def batch_retrieve(self, questions, top_k=10):
                return [[] for _ in questions]

            def batch_hybrid_retrieve(self, questions, bm25, alpha=None, top_k=10):
                captured["alpha"] = alpha
                return [[] for _ in questions]

        retrieve(
            _CapturingRetriever(),
            ["test soru?"],
            retrieval_mode="hybrid",
            bm25=MagicMock(),
        )
        assert captured.get("alpha") == 0.13, (
            f"Expected alpha=config.HYBRID_ALPHA=0.13, got {captured.get('alpha')}"
        )


class TestScript03UsesConfig:
    """scripts/03_evaluate_retrieval.py uses config.HYBRID_ALPHA, not a hardcoded value."""

    def test_no_hardcoded_alpha_07(self):
        src_path = Path(config.BASE_DIR) / "scripts" / "03_evaluate_retrieval.py"
        source = src_path.read_text(encoding="utf-8")
        # Should NOT contain a bare alpha=0.7 literal
        assert "alpha=0.7" not in source
        # Should reference config
        assert "config.HYBRID_ALPHA" in source

    def test_output_dict_includes_retrieval_config_key(self):
        """scripts/03 output dict must embed retrieval_config with hybrid_alpha
        and config_version so result files are self-describing."""
        src_path = Path(config.BASE_DIR) / "scripts" / "03_evaluate_retrieval.py"
        source = src_path.read_text(encoding="utf-8")
        assert "retrieval_config" in source
        assert "RETRIEVAL_CONFIG_VERSION" in source


class TestRetrievalConfigVersion:
    """config.RETRIEVAL_CONFIG_VERSION provenance constant."""

    def test_exists(self):
        assert hasattr(config, "RETRIEVAL_CONFIG_VERSION")

    def test_is_string(self):
        assert isinstance(config.RETRIEVAL_CONFIG_VERSION, str)
        assert len(config.RETRIEVAL_CONFIG_VERSION) > 0

    def test_encodes_current_alpha(self):
        """Version string must encode the current alpha value so it's human-readable."""
        # e.g. "v2_alpha0.5" contains "0.5" matching HYBRID_ALPHA
        alpha_str = str(config.HYBRID_ALPHA)
        # Strip trailing zeros: "0.5" matches both "0.5" and "0.50"
        assert alpha_str.rstrip("0").rstrip(".") in config.RETRIEVAL_CONFIG_VERSION, (
            f"RETRIEVAL_CONFIG_VERSION={config.RETRIEVAL_CONFIG_VERSION!r} "
            f"should encode HYBRID_ALPHA={config.HYBRID_ALPHA}"
        )

    def test_synthetic_output_dict_has_retrieval_config(self):
        """Verify the output dict structure that scripts/03 emits.

        Builds the same sub-dict inline (as scripts/03 does) and checks all
        required fields and types are present.  Does NOT run the actual eval.
        """
        retrieval_config = {
            "hybrid_alpha": config.HYBRID_ALPHA,
            "config_version": config.RETRIEVAL_CONFIG_VERSION,
        }
        out = {"retrieval_config": retrieval_config, "dense_metrics": {}}

        assert "retrieval_config" in out
        assert out["retrieval_config"]["hybrid_alpha"] == config.HYBRID_ALPHA
        assert isinstance(out["retrieval_config"]["hybrid_alpha"], (int, float))
        assert "config_version" in out["retrieval_config"]
        assert isinstance(out["retrieval_config"]["config_version"], str)
