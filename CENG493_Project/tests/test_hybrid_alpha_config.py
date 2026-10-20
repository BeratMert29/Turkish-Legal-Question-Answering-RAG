"""Tests for HYBRID_ALPHA config propagation.

Verifies that:
- config.HYBRID_ALPHA exists and is a float in [0, 1].
- Retriever defaults to config.HYBRID_ALPHA when alpha is not passed.
- pipeline.retrieval.retrieve passes config.HYBRID_ALPHA for hybrid mode.
- scripts/03 references config.HYBRID_ALPHA (source-level check).
"""

import ast
import inspect
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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

    def test_source_references_config_hybrid_alpha(self):
        """The pipeline module source contains 'config.HYBRID_ALPHA'."""
        src_path = Path(config.BASE_DIR) / "pipeline" / "retrieval.py"
        source = src_path.read_text(encoding="utf-8")
        assert "config.HYBRID_ALPHA" in source


class TestScript03UsesConfig:
    """scripts/03_evaluate_retrieval.py uses config.HYBRID_ALPHA, not a hardcoded value."""

    def test_no_hardcoded_alpha_07(self):
        src_path = Path(config.BASE_DIR) / "scripts" / "03_evaluate_retrieval.py"
        source = src_path.read_text(encoding="utf-8")
        # Should NOT contain a bare alpha=0.7 literal
        assert "alpha=0.7" not in source
        # Should reference config
        assert "config.HYBRID_ALPHA" in source
