"""Tests for GRAPH_EXPANSION_KINDS config propagation.

Verifies that:
- config.GRAPH_EXPANSION_KINDS exists and is a tuple of strings.
- Default value is ("adj",) (safe, adjacency-only).
- pipeline/retrieval.py references config.GRAPH_EXPANSION_KINDS in source.
"""

from pathlib import Path

import config


class TestGraphExpansionKindsConfig:
    """config.GRAPH_EXPANSION_KINDS existence and value."""

    def test_exists(self):
        assert hasattr(config, "GRAPH_EXPANSION_KINDS")

    def test_type(self):
        assert isinstance(config.GRAPH_EXPANSION_KINDS, tuple)
        assert all(isinstance(k, str) for k in config.GRAPH_EXPANSION_KINDS)

    def test_default_is_adj_only(self):
        """Default is adjacency-only; intra/cross require eval validation."""
        assert config.GRAPH_EXPANSION_KINDS == ("adj",)


class TestPipelineUsesGraphKindsConfig:
    """pipeline/retrieval.py uses config.GRAPH_EXPANSION_KINDS, not a hardcoded tuple."""

    def test_source_references_config(self):
        src_path = Path(config.BASE_DIR) / "pipeline" / "retrieval.py"
        source = src_path.read_text(encoding="utf-8")
        assert "config.GRAPH_EXPANSION_KINDS" in source

    def test_no_hardcoded_adj_only_tuple(self):
        """The pipeline source should not contain a hardcoded kinds=("adj",)."""
        src_path = Path(config.BASE_DIR) / "pipeline" / "retrieval.py"
        source = src_path.read_text(encoding="utf-8")
        assert 'kinds=("adj",)' not in source
