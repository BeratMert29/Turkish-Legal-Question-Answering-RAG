"""Tests for retrieval.graph_builder and retrieval.graph_index.

Uses realistic chunk_id / metadata shapes copied from the published
results/index/metadata.jsonl (both kaggle and madde_N formats).
"""

import json
from pathlib import Path

import pytest

from retrieval.graph_builder import (
    build_graph_from_metadata,
    graph_stats,
    lookup_by_source_madde,
    _extract_madde_no,
    _parse_chunk_suffix,
)
from retrieval.graph_index import GraphIndex


# ---------------------------------------------------------------------------
# Fixtures — realistic metadata matching the published metadata.jsonl shapes
# ---------------------------------------------------------------------------

# Format A: kaggle rows (source contains the first Madde in text)
_META_KAGGLE = [
    {
        "chunk_id": "Türk Ceza Kanunu_kaggle_5237_0",
        "doc_id": "kaggle_5237",
        "source": "Türk Ceza Kanunu",
        "text": "MADDE 86- (1) Kasten başkasının vücuduna acı veren veya sağlığının ya da algılama yeteneğinin bozulmasına neden olan kişi, bir yıldan üç yıla kadar hapis cezasıyla cezalandırılır.",
    },
    {
        "chunk_id": "Türk Ceza Kanunu_kaggle_5237_1",
        "doc_id": "kaggle_5237",
        "source": "Türk Ceza Kanunu",
        "text": "MADDE 87- (1) Kasten yaralama suçunun neticesi sebebiyle ağırlaşmış halleri bu maddede düzenlenmiştir.",
    },
    {
        "chunk_id": "Türk Ceza Kanunu_kaggle_5237_2",
        "doc_id": "kaggle_5237",
        "source": "Türk Ceza Kanunu",
        "text": "MADDE 88- (1) Kasten yaralama suçunun ihmali davranışla işlenmesi halinde.",
    },
]

# Format B: extra_laws madde_N doc_id format
_META_MADDE_N = [
    {
        "chunk_id": "Hukuk Muhakemeleri Kanunu_Hukuk Muhakemeleri Kanunu_madde_1_0",
        "doc_id": "Hukuk Muhakemeleri Kanunu_madde_1",
        "source": "Hukuk Muhakemeleri Kanunu",
        "text": "MADDE 1- (1) Mahkemelerin görevi, ancak kanunla düzenlenir.",
    },
    {
        "chunk_id": "Hukuk Muhakemeleri Kanunu_Hukuk Muhakemeleri Kanunu_madde_2_0",
        "doc_id": "Hukuk Muhakemeleri Kanunu_madde_2",
        "source": "Hukuk Muhakemeleri Kanunu",
        "text": "MADDE 2- (1) Asliye hukuk mahkemesi.",
    },
    # Sub-chunk of the same madde (oversized article split into two pieces)
    {
        "chunk_id": "Hukuk Muhakemeleri Kanunu_Hukuk Muhakemeleri Kanunu_madde_2_1",
        "doc_id": "Hukuk Muhakemeleri Kanunu_madde_2",
        "source": "Hukuk Muhakemeleri Kanunu",
        "text": "Devamı: asliye hukuk mahkemesinin yetki alanı.",
    },
]

# Format C: legacy _m<num> suffix (old test fixtures / backward compat)
_META_LEGACY = [
    {
        "chunk_id": "tck_5237_m12",
        "doc_id": "5237",
        "source": "Türk Ceza Kanunu",
        "text": "Madde 12 metni. Madde 13'e bakınız.",
    },
    {
        "chunk_id": "tck_5237_m13",
        "doc_id": "5237",
        "source": "Türk Ceza Kanunu",
        "text": "Madde 13 metni.",
    },
    {
        "chunk_id": "tck_5237_m14",
        "doc_id": "5237",
        "source": "Türk Ceza Kanunu",
        "text": "Madde 14 metni.",
    },
]

# Format D: madde_no field explicitly set (new CorpusChunk serialization)
_META_EXPLICIT = [
    {
        "chunk_id": "Türk Medeni Kanunu_kaggle_100_0",
        "doc_id": "kaggle_100",
        "source": "Türk Medeni Kanunu",
        "text": "MADDE 100- Mülkiyet hakkı.",
        "madde_no": "100",
    },
    {
        "chunk_id": "Türk Medeni Kanunu_kaggle_100_1",
        "doc_id": "kaggle_100",
        "source": "Türk Medeni Kanunu",
        "text": "MADDE 101- Mülkiyet devri.",
        "madde_no": "101",
    },
]

# Cross-reference: TMK chunk mentions TCK
_META_CROSS = [
    {
        "chunk_id": "Türk Medeni Kanunu_kaggle_4721_0",
        "doc_id": "kaggle_4721",
        "source": "Türk Medeni Kanunu",
        "text": "5237 sayılı Türk Ceza Kanunu Madde 86 uygulanır.",
    },
]


# ---------------------------------------------------------------------------
# _extract_madde_no tests
# ---------------------------------------------------------------------------


class TestExtractMaddeNo:
    def test_explicit_field(self):
        rec = {
            "chunk_id": "Src_kaggle_1_0",
            "doc_id": "kaggle_1",
            "source": "Src",
            "text": "",
            "madde_no": "42",
        }
        mn, si = _extract_madde_no(rec)
        assert mn == "42"
        assert si == 0

    def test_doc_id_madde_n_format(self):
        rec = {
            "chunk_id": "Devlet Memurları Kanunu_Devlet Memurları Kanunu_madde_44_0",
            "doc_id": "Devlet Memurları Kanunu_madde_44",
            "source": "Devlet Memurları Kanunu",
            "text": "MADDE 44- Devlet memurları...",
        }
        mn, si = _extract_madde_no(rec)
        assert mn == "44"
        assert si == 0

    def test_text_regular_madde(self):
        rec = {
            "chunk_id": "Türk Ceza Kanunu_kaggle_5237_0",
            "doc_id": "kaggle_5237",
            "source": "Türk Ceza Kanunu",
            "text": "MADDE 86- Kasten yaralama.",
        }
        mn, si = _extract_madde_no(rec)
        assert mn == "86"

    def test_text_ek_madde(self):
        rec = {
            "chunk_id": "Anayasa_kaggle_999_0",
            "doc_id": "kaggle_999",
            "source": "Anayasa",
            "text": "Ek Madde 3 – Ek hüküm içeriği.",
        }
        mn, si = _extract_madde_no(rec)
        assert mn == "ek-3"

    def test_text_gecici_madde(self):
        rec = {
            "chunk_id": "Anayasa_kaggle_888_0",
            "doc_id": "kaggle_888",
            "source": "Anayasa",
            "text": "Geçici Madde 2 – Geçici hüküm.",
        }
        mn, si = _extract_madde_no(rec)
        assert mn == "gecici-2"

    def test_text_gecici_madde_allcaps(self):
        rec = {
            "chunk_id": "Anayasa_kaggle_777_0",
            "doc_id": "kaggle_777",
            "source": "Anayasa",
            "text": "GEÇİCİ MADDE 4 – Geçici hüküm allcaps.",
        }
        mn, si = _extract_madde_no(rec)
        assert mn == "gecici-4"

    def test_legacy_chunk_suffix(self):
        rec = {
            "chunk_id": "tck_5237_m12",
            "doc_id": "5237",
            "source": "tck",
            "text": "no madde heading here",
        }
        mn, si = _extract_madde_no(rec)
        assert mn == "12"

    def test_no_madde_returns_none(self):
        rec = {
            "chunk_id": "Anayasa_kaggle_1_0",
            "doc_id": "kaggle_1",
            "source": "Anayasa",
            "text": "BAŞLANGIÇ — preamble without any article heading.",
        }
        mn, si = _extract_madde_no(rec)
        assert mn is None


# ---------------------------------------------------------------------------
# build_graph_from_metadata tests
# ---------------------------------------------------------------------------


class TestBuildGraph:
    def test_kaggle_format_adj_edges(self):
        """Consecutive MADDE N / N+1 in same doc_id → adj edges."""
        graph = build_graph_from_metadata(_META_KAGGLE)
        stats = graph_stats(graph)
        assert stats["total_nodes"] > 0
        assert stats["by_kind"].get("adj", 0) > 0

    def test_madde_n_format_adj_edges(self):
        """madde_N doc_id format → adj edges between madde_1 and madde_2."""
        graph = build_graph_from_metadata(_META_MADDE_N)
        stats = graph_stats(graph)
        # madde_1 and madde_2 are consecutive → expect adj
        assert stats["by_kind"].get("adj", 0) > 0

    def test_sub_chunk_adj(self):
        """Two sub-chunks of the same madde get adj edges."""
        graph = build_graph_from_metadata(_META_MADDE_N)
        cid0 = "Hukuk Muhakemeleri Kanunu_Hukuk Muhakemeleri Kanunu_madde_2_0"
        cid1 = "Hukuk Muhakemeleri Kanunu_Hukuk Muhakemeleri Kanunu_madde_2_1"
        neighbors_0 = {e[0] for e in graph.get(cid0, [])}
        assert cid1 in neighbors_0, "sub-chunks of madde_2 must be adj-connected"

    def test_explicit_madde_no_field(self):
        """Explicit madde_no field is respected for edge building."""
        graph = build_graph_from_metadata(_META_EXPLICIT)
        stats = graph_stats(graph)
        assert stats["by_kind"].get("adj", 0) > 0

    def test_legacy_suffix_fallback(self):
        """Old _m<num> chunk_id suffix still works as fallback."""
        graph = build_graph_from_metadata(_META_LEGACY)
        stats = graph_stats(graph)
        assert stats["total_nodes"] > 0
        assert stats["by_kind"].get("adj", 0) > 0

    def test_intra_reference(self):
        """A chunk mentioning 'Madde 13' links to the chunk with madde_no=13."""
        graph = build_graph_from_metadata(_META_LEGACY)
        # _META_LEGACY[0] text: "Madde 12 metni. Madde 13'e bakınız."
        # → intra edge from m12 to m13
        cid_m12 = "tck_5237_m12"
        kinds = {kind for _, kind in graph.get(cid_m12, [])}
        assert "intra" in kinds

    def test_cross_reference(self):
        """A chunk mentioning '5237 sayılı Türk Ceza Kanunu Madde 86' → cross edge."""
        combined = _META_KAGGLE + _META_CROSS
        graph = build_graph_from_metadata(combined)
        cid_cross = "Türk Medeni Kanunu_kaggle_4721_0"
        kinds = {kind for _, kind in graph.get(cid_cross, [])}
        assert "cross" in kinds

    def test_source_madde_lookup_present(self):
        """Graph includes _source_madde_lookup key."""
        graph = build_graph_from_metadata(_META_KAGGLE)
        assert "_source_madde_lookup" in graph

    def test_lookup_by_source_madde(self):
        """lookup_by_source_madde returns correct chunk_ids."""
        graph = build_graph_from_metadata(_META_KAGGLE)
        results = lookup_by_source_madde(graph, "Türk Ceza Kanunu", "86")
        assert "Türk Ceza Kanunu_kaggle_5237_0" in results

    def test_no_self_loops(self):
        """No chunk should have an edge pointing to itself."""
        all_meta = _META_KAGGLE + _META_MADDE_N + _META_LEGACY + _META_EXPLICIT
        graph = build_graph_from_metadata(all_meta)
        for cid, edges in graph.items():
            if cid.startswith("_"):
                continue
            for nb, _ in edges:
                assert nb != cid, f"Self-loop detected on {cid}"


# ---------------------------------------------------------------------------
# Real metadata smoke test (skipped when results/ is absent)
# ---------------------------------------------------------------------------

_RESULTS_META = (
    Path(__file__).resolve().parent.parent.parent  # worktree root
    / "results" / "index" / "metadata.jsonl"
)


@pytest.mark.skipif(not _RESULTS_META.exists(), reason="results/index/metadata.jsonl not available")
class TestRealMetadata:
    @pytest.fixture(scope="class")
    @classmethod
    def real_graph(cls):
        metadata = []
        with _RESULTS_META.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    metadata.append(json.loads(line))
        return build_graph_from_metadata(metadata)

    def test_nonzero_adj_edges(self, real_graph):
        stats = graph_stats(real_graph)
        assert stats["by_kind"].get("adj", 0) > 0, (
            f"Expected >0 adj edges on real metadata; got stats={stats}"
        )

    def test_nonzero_intra_edges(self, real_graph):
        stats = graph_stats(real_graph)
        assert stats["by_kind"].get("intra", 0) > 0, (
            f"Expected >0 intra edges on real metadata; got stats={stats}"
        )

    def test_nonzero_cross_edges(self, real_graph):
        """Cross edges require law-to-law references in corpus text.

        The published metadata contains cross-law references using Turkish
        inflected forms (e.g. "5237 sayılı Türk Ceza Kanununda") which are
        now matched by the updated _CROSS_LAW_RE (no \\b, allows case suffixes).
        """
        stats = graph_stats(real_graph)
        assert stats["by_kind"].get("cross", 0) > 0, (
            f"Expected >0 cross edges on real metadata; got stats={stats}"
        )

    def test_stats_report(self, real_graph):
        """Print stats for visibility in CI output."""
        stats = graph_stats(real_graph)
        print(f"\nAFTER graph_stats (real metadata): {stats}")
        assert stats["total_nodes"] > 100


# ---------------------------------------------------------------------------
# GraphIndex integration tests
# ---------------------------------------------------------------------------


def _make_graph_index(tmp_path: Path) -> GraphIndex:
    meta = _META_KAGGLE[:2]
    graph = build_graph_from_metadata(meta)
    g_path = tmp_path / "graph.json"
    m_path = tmp_path / "metadata.jsonl"
    g_path.write_text(json.dumps(graph), encoding="utf-8")
    m_path.write_text(
        "\n".join(json.dumps(r) for r in meta),
        encoding="utf-8",
    )
    return GraphIndex(g_path, m_path)


class TestGraphIndex:
    def test_expand_returns_neighbors(self, tmp_path):
        gi = _make_graph_index(tmp_path)
        seed = [
            {
                "chunk_id": "Türk Ceza Kanunu_kaggle_5237_0",
                "text": "...",
                "doc_id": "kaggle_5237",
                "source": "Türk Ceza Kanunu",
                "score": 1.0,
            }
        ]
        expanded = gi.expand(seed, hops=1, budget=5)
        assert len(expanded) >= len(seed)

    def test_expand_no_duplicates(self, tmp_path):
        gi = _make_graph_index(tmp_path)
        seed = [
            {
                "chunk_id": "Türk Ceza Kanunu_kaggle_5237_0",
                "text": "...",
                "doc_id": "kaggle_5237",
                "source": "Türk Ceza Kanunu",
                "score": 1.0,
            }
        ]
        expanded = gi.expand(seed, hops=1, budget=10)
        ids = [c["chunk_id"] for c in expanded]
        assert len(ids) == len(set(ids)), "Duplicates found in expanded results"
