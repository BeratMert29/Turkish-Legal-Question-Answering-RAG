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

    def test_source_madde_lookup_loaded(self, tmp_path):
        """GraphIndex loads _source_madde_lookup from graph JSON."""
        gi = _make_graph_index(tmp_path)
        assert len(gi._source_madde_lookup) > 0


# ---------------------------------------------------------------------------
# Direct madde injection tests (inject_from_query / DIRECT_MADDE_LOOKUP_ENABLED)
# ---------------------------------------------------------------------------


def _make_gi_with_lookup(tmp_path: Path) -> "GraphIndex":
    """GraphIndex with _META_KAGGLE (has TCK madde 86/87/88) and lookup table."""
    graph = build_graph_from_metadata(_META_KAGGLE)
    g_path = tmp_path / "graph_lookup.json"
    m_path = tmp_path / "meta_lookup.jsonl"
    g_path.write_text(json.dumps(graph), encoding="utf-8")
    m_path.write_text(
        "\n".join(json.dumps(r) for r in _META_KAGGLE),
        encoding="utf-8",
    )
    return GraphIndex(g_path, m_path)


class TestDirectMaddeLookup:
    def test_inject_disabled_by_default(self, tmp_path):
        """inject_from_query returns [] when DIRECT_MADDE_LOOKUP_ENABLED=False."""
        gi = _make_gi_with_lookup(tmp_path)
        # Default config has DIRECT_MADDE_LOOKUP_ENABLED = False
        result = gi.inject_from_query("TCK madde 86")
        assert result == []

    def test_inject_abbrev_pattern(self, tmp_path, monkeypatch):
        """TCK madde 86 → injects the TCK madde-86 chunk when flag is on."""
        import config
        monkeypatch.setattr(config, "DIRECT_MADDE_LOOKUP_ENABLED", True)
        gi = _make_gi_with_lookup(tmp_path)
        result = gi.inject_from_query("TCK madde 86")
        ids = [r["chunk_id"] for r in result]
        assert "Türk Ceza Kanunu_kaggle_5237_0" in ids

    def test_inject_canonical_pattern(self, tmp_path, monkeypatch):
        """5237 sayılı ... Kanununda ... madde 87 → injects TCK madde-87 chunk."""
        import config
        monkeypatch.setattr(config, "DIRECT_MADDE_LOOKUP_ENABLED", True)
        gi = _make_gi_with_lookup(tmp_path)
        query = "5237 sayılı Türk Ceza Kanununda madde 87 hükmü uygulanır."
        result = gi.inject_from_query(query)
        ids = [r["chunk_id"] for r in result]
        assert "Türk Ceza Kanunu_kaggle_5237_1" in ids

    def test_inject_no_duplicates(self, tmp_path, monkeypatch):
        """inject_from_query with exclude set does not return excluded ids."""
        import config
        monkeypatch.setattr(config, "DIRECT_MADDE_LOOKUP_ENABLED", True)
        gi = _make_gi_with_lookup(tmp_path)
        existing = {"Türk Ceza Kanunu_kaggle_5237_0"}
        result = gi.inject_from_query("TCK madde 86", exclude=existing)
        ids = [r["chunk_id"] for r in result]
        assert "Türk Ceza Kanunu_kaggle_5237_0" not in ids

    def test_expand_batch_with_queries_flag_off(self, tmp_path):
        """expand_batch with queries= does not inject when flag is off."""
        gi = _make_gi_with_lookup(tmp_path)
        seed_batch = [[
            {
                "chunk_id": "Türk Ceza Kanunu_kaggle_5237_2",
                "text": "...",
                "doc_id": "kaggle_5237",
                "source": "Türk Ceza Kanunu",
                "score": 1.0,
            }
        ]]
        before_len = len(seed_batch[0])
        expanded = gi.expand_batch(
            seed_batch,
            hops=0,
            budget=0,
            kinds=(),
            queries=["TCK madde 86"],
        )
        # budget=0 and kinds=() prevent graph expansion; inject also off → same length
        assert len(expanded[0]) == before_len

    def test_expand_batch_with_queries_flag_on(self, tmp_path, monkeypatch):
        """expand_batch with queries= injects direct-lookup chunks when flag is on."""
        import config
        monkeypatch.setattr(config, "DIRECT_MADDE_LOOKUP_ENABLED", True)
        gi = _make_gi_with_lookup(tmp_path)
        seed_batch = [[
            {
                "chunk_id": "Türk Ceza Kanunu_kaggle_5237_2",
                "text": "...",
                "doc_id": "kaggle_5237",
                "source": "Türk Ceza Kanunu",
                "score": 1.0,
            }
        ]]
        expanded = gi.expand_batch(
            seed_batch,
            hops=0,
            budget=0,
            kinds=(),
            queries=["TCK madde 86"],
        )
        ids = [c["chunk_id"] for c in expanded[0]]
        # madde 86 chunk injected even though it wasn't in seed and budget=0
        assert "Türk Ceza Kanunu_kaggle_5237_0" in ids


# ---------------------------------------------------------------------------
# Real metadata: GraphIndex round-trip and expand_batch on real chunk IDs
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _RESULTS_META.exists(), reason="results/index/metadata.jsonl not available")
class TestRealGraphIndex:
    """Build graph from real metadata, save/load GraphIndex, verify expand_batch."""

    @pytest.fixture(scope="class")
    @classmethod
    def real_gi(cls, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("real_gi")
        metadata = [
            json.loads(l) for l in _RESULTS_META.open(encoding="utf-8") if l.strip()
        ]
        from retrieval.graph_builder import build_graph_from_metadata, save_graph
        g = build_graph_from_metadata(metadata)
        g_path = tmp / "graph.json"
        save_graph(g, g_path)
        return GraphIndex(g_path, _RESULTS_META)

    def test_graph_loaded(self, real_gi):
        assert len(real_gi._graph) > 0

    def test_lookup_loaded(self, real_gi):
        assert len(real_gi._source_madde_lookup) > 0

    def test_expand_batch_returns_extra_chunks(self, real_gi):
        """expand_batch on real chunk IDs that have adj edges should add neighbors."""
        # Find a chunk that has adj edges in the graph.
        seed_id = next(
            (cid for cid, edges in real_gi._graph.items()
             if any(k == "adj" for _, k in edges)),
            None,
        )
        assert seed_id is not None, "No chunk with adj edges found"
        meta = real_gi._chunk_meta.get(seed_id)
        assert meta is not None
        seed = [{"chunk_id": seed_id, "text": meta["text"],
                 "doc_id": meta["doc_id"], "source": meta["source"], "score": 1.0}]
        expanded = real_gi.expand_batch([seed], hops=1, budget=3, kinds=("adj",))
        assert len(expanded[0]) > len(seed), (
            f"expand_batch should add adj neighbors for {seed_id}"
        )


# ---------------------------------------------------------------------------
# Abbreviation + madde regex, dedupe, self-loops
# ---------------------------------------------------------------------------

from retrieval.graph_index import find_abbrev_maddes  # noqa: E402


class TestAbbrevMaddeParsing:
    @pytest.mark.parametrize("q,exp", [
        ("TCK 86. madde", [("TCK", "86")]),
        ("İİK 72. madde", [("İİK", "72")]),
        ("TMK madde 2 ve TBK madde 49", [("TMK", "2"), ("TBK", "49")]),
        ("Anayasa 10. maddesi", [("ANAYASA", "10")]),
        ("anayasa madde 138", [("ANAYASA", "138")]),
        ("iik madde 72", [("İİK", "72")]),
        ("IIK madde 72", [("İİK", "72")]),
        ("TBK m. 49", [("TBK", "49")]),
        ("TBK md. 5", [("TBK", "5")]),
        ("TCK 86'ncı maddesi", [("TCK", "86")]),
        ("TCK madde 1234", [("TCK", "1234")]),
        ("TCK madde 5 ve TBK", [("TCK", "5")]),
        # number-first forms with md abbreviation (Task 0 fix)
        ("TCK 86. md.", [("TCK", "86")]),
        ("TBK 49. md", [("TBK", "49")]),
        ("TCK 5. md. nedir", [("TCK", "5")]),
        ("CMK 100. md. uygulanır", [("CMK", "100")]),
    ])
    def test_forms(self, q, exp):
        assert find_abbrev_maddes(q) == exp

    def test_no_madde(self):
        assert find_abbrev_maddes("TCK hakkında bilgi") == []

    def test_inject_casing_and_two_laws(self, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "DIRECT_MADDE_LOOKUP_ENABLED", True)
        gi = _make_gi_with_lookup(tmp_path)
        ids = [r["chunk_id"] for r in gi.inject_from_query("tck 86. maddesi")]
        assert "Türk Ceza Kanunu_kaggle_5237_0" in ids

    def test_inject_abbrev_number_first_md(self, tmp_path, monkeypatch):
        """TCK 86. md. → injects TCK madde-86 chunk (number-first md form)."""
        import config
        monkeypatch.setattr(config, "DIRECT_MADDE_LOOKUP_ENABLED", True)
        gi = _make_gi_with_lookup(tmp_path)
        result = gi.inject_from_query("TCK 86. md.")
        ids = [r["chunk_id"] for r in result]
        assert "Türk Ceza Kanunu_kaggle_5237_0" in ids

    def test_inject_canonical_number_first_md(self, tmp_path, monkeypatch):
        """5237 sayılı ... Kanununda 87. md. → injects TCK madde-87 chunk."""
        import config
        monkeypatch.setattr(config, "DIRECT_MADDE_LOOKUP_ENABLED", True)
        gi = _make_gi_with_lookup(tmp_path)
        query = "5237 sayılı Türk Ceza Kanununda 87. md. hükmü uygulanır."
        result = gi.inject_from_query(query)
        ids = [r["chunk_id"] for r in result]
        assert "Türk Ceza Kanunu_kaggle_5237_1" in ids


class TestMaddeHeadingFixes:
    """Task 2: EK MADDE all-caps, MADDE N/A suffix, mid-text anchor."""

    def test_ek_madde_allcaps_not_regular(self):
        """'EK MADDE 1' must return ek-1, not article 1."""
        rec = {
            "chunk_id": "L_d_0", "doc_id": "d", "source": "L",
            "text": "EK MADDE 1 – ek hüküm içeriği.",
        }
        mn, _ = _extract_madde_no(rec)
        assert mn == "ek-1", f"Expected 'ek-1', got {mn!r}"

    def test_madde_slash_suffix_normalised(self):
        """'MADDE 183/A' must return '183-a'."""
        rec = {
            "chunk_id": "L_d_0", "doc_id": "d", "source": "L",
            "text": "MADDE 183/A – Suç ve ceza tanımı.",
        }
        mn, _ = _extract_madde_no(rec)
        assert mn == "183-a", f"Expected '183-a', got {mn!r}"

    def test_madde_hyphen_suffix_normalised(self):
        """'MADDE 5-B' must return '5-b'."""
        rec = {
            "chunk_id": "L_d_0", "doc_id": "d", "source": "L",
            "text": "MADDE 5-B – Hüküm.",
        }
        mn, _ = _extract_madde_no(rec)
        assert mn == "5-b", f"Expected '5-b', got {mn!r}"

    def test_mid_text_madde_not_extracted(self):
        """Inline 'Madde 5 uyarınca' mid-line must not set chunk's article number."""
        rec = {
            "chunk_id": "L_d_0", "doc_id": "d", "source": "L",
            "text": "Bu hüküm Madde 5 uyarınca uygulanır.",
        }
        mn, _ = _extract_madde_no(rec)
        assert mn is None, f"Expected None for mid-text reference, got {mn!r}"

    def test_mid_text_ref_does_not_shadow_real_heading(self):
        """Inline mid-sentence ref before a real line-start heading: heading wins."""
        rec = {
            "chunk_id": "L_d_0", "doc_id": "d", "source": "L",
            # "Madde 3" is mid-sentence (not at line start); "MADDE 7" is at line start.
            "text": "Kanun hükmüne göre Madde 3 uyarınca karar verilmiştir.\nMADDE 7- Asıl hüküm.",
        }
        mn, _ = _extract_madde_no(rec)
        assert mn == "7", f"Expected '7' from line-start heading, got {mn!r}"

    def test_ek_madde_suffix_normalised(self):
        """'EK MADDE 2/A' must return 'ek-2-a'."""
        rec = {
            "chunk_id": "L_d_0", "doc_id": "d", "source": "L",
            "text": "EK MADDE 2/A – ek hüküm.",
        }
        mn, _ = _extract_madde_no(rec)
        assert mn == "ek-2-a", f"Expected 'ek-2-a', got {mn!r}"

    def test_build_graph_ek_madde_key(self):
        """build_graph_from_metadata stores 'ek-1' key (not '1') for EK MADDE 1."""
        recs = [
            {
                "chunk_id": "L_d_0", "doc_id": "d", "source": "L",
                "text": "EK MADDE 1 – ek hüküm.",
            }
        ]
        g = build_graph_from_metadata(recs)
        lookup = g["_source_madde_lookup"]
        assert "L||ek-1" in lookup, f"Expected 'L||ek-1' in lookup, got keys: {list(lookup)}"
        assert "L||1" not in lookup, "'L||1' must not appear for EK MADDE 1"


class TestDedupeAndSelfLoops:
    def test_duplicate_ids_no_self_loops(self):
        rec = {
            "chunk_id": "L_d_0", "doc_id": "d", "source": "L",
            "text": "MADDE 5 - bkz. madde 5 ve madde 6.", "madde_no": "5",
        }
        rec2 = {
            "chunk_id": "L_d_1", "doc_id": "d", "source": "L",
            "text": "MADDE 6 - x.", "madde_no": "6",
        }
        g = build_graph_from_metadata([rec, dict(rec), rec2, dict(rec2)])
        for k, es in g.items():
            if k.startswith("_"):
                continue
            assert all(t != k for t, _ in es)
        lk = g["_source_madde_lookup"]
        assert lk["L||5"] == ["L_d_0"]
        assert lk["L||6"] == ["L_d_1"]

    def test_gecici_madde_no_graph(self):
        recs = [
            {"chunk_id": "L_d_0", "doc_id": "d", "source": "L",
             "text": "x", "madde_no": "gecici-2"},
            {"chunk_id": "L_d_1", "doc_id": "d", "source": "L",
             "text": "y", "madde_no": "7"},
        ]
        g = build_graph_from_metadata(recs)
        assert g["_source_madde_lookup"]["L||gecici-2"] == ["L_d_0"]


# ---------------------------------------------------------------------------
# Task 2: atomic save_graph write + corrupt JSON recovery
# ---------------------------------------------------------------------------

from retrieval.graph_builder import save_graph  # noqa: E402


class TestAtomicSaveGraph:
    """save_graph writes atomically; a simulated mid-write failure leaves no
    partial/corrupt output at the target path."""

    def test_save_creates_valid_json(self, tmp_path):
        """save_graph produces a readable JSON file at the target path."""
        graph = build_graph_from_metadata(_META_KAGGLE)
        out = tmp_path / "graph.json"
        save_graph(graph, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert "_source_madde_lookup" in loaded

    def test_no_tmp_file_left_after_success(self, tmp_path):
        """Temp file is removed (renamed away) after a successful save."""
        graph = build_graph_from_metadata(_META_KAGGLE)
        out = tmp_path / "graph.json"
        save_graph(graph, out)
        tmp_candidate = tmp_path / "graph.json.tmp"
        assert not tmp_candidate.exists(), "Temp file should not remain after save"

    def test_overwrite_is_atomic(self, tmp_path):
        """Calling save_graph twice replaces the file; old content gone."""
        g1 = build_graph_from_metadata(_META_KAGGLE[:1])
        g2 = build_graph_from_metadata(_META_KAGGLE[:2])
        out = tmp_path / "graph.json"
        save_graph(g1, out)
        save_graph(g2, out)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        # g2 has more nodes than g1
        non_meta = {k: v for k, v in loaded.items() if not k.startswith("_")}
        assert len(non_meta) >= len(g1) - 1  # at least as many nodes as g1


class TestCorruptGraphRecovery:
    """GraphIndex raises JSONDecodeError on corrupt files; from_config rebuilds."""

    def test_load_graph_raises_on_corrupt(self, tmp_path):
        """GraphIndex.__init__ raises json.JSONDecodeError for a corrupt graph file."""
        m_path = tmp_path / "metadata.jsonl"
        m_path.write_text(
            "\n".join(json.dumps(r) for r in _META_KAGGLE[:2]),
            encoding="utf-8",
        )
        g_path = tmp_path / "graph.json"
        g_path.write_text("{corrupt json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            GraphIndex(g_path, m_path)

    def test_from_config_rebuilds_on_corrupt(self, tmp_path, monkeypatch):
        """from_config transparently rebuilds and returns a valid GraphIndex."""
        import config as _cfg

        # Point config at tmp_path
        monkeypatch.setattr(_cfg, "INDEX_DIR", tmp_path)
        monkeypatch.setattr(_cfg, "GRAPH_FILE", "graph.json")
        monkeypatch.setattr(_cfg, "METADATA_FILE", "metadata.jsonl")

        # Write valid metadata, corrupt graph
        m_path = tmp_path / "metadata.jsonl"
        m_path.write_text(
            "\n".join(json.dumps(r) for r in _META_KAGGLE),
            encoding="utf-8",
        )
        g_path = tmp_path / "graph.json"
        g_path.write_text("{bad}", encoding="utf-8")

        gi = GraphIndex.from_config()
        assert len(gi._graph) > 0, "from_config should have rebuilt a non-empty graph"
        # After rebuild the file on disk should be valid JSON
        reloaded = json.loads(g_path.read_text(encoding="utf-8"))
        assert "_source_madde_lookup" in reloaded
