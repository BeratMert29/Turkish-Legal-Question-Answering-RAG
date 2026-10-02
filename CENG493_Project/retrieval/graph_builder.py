"""Build a cross-reference graph over Turkish legal corpus chunks."""

import re
import json
import logging
from collections import defaultdict
from pathlib import Path

import config

log = logging.getLogger(__name__)

# ── compiled patterns ─────────────────────────────────────────────

_INTRA_RE = re.compile(
    r"(?:^|[\s\(,;\.])(?:Madde|MADDE|m\.)\s*(\d{1,4})(?:/\d+[a-zçğıöşü]?)?",
    re.MULTILINE,
)

_CROSS_LAW_RE = re.compile(
    r"(\d{2,5})\s*sayılı\s+"
    r"([A-Za-zÇĞİÖŞÜçğıöşü \.''\-]+?)\s+"
    r"(?:Kanunu?|Yasası?)[a-zçğıöşü]*",
    # No \b: Turkish case suffixes like "Kanununda", "Kanunundan" must also match.
)

_MADDE_WINDOW_RE = re.compile(
    r"(?:(?:Madde|MADDE|m\.)\s*(\d{1,4})"
    r"|(\d{1,4})\s*(?:\.?\s*)?(?:inci|ıncı|nci|ncı|üncü|uncu)?\s*\.?\s*(?:madde))",
    re.IGNORECASE,
)

# Legacy: chunk_ids ending with _m<num>(_<sub>)? (old format).
_CHUNK_SUFFIX_RE = re.compile(r"m(\d+)(?:_(\d+))?$")

# doc_id pattern: e.g. "Hukuk Muhakemeleri Kanunu_madde_42"
_DOC_ID_MADDE_RE = re.compile(r"_madde_(\d+)$", re.IGNORECASE)

# First MADDE heading in text (covers Ek Madde / Geçici Madde / regular).
_TEXT_MADDE_RE = re.compile(
    r"(?:"
    r"(?:Ek|EK)\s+[Mm]adde\s+(\d+)"        # group 1: ek-N
    r"|[Gg]eçici\s+[Mm]adde\s+(\d+)"        # group 2: gecici-N
    r"|(?:GEÇİCİ\s+MADDE)\s+(\d+)"          # group 3: gecici-N all-caps
    r"|(?:MADDE|Madde)\s+(\d+)"             # group 4: regular N
    r")"
)

# Trailing integer in chunk_id for sub-chunk ordering.
_TRAILING_INT_RE = re.compile(r"_(\d+)$")

_CROSS_WINDOW = 200

# ── reverse look-ups from HMGS_SOURCE_MAP ─────────────────────────


def _build_reverse_maps() -> tuple[dict[str, str], dict[str, str]]:
    """Build kanun_number→source and lowered_name→source maps."""
    num_map: dict[str, str] = {}
    name_map: dict[str, str] = {}
    for key, norm in config.HMGS_SOURCE_MAP.items():
        m = re.match(r"(\d+)", key)
        if m:
            num_map[m.group(1)] = norm
        name_map[norm.lower()] = norm
    return num_map, name_map


_NUM_MAP, _NAME_MAP = _build_reverse_maps()

# ── internal helpers ──────────────────────────────────────────────


def _parse_sub_idx(chunk_id: str) -> int | None:
    """Return the trailing integer of chunk_id for sub-chunk ordering."""
    m = _TRAILING_INT_RE.search(chunk_id)
    return int(m.group(1)) if m else None


def _parse_chunk_suffix(
    chunk_id: str, source: str, doc_id: str,
) -> tuple[str | None, int | None]:
    """Legacy fallback: return (madde_no, sub_idx) from old _m<num>(_sub)? suffix."""
    prefix = f"{source}_{doc_id}_"
    if not chunk_id.startswith(prefix):
        return None, None
    m = _CHUNK_SUFFIX_RE.match(chunk_id[len(prefix):])
    if not m:
        return None, None
    return m.group(1), (int(m.group(2)) if m.group(2) is not None else None)


def _extract_madde_no(rec: dict) -> tuple[str | None, int | None]:
    """Return (madde_no, sub_idx) from a metadata record using multi-source resolution.

    Resolution order (first match wins):
    1. ``rec["madde_no"]`` — explicit field written by updated ``_article_chunk``.
    2. doc_id suffix ``_madde_<N>`` — extra_laws.jsonl format.
    3. First MADDE / Ek Madde / Geçici Madde heading in ``rec["text"]``.
    4. Legacy chunk_id suffix ``_m<N>(_sub)?`` — old test fixtures / pre-fix index.

    ``sub_idx`` is always derived from the trailing integer of ``chunk_id`` so that
    sub-chunks of the same article can be ordered correctly.
    """
    # 1. Explicit madde_no field (new CorpusChunk serialization).
    madde_no = rec.get("madde_no")
    if madde_no is not None:
        return str(madde_no), _parse_sub_idx(rec["chunk_id"])

    # 2. doc_id pattern: e.g. "Devlet Memurları Kanunu_madde_44"
    doc_id = rec.get("doc_id", "")
    m = _DOC_ID_MADDE_RE.search(doc_id)
    if m:
        return m.group(1), _parse_sub_idx(rec["chunk_id"])

    # 3. Text-based extraction — look for the first MADDE heading in the chunk text.
    text = rec.get("text", "")
    if text:
        tm = _TEXT_MADDE_RE.search(text[:600])
        if tm:
            if tm.group(1):
                return f"ek-{tm.group(1)}", _parse_sub_idx(rec["chunk_id"])
            if tm.group(2) or tm.group(3):
                n = tm.group(2) or tm.group(3)
                return f"gecici-{n}", _parse_sub_idx(rec["chunk_id"])
            if tm.group(4):
                return tm.group(4), _parse_sub_idx(rec["chunk_id"])

    # 4. Legacy: old chunk_id suffix _m<num>(_sub)?
    src = rec.get("source", "")
    return _parse_chunk_suffix(rec["chunk_id"], src, doc_id)


def _madde_no_is_numeric(madde_no: str) -> bool:
    """Return True when madde_no represents a plain integer (not ek-/gecici-)."""
    return madde_no.isdigit()


def _resolve_cross_source(kanun_no: str, kanun_name: str) -> str | None:
    """Map a kanun number / partial name to a normalized source."""
    if kanun_no in _NUM_MAP:
        return _NUM_MAP[kanun_no]
    for candidate in (f"{kanun_name} Kanunu", kanun_name):
        hit = _NAME_MAP.get(candidate.lower().strip())
        if hit:
            return hit
    kl = kanun_name.lower().strip()
    for known_lower, known_norm in _NAME_MAP.items():
        if kl in known_lower or known_lower in kl:
            return known_norm
    return None


# ── public API ────────────────────────────────────────────────────


def extract_references(
    text: str, doc_id: str, source: str,
) -> dict[str, list[tuple[str, ...]]]:
    """Return {"intra": [(madde, raw), ...], "cross": [(kanun_no, madde, raw), ...]}."""
    cross: list[tuple[str, ...]] = []
    cross_spans: set[tuple[int, int]] = set()

    for lm in _CROSS_LAW_RE.finditer(text):
        win_start = lm.end()
        win = text[win_start: win_start + _CROSS_WINDOW]
        mm = _MADDE_WINDOW_RE.search(win)
        if mm:
            mno = mm.group(1) or mm.group(2)
            cross.append((lm.group(1), mno, lm.group(0)))
            cross_spans.add((win_start + mm.start(), win_start + mm.end()))

    intra: list[tuple[str, ...]] = []
    for m in _INTRA_RE.finditer(text):
        overlaps = any(m.start() < ce and m.end() > cs for cs, ce in cross_spans)
        if not overlaps:
            intra.append((m.group(1), m.group(0).strip()))

    return {"intra": intra, "cross": cross}


def build_graph_from_metadata(
    metadata: list[dict],
) -> dict[str, list[tuple[str, str]]]:
    """Build the full cross-reference graph (two passes over metadata).

    Pass 1 — resolve madde_no for every chunk via :func:`_extract_madde_no`,
    which handles four formats:

    * New ``madde_no`` field written by updated ``_article_chunk``.
    * doc_id suffix ``_madde_<N>`` (extra_laws.jsonl format).
    * First ``MADDE / Ek Madde / Geçici Madde`` heading in chunk text
      (catches existing published metadata.jsonl without re-chunking).
    * Legacy ``_m<N>(_sub)?`` chunk_id suffix (old test fixtures / backward compat).

    Pass 2 — scan chunk texts for intra- and cross-law references and add edges.
    """
    src_madde: dict[tuple[str, str], list[str]] = defaultdict(list)
    doc_madde: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list),
    )
    # cinfo: chunk_id → (madde_no, sub_idx)
    cinfo: dict[str, tuple[str | None, int | None]] = {}

    seen_ids: set[str] = set()
    for rec in metadata:
        cid, src, did = rec["chunk_id"], rec["source"], rec["doc_id"]
        if cid in seen_ids:
            continue
        seen_ids.add(cid)
        mn, si = _extract_madde_no(rec)
        cinfo[cid] = (mn, si)
        if mn is not None:
            src_madde[(src, mn)].append(cid)
            doc_madde[(src, did)][mn].append(cid)

    log.info(
        "Pass-1 done: %d chunks, %d source·madde keys, %d doc groups",
        len(metadata), len(src_madde), len(doc_madde),
    )

    edges: dict[str, set[tuple[str, str]]] = defaultdict(set)

    for (_src, _did), mm in doc_madde.items():
        # Sort only numeric madde_nos for adjacency; skip ek-/gecici- in ordering.
        numeric_nums = sorted(
            (mn for mn in mm if _madde_no_is_numeric(mn)),
            key=int,
        )
        for i, mn in enumerate(numeric_nums):
            if i + 1 < len(numeric_nums) and int(numeric_nums[i + 1]) - int(mn) == 1:
                nxt = numeric_nums[i + 1]
                for a in mm[mn]:
                    for b in mm[nxt]:
                        if a == b:
                            continue
                        edges[a].add((b, "adj"))
                        edges[b].add((a, "adj"))
            # Sub-chunk adjacency within same madde (e.g. long article split into pieces).
            clist = mm[mn]
            if len(clist) > 1:
                ordered = sorted(
                    clist,
                    key=lambda c: cinfo[c][1] if cinfo[c][1] is not None else -1,
                )
                for j in range(len(ordered) - 1):
                    if ordered[j] == ordered[j + 1]:
                        continue
                    edges[ordered[j]].add((ordered[j + 1], "adj"))
                    edges[ordered[j + 1]].add((ordered[j], "adj"))

    done_ids: set[str] = set()
    for rec in metadata:
        cid, src = rec["chunk_id"], rec["source"]
        if cid in done_ids:
            continue
        done_ids.add(cid)
        text = rec.get("text", "")
        if not text:
            continue

        refs = extract_references(text, rec["doc_id"], src)

        for mno, _raw in refs["intra"]:
            for tgt in src_madde.get((src, mno), []):
                if tgt != cid:
                    edges[cid].add((tgt, "intra"))

        for kno, mno, raw in refs["cross"]:
            lm = _CROSS_LAW_RE.search(raw)
            kname = lm.group(2).strip() if lm else ""
            tsrc = _resolve_cross_source(kno, kname)
            if tsrc is None or tsrc == src:
                continue
            for tgt in src_madde.get((tsrc, mno), []):
                if tgt != cid:
                    edges[cid].add((tgt, "cross"))

    graph: dict[str, list[tuple[str, str]]] = {
        cid: sorted(es) for cid, es in edges.items()
    }

    lookup_ser: dict[str, list[str]] = {
        f"{s}||{m}": list(dict.fromkeys(cids)) for (s, m), cids in src_madde.items()
    }
    graph["_source_madde_lookup"] = lookup_ser  # type: ignore[assignment]

    log.info(
        "Graph ready: %d nodes with edges",
        sum(1 for k in graph if not k.startswith("_")),
    )
    return graph


def lookup_by_source_madde(
    graph: dict,
    source: str,
    madde_no: str,
) -> list[str]:
    """Return chunk_ids for the given (source, madde_no) pair from the lookup table.

    This is the *direct reference lookup* used when a query explicitly mentions
    e.g. "TCK 86. madde" or "5237 sayılı kanun madde 86".  The caller is
    responsible for resolving the source name via :func:`_resolve_cross_source`
    and normalising madde_no to a string integer before calling this function.

    Enabled only when ``config.DIRECT_MADDE_LOOKUP_ENABLED`` is True (default
    False) so existing pipelines are not affected.
    """
    lookup = graph.get("_source_madde_lookup", {})
    return list(lookup.get(f"{source}||{madde_no}", []))


def save_graph(graph: dict, path: Path) -> None:
    """Write graph JSON (includes _source_madde_lookup)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=1)
    log.info("Graph saved → %s (%d bytes)", path, path.stat().st_size)


def load_graph(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Load graph from JSON, stripping _source_madde_lookup."""
    with open(Path(path), "r", encoding="utf-8") as f:
        raw = json.load(f)
    raw.pop("_source_madde_lookup", None)
    return {k: [tuple(e) for e in v] for k, v in raw.items()}


def graph_stats(graph: dict) -> dict:
    """Return {total_nodes, total_edges, by_kind: {adj, intra, cross}}."""
    by_kind: dict[str, int] = defaultdict(int)
    total = 0
    nodes = 0
    for k, es in graph.items():
        if k.startswith("_"):
            continue
        nodes += 1
        for _, kind in es:
            by_kind[kind] += 1
            total += 1
    return {"total_nodes": nodes, "total_edges": total, "by_kind": dict(by_kind)}
