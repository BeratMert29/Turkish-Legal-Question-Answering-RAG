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
    r"(?:Kanunu?|Yasası?)\b",
)

_MADDE_WINDOW_RE = re.compile(
    r"(?:(?:Madde|MADDE|m\.)\s*(\d{1,4})"
    r"|(\d{1,4})\s*(?:\.?\s*)?(?:inci|ıncı|nci|ncı|üncü|uncu)?\s*\.?\s*(?:madde))",
    re.IGNORECASE,
)

_CHUNK_SUFFIX_RE = re.compile(r"m(\d+)(?:_(\d+))?$")

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


def _parse_chunk_suffix(
    chunk_id: str, source: str, doc_id: str,
) -> tuple[str | None, int | None]:
    """Return (madde_no, sub_idx) from the chunk_id suffix."""
    prefix = f"{source}_{doc_id}_"
    if not chunk_id.startswith(prefix):
        return None, None
    m = _CHUNK_SUFFIX_RE.match(chunk_id[len(prefix):])
    if not m:
        return None, None
    return m.group(1), (int(m.group(2)) if m.group(2) is not None else None)


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
    """Build the full cross-reference graph (two passes over metadata)."""
    src_madde: dict[tuple[str, str], list[str]] = defaultdict(list)
    doc_madde: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list),
    )
    cinfo: dict[str, tuple[str | None, int | None]] = {}

    for rec in metadata:
        cid, src, did = rec["chunk_id"], rec["source"], rec["doc_id"]
        mn, si = _parse_chunk_suffix(cid, src, did)
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
        nums = sorted(mm, key=int)
        for i, mn in enumerate(nums):
            if i + 1 < len(nums) and int(nums[i + 1]) - int(mn) == 1:
                nxt = nums[i + 1]
                for a in mm[mn]:
                    for b in mm[nxt]:
                        edges[a].add((b, "adj"))
                        edges[b].add((a, "adj"))
            clist = mm[mn]
            if len(clist) > 1:
                ordered = sorted(
                    clist,
                    key=lambda c: cinfo[c][1] if cinfo[c][1] is not None else -1,
                )
                for j in range(len(ordered) - 1):
                    edges[ordered[j]].add((ordered[j + 1], "adj"))
                    edges[ordered[j + 1]].add((ordered[j], "adj"))

    for rec in metadata:
        cid, src = rec["chunk_id"], rec["source"]
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
        f"{s}||{m}": cids for (s, m), cids in src_madde.items()
    }
    graph["_source_madde_lookup"] = lookup_ser  # type: ignore[assignment]

    log.info(
        "Graph ready: %d nodes with edges",
        sum(1 for k in graph if not k.startswith("_")),
    )
    return graph


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
