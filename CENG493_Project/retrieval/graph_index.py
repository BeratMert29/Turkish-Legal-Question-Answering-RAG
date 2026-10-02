"""In-memory cross-reference graph for expanding retrieved chunks with neighbors."""

import json
import logging
import re
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_DECAY: dict[str, float] = {"adj": 0.85, "intra": 0.70, "cross": 0.60}

# ── direct madde lookup patterns ──────────────────────────────────────────

# "5237 sayılı Türk Ceza Kanununda" — captures kanun_no and law name.
_QUERY_KANUN_RE = re.compile(
    r"(\d{2,5})\s*sayılı\s+"
    r"([A-Za-zÇĞİÖŞÜçğıöşü .\-']+?)\s+"
    r"(?:Kanunu?|Yasası?)[a-zçğıöşü]*",
)

# "madde 86" / "MADDE 86" — used in the window after a law reference, or standalone.
_QUERY_MADDE_NUM_RE = re.compile(r"(?:madde|MADDE)\s*(\d{1,4})", re.IGNORECASE)

# Common Turkish law abbreviations → normalized source name.
_LAW_ABBREVS: dict[str, str] = {
    "TCK": "Türk Ceza Kanunu",
    "CMK": "Ceza Muhakemesi Kanunu",
    "TMK": "Türk Medeni Kanunu",
    "TBK": "Türk Borçlar Kanunu",
    "HMK": "Hukuk Muhakemeleri Kanunu",
    "TTK": "Türk Ticaret Kanunu",
    "İYUK": "İdari Yargılama Usulü Kanunu",
    "İİK": "İcra ve İflas Kanunu",
    "DMK": "Devlet Memurları Kanunu",
    "Anayasa": "Türkiye Cumhuriyeti Anayasası",
}


def _abbrev_key(s: str) -> str:
    """Turkish-aware key: all i/İ/ı/I variants collapse to İ, then upper()."""
    return s.translate(str.maketrans("iIı", "İİİ")).upper()


_LAW_ABBREVS_NORM: dict[str, str] = {_abbrev_key(k): v for k, v in _LAW_ABBREVS.items()}

_TR_LETTER = "A-Za-zÇĞİIıÖŞÜçğöşü"
_I = "[İIiı]"

# Law abbreviation (case-insensitive, Turkish i/İ/ı/I variants accepted).
_ABBREV_RE = re.compile(
    rf"(?<![{_TR_LETTER}])"
    rf"(TCK|CMK|TMK|TBK|HMK|TTK|{_I}YUK|{_I}{_I}K|DMK|Anayasa)"
    rf"(?![{_TR_LETTER}])",
    re.IGNORECASE,
)

# Article number right after a law name, within that law's own window:
# "madde 86", "md. 5", "m. 49", "86. madde", "86. maddesi", "86'ncı maddesi".
_MADDE_AFTER_ABBREV_RE = re.compile(
    rf"(?<![{_TR_LETTER}])(?:madde|md|m)\.?\s*(?<!\d)(\d{{1,4}})(?!\d)"
    r"|(?<!\d)(\d{1,4})(?!\d)"
    r"(?:\s*['\u2019]?\s*(?:inci|ıncı|nci|ncı|üncü|uncu))?"
    r"\s*\.?\s*madde",
    re.IGNORECASE,
)

_ABBREV_WINDOW = 60  # max chars after an abbreviation to look for its madde


def find_abbrev_maddes(query: str) -> list[tuple[str, str]]:
    """Return [(abbrev_key, madde_no)] binding each number to the nearest
    preceding law abbreviation; each abbreviation's window ends at the next one."""
    ams = list(_ABBREV_RE.finditer(query))
    out: list[tuple[str, str]] = []
    for i, am in enumerate(ams):
        end = am.end() + _ABBREV_WINDOW
        if i + 1 < len(ams):
            end = min(end, ams[i + 1].start())
        window = query[am.end(): end]
        window = window.split("\n", 1)[0]
        mm = _MADDE_AFTER_ABBREV_RE.search(window)
        if mm:
            out.append((_abbrev_key(am.group(1)), mm.group(1) or mm.group(2)))
    return out


_LOOKUP_WINDOW = 250  # chars after law name to search for madde number


class GraphIndex:
    """In-memory cross-reference graph; expands retrieved chunks with neighbors."""

    def __init__(self, graph_path: str | Path, metadata_path: str | Path) -> None:
        self._graph: dict[str, list[tuple[str, str]]] = {}
        self._chunk_meta: dict[str, dict] = {}
        self._source_madde_lookup: dict[str, list[str]] = {}
        self._load_graph(Path(graph_path))
        self._load_metadata(Path(metadata_path))

    def _load_graph(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            raw: dict = json.load(fh)
        for key, edges in raw.items():
            if key == "_source_madde_lookup":
                # Store lookup table; values may be lists or dicts depending on
                # serialisation format.
                self._source_madde_lookup = {
                    k: (v if isinstance(v, list) else list(v))
                    for k, v in edges.items()
                }
            elif not key.startswith("_"):
                self._graph[key] = [(nb_id, kind) for nb_id, kind in edges]

    def _load_metadata(self, path: Path) -> None:
        from utils import read_jsonl
        for record in read_jsonl(path):
            cid = record.get("chunk_id")
            if cid is None:
                continue
            self._chunk_meta[cid] = {
                "text": record.get("text", ""),
                "doc_id": record.get("doc_id", ""),
                "source": record.get("source", ""),
            }

    # ── direct madde injection ────────────────────────────────────────────

    def inject_from_query(self, query: str, exclude: "set[str] | None" = None) -> list[dict]:
        """Return chunks for articles explicitly referenced in *query*.

        Recognises two patterns:

        * Canonical: ``"5237 sayılı Türk Ceza Kanununda … madde 86"``
        * Abbreviation: ``"TCK madde 86"`` / ``"TCK 86. madde"``

        Only active when ``config.DIRECT_MADDE_LOOKUP_ENABLED`` is True
        (default False).  Returns ``[]`` when the lookup table is empty or
        the flag is off.
        """
        try:
            import config as _cfg
            if not getattr(_cfg, "DIRECT_MADDE_LOOKUP_ENABLED", False):
                return []
        except ImportError:
            return []

        if not self._source_madde_lookup:
            return []

        if exclude is None:
            exclude = set()

        from retrieval.graph_builder import _resolve_cross_source

        results: list[dict] = []
        seen: set[str] = set(exclude)

        def _add_chunks(source: str, madde_no: str) -> None:
            key = f"{source}||{madde_no}"
            for cid in self._source_madde_lookup.get(key, []):
                if cid in seen:
                    continue
                meta = self._chunk_meta.get(cid)
                if meta:
                    results.append({
                        "chunk_id": cid,
                        "text": meta["text"],
                        "doc_id": meta["doc_id"],
                        "source": meta["source"],
                        "score": 1.0,  # injected directly, no retrieval score
                    })
                    seen.add(cid)

        # Pattern A: "N sayılı ... Kanunu ... madde M"
        for lm in _QUERY_KANUN_RE.finditer(query):
            kno = lm.group(1)
            kname = lm.group(2).strip()
            src = _resolve_cross_source(kno, kname)
            if src:
                window = query[lm.end(): lm.end() + _LOOKUP_WINDOW]
                mm = _QUERY_MADDE_NUM_RE.search(window)
                if mm:
                    _add_chunks(src, mm.group(1))

        # Pattern B: abbreviation like "TCK madde 86" / "TCK 86. madde"
        for abbrev, madde_no in find_abbrev_maddes(query):
            src = _LAW_ABBREVS_NORM.get(abbrev)
            if src:
                _add_chunks(src, madde_no)

        return results

    # ── graph expansion ───────────────────────────────────────────────────

    def expand(
        self,
        chunks: list[dict],
        hops: int = 1,
        budget: int = 3,
        kinds: tuple[str, ...] = ("adj", "intra", "cross"),
        decay: dict[str, float] | None = None,
        query: "str | None" = None,
    ) -> list[dict]:
        if decay is None:
            decay = _DEFAULT_DECAY

        seen: set[str] = {c["chunk_id"] for c in chunks}

        # Inject directly-referenced article chunks first (when enabled).
        injected: list[dict] = []
        if query is not None:
            injected = self.inject_from_query(query, exclude=seen)
            seen.update(c["chunk_id"] for c in injected)

        added: list[dict] = list(injected)
        remaining_budget = budget

        sorted_chunks = sorted(chunks, key=lambda c: c["score"], reverse=True)
        queue: deque[tuple[str, float, int]] = deque(
            (c["chunk_id"], c["score"], 0) for c in sorted_chunks
        )

        while queue and remaining_budget > 0:
            current_id, parent_score, depth = queue.popleft()
            for nb_id, kind in self._graph.get(current_id, []):
                if remaining_budget <= 0:
                    break
                if kind not in kinds or nb_id in seen:
                    continue
                meta = self._chunk_meta.get(nb_id)
                if meta is None:
                    seen.add(nb_id)
                    continue
                nb_score = parent_score * decay.get(kind, 0.7)
                added.append({
                    "chunk_id": nb_id,
                    "text": meta["text"],
                    "doc_id": meta["doc_id"],
                    "source": meta["source"],
                    "score": nb_score,
                })
                seen.add(nb_id)
                remaining_budget -= 1
                if depth + 1 < hops:
                    queue.append((nb_id, nb_score, depth + 1))

        return chunks + added

    def expand_batch(
        self,
        batch: list[list[dict]],
        hops: int = 1,
        budget: int = 3,
        kinds: tuple[str, ...] = ("adj", "intra", "cross"),
        decay: dict[str, float] | None = None,
        queries: "list[str] | None" = None,
    ) -> list[list[dict]]:
        if queries is None:
            queries = [None] * len(batch)  # type: ignore[list-item]
        return [
            self.expand(chunks, hops=hops, budget=budget, kinds=kinds, decay=decay, query=q)
            for chunks, q in zip(batch, queries)
        ]

    @classmethod
    def from_config(cls) -> "GraphIndex":
        import config
        return cls(
            config.INDEX_DIR / config.GRAPH_FILE,
            config.INDEX_DIR / config.METADATA_FILE,
        )
