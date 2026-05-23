"""In-memory cross-reference graph for expanding retrieved chunks with neighbors."""

import json
import logging
from collections import deque
from pathlib import Path

log = logging.getLogger(__name__)

_DEFAULT_DECAY: dict[str, float] = {"adj": 0.85, "intra": 0.70, "cross": 0.60}


class GraphIndex:
    """In-memory cross-reference graph; expands retrieved chunks with neighbors."""

    def __init__(self, graph_path: str | Path, metadata_path: str | Path) -> None:
        self._graph: dict[str, list[tuple[str, str]]] = {}
        self._chunk_meta: dict[str, dict] = {}
        self._load_graph(Path(graph_path))
        self._load_metadata(Path(metadata_path))

    def _load_graph(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            raw: dict = json.load(fh)
        for key, edges in raw.items():
            if key.startswith("_"):
                continue
            self._graph[key] = [(nb_id, kind) for nb_id, kind in edges]

    def _load_metadata(self, path: Path) -> None:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record: dict = json.loads(line)
                cid = record.get("chunk_id")
                if cid is None:
                    continue
                self._chunk_meta[cid] = {
                    "text": record.get("text", ""),
                    "doc_id": record.get("doc_id", ""),
                    "source": record.get("source", ""),
                }

    def expand(
        self,
        chunks: list[dict],
        hops: int = 1,
        budget: int = 3,
        kinds: tuple[str, ...] = ("adj", "intra", "cross"),
        decay: dict[str, float] | None = None,
    ) -> list[dict]:
        if decay is None:
            decay = _DEFAULT_DECAY

        seen: set[str] = {c["chunk_id"] for c in chunks}
        added: list[dict] = []
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
    ) -> list[list[dict]]:
        return [
            self.expand(chunks, hops=hops, budget=budget, kinds=kinds, decay=decay)
            for chunks in batch
        ]

    @classmethod
    def from_config(cls) -> "GraphIndex":
        import config
        return cls(
            config.INDEX_DIR / config.GRAPH_FILE,
            config.INDEX_DIR / config.METADATA_FILE,
        )
