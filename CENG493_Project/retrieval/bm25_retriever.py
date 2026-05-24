"""BM25 retrieval for hybrid dense+sparse search."""
import re
import numpy as np
from rank_bm25 import BM25Okapi
from utils import normalize_turkish
import config

try:
    from snowballstemmer import stemmer as _snowball_stemmer
    _TR_STEMMER = _snowball_stemmer("turkish")
    def _stem(token: str) -> str:
        return _TR_STEMMER.stemWord(token)
except ImportError:
    def _stem(token: str) -> str:
        return token

import nltk
try:
    _STOPWORDS = set(nltk.corpus.stopwords.words('turkish'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    try:
        _STOPWORDS = set(nltk.corpus.stopwords.words('turkish'))
    except Exception:
        _STOPWORDS = set()


def tokenize(text: str) -> list[str]:
    normalized = normalize_turkish(text)
    # Strip punctuation so tokens like "madde," match "madde"
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    return [_stem(w) for w in normalized.split() if len(w) >= config.BM25_MIN_TOKEN_LENGTH and w not in _STOPWORDS]


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.metadata: list[dict] = []

    def build(self, metadata: list[dict]) -> None:
        self.metadata = metadata
        corpus = [tokenize(m['text']) for m in metadata]
        self.bm25 = BM25Okapi(corpus)

    def get_scores(self, query: str) -> np.ndarray:
        """Return min-max normalized BM25 scores for all documents."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        min_s = scores.min()
        max_s = scores.max()
        range_s = max_s - min_s
        if range_s > 0:
            scores = (scores - min_s) / range_s
        else:
            scores = np.zeros_like(scores)
        return scores.astype(np.float32)

    def get_top_k(self, query: str, k: int = 100) -> list[tuple[int, float]]:
        """Return top-k (index, raw_score) pairs sorted by BM25 score."""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]
