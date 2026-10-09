import numpy as np

from evaluation.semantic_similarity import _chunk_words, _encode_long


class FakeModel:
    def __init__(self):
        self.seen = []

    def encode(self, texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True):
        self.seen.extend(texts)
        # embedding: first word hash -> axis; deterministic unit vectors
        out = np.zeros((len(texts), 4))
        for i, t in enumerate(texts):
            out[i, len(t.split()) % 4] = 1.0
        return out


def test_chunk_words_short_unchanged():
    assert _chunk_words("a b c", 10) == ["a b c"]


def test_chunk_words_splits_long():
    text = " ".join(str(i) for i in range(25))
    chunks = _chunk_words(text, 10)
    assert [len(c.split()) for c in chunks] == [10, 10, 5]
    assert " ".join(chunks) == text


def test_long_text_not_truncated_and_averaged():
    m = FakeModel()
    text = " ".join(["w"] * 600)  # window 512 -> 256 words per chunk
    emb = _encode_long(m, [text, "kısa"], 512)
    assert len(m.seen) == 4  # 3 chunks + 1 short
    assert emb.shape == (2, 4)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)
    # chunks have 256,256,88 words -> axes 0,0,0 => averaged stays on axis 0
    assert emb[0, 0] == 1.0


def test_config_defaults():
    import config
    # Model name is now sourced directly from config (no module-level fallback)
    assert "multilingual" in config.SEMANTIC_SIM_MODEL and "MiniLM" not in config.SEMANTIC_SIM_MODEL
