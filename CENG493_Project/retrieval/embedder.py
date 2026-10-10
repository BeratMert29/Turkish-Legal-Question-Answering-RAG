import re
from pathlib import PurePath

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import config

def uses_e5_prefixes(model_name: str) -> bool:
    """True for E5-family models ("intfloat/multilingual-e5-large"), which need
    "query: " / "passage: " prefixes.  Only the last path component is checked,
    as a dash/underscore-delimited token, so a local path that merely
    contains "e5" somewhere (e.g. ".../run-3fe51a/bge-m3") does not match."""
    name = PurePath(str(model_name).replace("\\", "/")).name.lower()
    return re.search(r"(?:^|[-_])e5(?:[-_]|$)", name) is not None


class Embedder:
    def __init__(self, model_name: str = config.EMBEDDING_MODEL,
                 batch_size: int = config.EMBEDDING_BATCH_SIZE,
                 device: str = None):
        self.model_name = model_name
        self.batch_size = batch_size
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model = None  # loaded lazily via load_model()

    def load_model(self) -> None:
        """Load SentenceTransformer model onto device."""
        self.model = SentenceTransformer(self.model_name, device=self.device)
        if self.device == "cuda":
            self.model.half()  # fp16 on CUDA: ~half the VRAM, faster encode

    def encode(self, texts: list[str], is_query: bool = False,
               show_progress: bool = True) -> np.ndarray:
        """
        Applies E5 query/passage prefixes only for E5 models; BGE-M3 and others receive raw text.
        is_query=True  → prepends "query: " (E5 only)
        is_query=False → prepends "passage: " (E5 only)
        Returns (N, 1024) float32, explicitly L2-normalized.
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before encode()")
        if uses_e5_prefixes(self.model_name):
            prefix = "query: " if is_query else "passage: "
            prefixed = [prefix + t for t in texts]
        else:
            prefixed = texts
        embeddings = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        assert embeddings.shape[1] == config.EMBEDDING_DIM, (
            f"Embedding dim mismatch: model produced {embeddings.shape[1]}, "
            f"expected config.EMBEDDING_DIM={config.EMBEDDING_DIM}"
        )
        return embeddings
