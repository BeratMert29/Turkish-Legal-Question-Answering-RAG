import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from typing import Protocol, runtime_checkable
import config

@runtime_checkable
class EmbedderProtocol(Protocol):
    """Interface contract for embedding models. Implement this to swap embedders in Stage 2."""
    def load_model(self) -> None: ...
    def encode(self, texts: list[str], is_query: bool = False,
               show_progress: bool = True) -> "np.ndarray": ...

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
        if "e5" in self.model_name.lower():
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


class BGEM3Embedder:
    """
    BGE-M3 multi-vector embedder using FlagEmbedding.
    Supports dense, sparse (lexical), and ColBERT (multi-vector) retrieval modes simultaneously.
    Requires: pip install FlagEmbedding

    VRAM usage: ~3GB with use_fp16=True on RTX 4070 Super (12GB).
    """

    def __init__(self, model_name: str = None, batch_size: int = None, device: str = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
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
        """Load BGEM3FlagModel with fp16 for VRAM efficiency (~3GB)."""
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError(
                "FlagEmbedding is required for BGEM3Embedder. "
                "Install with: pip install FlagEmbedding"
            )
        self.model = BGEM3FlagModel(
            self.model_name,
            use_fp16=True,
            device=self.device,
        )

    def encode(self, texts: list[str], is_query: bool = False,
               show_progress: bool = True) -> np.ndarray:
        """Returns (N, 1024) dense embeddings, L2-normalized. Satisfies EmbedderProtocol."""
        return self.encode_dense(texts, is_query=is_query, show_progress=show_progress)

    def encode_dense(self, texts: list[str], is_query: bool = False,
                     show_progress: bool = True) -> np.ndarray:
        """Returns (N, 1024) float32 dense embeddings, already L2-normalized by BGEM3FlagModel."""
        if self.model is None:
            raise RuntimeError("Call load_model() before encode_dense()")
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=1024,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            show_progress_bar=show_progress,
        )
        return out["dense_vecs"].astype(np.float32)

    def encode_sparse(self, texts: list[str], is_query: bool = False) -> list[dict]:
        """Returns list of {token_id: weight} sparse lexical vectors (one per text)."""
        if self.model is None:
            raise RuntimeError("Call load_model() before encode_sparse()")
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=1024,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
            show_progress_bar=False,
        )
        return out["lexical_weights"]

    def encode_colbert(self, texts: list[str], is_query: bool = False) -> list[np.ndarray]:
        """Returns list of (seq_len, 1024) per-token ColBERT embeddings."""
        if self.model is None:
            raise RuntimeError("Call load_model() before encode_colbert()")
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=1024,
            return_dense=False,
            return_sparse=False,
            return_colbert_vecs=True,
            show_progress_bar=False,
        )
        return out["colbert_vecs"]

    def encode_multi(self, texts: list[str], is_query: bool = False,
                     show_progress: bool = True) -> dict:
        """
        Encode texts with all three retrieval modes in a single forward pass.

        Returns:
            {
                "dense":   np.ndarray of shape (N, 1024),
                "sparse":  list of N dicts {token_id: weight},
                "colbert": list of N np.ndarray of shape (seq_len, 1024),
            }
        """
        if self.model is None:
            raise RuntimeError("Call load_model() before encode_multi()")
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=1024,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            show_progress_bar=show_progress,
        )
        return {
            "dense": out["dense_vecs"].astype(np.float32),
            "sparse": out["lexical_weights"],
            "colbert": out["colbert_vecs"],
        }
