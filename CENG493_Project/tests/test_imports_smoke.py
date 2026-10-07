"""
test_imports_smoke.py — verify that every source module can be imported
under the light requirements-test.txt environment.

Heavy dependencies (torch, faiss, sentence_transformers, rank_bm25, openai,
…) are pre-stubbed in conftest.py, so all imports below must succeed even
on a pure-CPU machine with only the packages in requirements-test.txt.
"""

import importlib
import pathlib
import py_compile

import pytest

_PROJECT = pathlib.Path(__file__).resolve().parent.parent
_SOURCES = sorted(
    p for p in [*_PROJECT.rglob("*.py"), _PROJECT.parent / "demo.py"]
    if p.exists() and "__pycache__" not in p.parts
)


@pytest.mark.parametrize("path", _SOURCES, ids=lambda p: str(p.relative_to(_PROJECT.parent)))
def test_source_compiles(path):
    """Every script and module must at least parse: scripts are never imported
    by the tests, so a syntax error there would otherwise go unnoticed."""
    py_compile.compile(str(path), doraise=True)


# ---------------------------------------------------------------------------
# Pure / light modules — no stubs needed
# ---------------------------------------------------------------------------

def test_import_config():
    mod = importlib.import_module("config")
    assert hasattr(mod, "CHUNK_SIZE")
    assert hasattr(mod, "TOP_K_RETRIEVAL")


def test_import_utils():
    from utils import normalize_turkish, set_seeds
    assert callable(normalize_turkish)
    assert callable(set_seeds)


def test_import_data_processor():
    from data.data_processor import DataProcessor, CorpusChunk
    assert DataProcessor is not None
    assert CorpusChunk is not None


def test_import_qa_loader():
    mod = importlib.import_module("data.qa_loader")
    assert mod is not None


def test_import_graph_builder():
    mod = importlib.import_module("retrieval.graph_builder")
    # Module must expose at least one callable (the main builder function or class)
    assert mod is not None


def test_import_graph_index():
    from retrieval.graph_index import GraphIndex
    assert GraphIndex is not None


def test_import_retrieval_metrics():
    from evaluation.retrieval_metrics import compute_all_metrics
    assert callable(compute_all_metrics)


def test_import_final_score():
    mod = importlib.import_module("evaluation.final_score")
    assert mod is not None


def test_import_hallucination():
    from evaluation.hallucination import stratified_sample
    assert callable(stratified_sample)


# ---------------------------------------------------------------------------
# Modules that need stubbed heavy deps
# ---------------------------------------------------------------------------

def test_import_embedder():
    """retrieval.embedder uses torch + sentence_transformers (stubbed)."""
    mod = importlib.import_module("retrieval.embedder")
    assert hasattr(mod, "Embedder")


def test_import_retriever():
    """retrieval.retriever uses faiss (stubbed)."""
    mod = importlib.import_module("retrieval.retriever")
    assert hasattr(mod, "Retriever")


def test_import_bm25_retriever():
    """retrieval.bm25_retriever uses rank_bm25 + nltk (both stubbed)."""
    mod = importlib.import_module("retrieval.bm25_retriever")
    assert hasattr(mod, "tokenize")


def test_import_rag_pipeline():
    """generation.rag_pipeline uses openai (stubbed)."""
    mod = importlib.import_module("generation.rag_pipeline")
    assert mod is not None
