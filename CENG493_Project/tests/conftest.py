"""
conftest.py — shared fixtures and heavy-module stubs for the test suite.

Stubs are installed into sys.modules *before* any test module is collected,
so every import of a stubbed module inside source files resolves to the mock
rather than raising ImportError.  Stubs are only inserted when the real
package is NOT importable (i.e. not installed), keeping behaviour identical
when the full environment is available.
"""

from __future__ import annotations

import importlib.machinery
import pathlib
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Heavy optional module stubs
# ---------------------------------------------------------------------------
# These packages are NOT listed in requirements-test.txt (GPU / heavy).
# Source modules import them at module level, so we pre-populate sys.modules
# with MagicMock objects so those modules can still be imported in pure-CPU CI.
_HEAVY: list[str] = [
    "torch",
    "faiss",
    "sentence_transformers",
    "transformers",
    "ollama",
    "rank_bm25",
    "openai",
    "nltk",
    "snowballstemmer",
    "evaluate",          # huggingface evaluate — no longer imported by the pipeline
    "bitsandbytes",
    "peft",
    "trl",
    "datasets",
]


def _module_stub(name: str) -> MagicMock:
    """MagicMock module with a real ``__spec__``: importlib.util.find_spec
    (used by e.g. transformers / datasets to probe for torch) raises
    ``ValueError: <name>.__spec__ is not set`` on a bare MagicMock."""
    mock = MagicMock(name=name)
    mock.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    mock.__path__ = []
    return mock


def _stub_if_missing(name: str) -> None:
    """Insert a MagicMock into sys.modules for *name* only if unimportable.

    Any exception counts as unimportable: a partly installed package (e.g.
    ``datasets`` present while ``torch`` is stubbed) can fail with more than
    ImportError at import time.
    """
    if name in sys.modules:
        return
    try:
        __import__(name)
    except Exception:
        for mod in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            sys.modules.pop(mod, None)
        mock = _module_stub(name)
        sys.modules[name] = mock

        # Populate common sub-module paths that source code accesses via
        # attribute chains so that e.g. `torch.cuda.is_available()` works.
        if name == "evaluate":
            # Force qa_metrics onto its pure-Python BLEU/ROUGE fallback:
            # a bare MagicMock would make hf_evaluate.load() succeed and
            # BLEU would come back as a MagicMock/1.0.
            mock.load.side_effect = RuntimeError("evaluate stub: load unavailable")
        if name == "torch":
            # scipy (installed as a ranx dep) checks:
            #   issubclass(cls, torch.Tensor)
            # issubclass() requires its second arg to be a real class, not a
            # MagicMock.  Give Tensor a real stub class so scipy won't raise
            # TypeError at import time.
            class _FakeTorchTensor:
                pass
            mock.Tensor = _FakeTorchTensor

            for sub in (
                "torch.cuda",
                "torch.backends",
                "torch.backends.mps",
                "torch.nn",
                "torch.nn.functional",
                "torch.utils",
                "torch.utils.data",
            ):
                if sub not in sys.modules:
                    sys.modules[sub] = _module_stub(sub)
        elif name == "transformers":
            for sub in (
                "transformers.AutoTokenizer",
                "transformers.AutoModelForCausalLM",
                "transformers.TrainingArguments",
            ):
                if sub not in sys.modules:
                    sys.modules[sub] = _module_stub(sub)
        elif name == "nltk":
            for sub in ("nltk.corpus", "nltk.corpus.stopwords"):
                if sub not in sys.modules:
                    sys.modules[sub] = _module_stub(sub)


for _mod in _HEAVY:
    _stub_if_missing(_mod)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    """Absolute path to the worktree root (one level above CENG493_Project/)."""
    # conftest.py lives at CENG493_Project/tests/conftest.py
    # .parent  → tests/
    # .parent  → CENG493_Project/
    # .parent  → worktree root
    return pathlib.Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def real_metadata_path(repo_root: pathlib.Path) -> pathlib.Path:
    """
    Path to results/index/metadata.jsonl at the repo root.

    Skips the test (pytest.skip) when the file does not exist so that
    data-dependent tests are still valid locally once the index is built.
    """
    path = repo_root / "results" / "index" / "metadata.jsonl"
    if not path.exists():
        pytest.skip(f"metadata.jsonl not found at {path}; skipping data-dependent test")
    return path
