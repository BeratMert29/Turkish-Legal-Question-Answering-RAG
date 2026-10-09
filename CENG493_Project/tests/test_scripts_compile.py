"""test_scripts_compile.py — verify every .py in scripts/ compiles without SyntaxError."""

import py_compile
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
_SCRIPT_PATHS = sorted(_SCRIPTS_DIR.glob("*.py"))


@pytest.mark.parametrize("script_path", _SCRIPT_PATHS, ids=lambda p: p.name)
def test_script_compiles(script_path: Path) -> None:
    """py_compile raises SyntaxError on any syntax problem."""
    py_compile.compile(str(script_path), doraise=True)
