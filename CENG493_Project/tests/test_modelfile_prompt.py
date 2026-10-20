"""Tests for Modelfile SYSTEM block helper and its adoption in scripts 09/13."""

import ast
import re
from pathlib import Path

import pytest

from generation.modelfile import build_system_block
from generation.rag_pipeline import TURKISH_PROMPT

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_SCRIPT_09 = _SCRIPTS_DIR / "09_load_finetuned_model.py"
_SCRIPT_13 = _SCRIPTS_DIR / "13_export_lora_to_ollama.py"

# Regex that extracts the prompt from a SYSTEM """...""" block.
# Ollama Modelfile grammar: SYSTEM """<content>"""
_SYSTEM_RE = re.compile(r'^SYSTEM\s+"""(.*?)"""', re.MULTILINE | re.DOTALL)


# ── Round-trip: build -> parse -> byte-identical ──────────────────────────

def test_round_trip_default_prompt():
    """SYSTEM block built with TURKISH_PROMPT must parse back identically."""
    block = build_system_block()
    match = _SYSTEM_RE.search(block)
    assert match is not None, "Generated block does not match Ollama SYSTEM syntax"
    parsed = match.group(1)
    assert parsed == TURKISH_PROMPT, (
        "Round-trip mismatch: parsed prompt differs from TURKISH_PROMPT"
    )


def test_round_trip_custom_prompt():
    custom = "Custom multi-line\nsystem prompt."
    block = build_system_block(custom)
    match = _SYSTEM_RE.search(block)
    assert match is not None
    assert match.group(1) == custom


def test_no_extra_newlines():
    """The block must NOT inject leading/trailing newlines around the prompt."""
    block = build_system_block()
    # After opening """, the very next character must be the first char of the prompt
    idx = block.index('"""') + 3
    assert block[idx] != "\n", "Extra leading newline after opening triple-quote"
    # Before closing """, the character must be the last char of the prompt
    closing = block.rindex('"""')
    assert block[closing - 1] != "\n", "Extra trailing newline before closing triple-quote"


# ── Validation: triple-quote in prompt -> ValueError ─────────────────────

def test_triple_quote_in_prompt_raises():
    bad = 'This prompt has """ inside it'
    with pytest.raises(ValueError, match='triple-quote'):
        build_system_block(bad)


def test_triple_quote_at_edges():
    with pytest.raises(ValueError):
        build_system_block('"""leading')
    with pytest.raises(ValueError):
        build_system_block('trailing"""')


# ── Scripts 09 & 13 use the helper ──────────────────────────────────────

def _script_calls_build_system_block(script_path: Path) -> bool:
    """Return True if *script_path* calls ``build_system_block`` (AST check)."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            # Direct call: build_system_block(...)
            if isinstance(func, ast.Name) and func.id == "build_system_block":
                return True
            # Qualified call: modelfile.build_system_block(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "build_system_block"
            ):
                return True
    return False


def _script_imports_build_system_block(script_path: Path) -> bool:
    """Return True if *script_path* imports ``build_system_block``."""
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "modelfile" in node.module:
            if any(alias.name == "build_system_block" for alias in node.names):
                return True
    return False


def test_script_09_uses_helper():
    assert _script_imports_build_system_block(_SCRIPT_09), (
        "script 09 must import build_system_block from generation.modelfile"
    )
    assert _script_calls_build_system_block(_SCRIPT_09), (
        "script 09 must call build_system_block()"
    )


def test_script_13_uses_helper():
    assert _script_imports_build_system_block(_SCRIPT_13), (
        "script 13 must import build_system_block from generation.modelfile"
    )
    assert _script_calls_build_system_block(_SCRIPT_13), (
        "script 13 must call build_system_block()"
    )


# ── Old hardcoded single-line SYSTEM prompt is gone ──────────────────────

_OLD_SINGLE_LINE = 'SYSTEM "Sen Türk hukuku alanında uzman bir hukuki asistansın."'


def test_old_single_line_not_in_09():
    source = _SCRIPT_09.read_text(encoding="utf-8")
    assert _OLD_SINGLE_LINE not in source, (
        "script 09 still contains the old single-line SYSTEM prompt"
    )


def test_old_single_line_not_in_13():
    source = _SCRIPT_13.read_text(encoding="utf-8")
    assert _OLD_SINGLE_LINE not in source, (
        "script 13 still contains the old single-line SYSTEM prompt"
    )


# ── Script 13 still imports TURKISH_PROMPT (single source of truth) ─────

def test_script_13_imports_turkish_prompt():
    tree = ast.parse(_SCRIPT_13.read_text(encoding="utf-8"))
    found = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "generation.rag_pipeline"
        and any(alias.name == "TURKISH_PROMPT" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert found, "script 13 must import TURKISH_PROMPT from generation.rag_pipeline"
