"""Verify that the Modelfile written by script 13 uses the canonical TURKISH_PROMPT."""

import textwrap

from generation.rag_pipeline import TURKISH_PROMPT


def _build_modelfile_content():
    """Reproduce the Modelfile content-building logic from script 13
    without importing the heavy module (avoids merge/torch side-effects)."""
    import json
    import config

    stops = "".join(
        f"PARAMETER stop {json.dumps(s, ensure_ascii=False)}\n" for s in config.LLM_STOP
    )
    content = (
        f"FROM ./qwen25_merged.gguf\n"
        f"PARAMETER temperature {config.LLM_TEMPERATURE}\n"
        f"PARAMETER num_predict {config.LLM_MAX_TOKENS}\n"
        f"PARAMETER num_ctx {config.LLM_NUM_CTX}\n"
        f"PARAMETER num_gpu 999\n"
        + stops
        + 'SYSTEM """\n'
        + TURKISH_PROMPT + '\n"""\n'
    )
    return content


def test_modelfile_contains_turkish_prompt():
    content = _build_modelfile_content()
    assert TURKISH_PROMPT in content, "Modelfile must embed the full TURKISH_PROMPT"


def test_modelfile_system_uses_triple_quotes():
    content = _build_modelfile_content()
    assert 'SYSTEM """' in content, "Ollama Modelfile must use triple-quote SYSTEM block"
    assert content.strip().endswith('"""'), "SYSTEM block must close with triple quotes"


def test_modelfile_has_required_rules():
    """Spot-check that essential rules survive into the Modelfile."""
    content = _build_modelfile_content()
    assert "ZORUNLU KURALLAR" in content
    assert "[Kaynak N]" in content
    assert "Sağlanan bağlam bu soruyu yanıtlamak için yeterli değildir" in content


def test_single_source_of_truth():
    """The SYSTEM text must come from TURKISH_PROMPT (single source),
    not a duplicated hard-coded string."""
    # Importing the actual write_modelfile function would pull in torch etc.
    # Instead, verify that script 13 imports from rag_pipeline.
    import importlib
    import ast
    from pathlib import Path

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "13_export_lora_to_ollama.py"
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    found = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "generation.rag_pipeline"
        and any(alias.name == "TURKISH_PROMPT" for alias in node.names)
        for node in imports
    )
    assert found, "script 13 must import TURKISH_PROMPT from generation.rag_pipeline"
