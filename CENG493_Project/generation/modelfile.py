"""Ollama Modelfile SYSTEM block builder -- single source of truth.

Every script that emits a Modelfile SYSTEM directive must call
``build_system_block`` so the prompt text is defined in exactly one place.
"""

from __future__ import annotations


def build_system_block(prompt: str | None = None) -> str:
    """Return an Ollama Modelfile ``SYSTEM`` directive.

    Parameters
    ----------
    prompt : str, optional
        The system-prompt text.  Defaults to
        ``generation.rag_pipeline.TURKISH_PROMPT``.

    Returns
    -------
    str
        ``SYSTEM \\"\\"\\"<prompt>\\"\\"\\"`` with **no** extra leading or
        trailing whitespace / newlines inside the triple-quotes, so that
        parsing the block back yields the original *prompt* byte-identically.

    Raises
    ------
    ValueError
        If *prompt* contains a triple-quote sequence (``\\"\\"\\"``), which
        would break the Ollama Modelfile grammar.
    """
    if prompt is None:
        from generation.rag_pipeline import TURKISH_PROMPT

        prompt = TURKISH_PROMPT

    if '"""' in prompt:
        raise ValueError(
            'Prompt contains a triple-quote sequence (""") which would '
            "break Ollama Modelfile SYSTEM block syntax"
        )

    return f'SYSTEM """{prompt}"""'
