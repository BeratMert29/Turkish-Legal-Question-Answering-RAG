"""Tests for scripts/08 context-length handling, config integration, and
truncation safety.

Covers:
  - [:4000] hard-coded truncation is removed
  - max_length reads from config.FINETUNE_MAX_SEQ_LEN
  - Context (not answer) is truncated when sequence is too long
  - Answer/citation target is NEVER truncated
  - Dataset resolution uses resolve_processed_file (no silent fallback)
  - gradient_checkpointing is enabled
  - Batch/accumulation keeps effective batch size = 8
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "08_finetune_llm.py"


def _source() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _import_module():
    """Import scripts/08 as a module (deferred heavy imports won't fire)."""
    import config  # noqa: F401 — ensure config is importable
    spec = importlib.util.spec_from_file_location("finetune_llm", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# (a) [:4000] truncation removed
# ---------------------------------------------------------------------------

class TestNoHardcodedTruncation:
    def test_no_4000_slice_in_format_as_chat(self):
        """The old ``[:4000]`` char truncation must not appear in format_as_chat."""
        src = _source()
        # Match any [:N] slice with N near 4000 in the format_as_chat function area
        assert "[:4000]" not in src, (
            "Hard-coded [:4000] context truncation still present in 08_finetune_llm.py"
        )

    def test_no_default_6000_comment(self):
        """The old comment claimed inference default was 6000; actual is 14000."""
        src = _source()
        assert "default 6000" not in src.lower(), (
            "Stale comment claiming inference default is 6000 still present"
        )


# ---------------------------------------------------------------------------
# (b) max_length from config.FINETUNE_MAX_SEQ_LEN
# ---------------------------------------------------------------------------

class TestMaxLengthFromConfig:
    def test_config_has_finetune_max_seq_len(self):
        import config
        assert hasattr(config, "FINETUNE_MAX_SEQ_LEN"), (
            "config.py must define FINETUNE_MAX_SEQ_LEN"
        )
        assert isinstance(config.FINETUNE_MAX_SEQ_LEN, int)
        assert config.FINETUNE_MAX_SEQ_LEN >= 2048, (
            "FINETUNE_MAX_SEQ_LEN should be >= 2048 (measured p99 ~ 3093)"
        )

    def test_training_config_reads_from_config(self):
        """TRAINING_CONFIG['training']['max_length'] must reference config, not a literal."""
        src = _source()
        # The value in the dict should reference config.FINETUNE_MAX_SEQ_LEN,
        # NOT be a hardcoded integer like 2048.
        assert "config.FINETUNE_MAX_SEQ_LEN" in src, (
            "max_length should reference config.FINETUNE_MAX_SEQ_LEN"
        )

    def test_training_config_max_length_value(self):
        mod = _import_module()
        import config
        assert mod.TRAINING_CONFIG["training"]["max_length"] == config.FINETUNE_MAX_SEQ_LEN


# ---------------------------------------------------------------------------
# (c) Truncation direction: answer NEVER truncated
# ---------------------------------------------------------------------------

class TestTruncationDirection:
    def test_safe_truncate_preserves_answer(self):
        """Given a very long context and short answer, the answer must survive
        intact in the formatted output."""
        mod = _import_module()

        # Minimal mock tokenizer that behaves like a real one for this test
        class FakeTokenizer:
            def encode(self, text, **kw):
                # ~1 token per 3 chars (conservative for Turkish)
                return list(range(len(text) // 3))

            def decode(self, token_ids, **kw):
                # Inverse: 3 chars per token
                return "x" * (len(token_ids) * 3)

            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=False):
                parts = []
                for m in messages:
                    parts.append(f"<|{m['role']}|>\n{m['content']}")
                return "\n".join(parts)

        tokenizer = FakeTokenizer()
        max_len = 500  # Tight budget

        long_context = "A" * 9000  # Will need truncation
        short_answer = "Kisa cevap. [Kaynak 1]"
        question = "Test sorusu?"

        truncated_ctx = mod._safe_truncate_context(
            long_context, question, short_answer, tokenizer, max_len,
        )
        # Context should be shorter than original
        assert len(truncated_ctx) < len(long_context), "Context was not truncated"

        # Now format the full chat — answer must be intact
        example = {
            "question": question,
            "context_str": long_context,
            "answer": short_answer,
        }
        # Monkey-patch config for this test
        import config
        original = config.FINETUNE_MAX_SEQ_LEN
        try:
            config.FINETUNE_MAX_SEQ_LEN = max_len
            formatted = mod.format_as_chat(example, tokenizer)
        finally:
            config.FINETUNE_MAX_SEQ_LEN = original

        assert short_answer in formatted, (
            f"Answer was truncated or missing from formatted output!\n"
            f"Answer: {short_answer!r}\n"
            f"Formatted (tail): ...{formatted[-200:]!r}"
        )

    def test_short_sequence_no_truncation(self):
        """When the full sequence fits, context is returned unchanged."""
        mod = _import_module()

        class FakeTokenizer:
            def encode(self, text, **kw):
                return list(range(len(text) // 3))
            def decode(self, token_ids, **kw):
                return "x" * (len(token_ids) * 3)
            def apply_chat_template(self, messages, tokenize=False,
                                    add_generation_prompt=False):
                return "\n".join(m["content"] for m in messages)

        tokenizer = FakeTokenizer()
        ctx = "Short context."
        result = mod._safe_truncate_context(
            ctx, "Q?", "A.", tokenizer, 99999,
        )
        assert result == ctx, "Short context should not be modified"


# ---------------------------------------------------------------------------
# (d) VRAM: gradient checkpointing & batch settings
# ---------------------------------------------------------------------------

class TestVRAMSettings:
    def test_gradient_checkpointing_enabled(self):
        mod = _import_module()
        assert mod.TRAINING_CONFIG["training"]["gradient_checkpointing"] is True

    def test_effective_batch_size_preserved(self):
        """per_device_train_batch_size * gradient_accumulation_steps == effective_batch_size."""
        mod = _import_module()
        tc = mod.TRAINING_CONFIG["training"]
        actual = tc["per_device_train_batch_size"] * tc["gradient_accumulation_steps"]
        assert actual == tc["effective_batch_size"], (
            f"Effective batch size mismatch: {tc['per_device_train_batch_size']} * "
            f"{tc['gradient_accumulation_steps']} = {actual}, expected {tc['effective_batch_size']}"
        )

    def test_batch_size_reduced_for_longer_sequences(self):
        """With max_length >= 4096, per_device_train_batch_size should be <= 2."""
        mod = _import_module()
        import config
        if config.FINETUNE_MAX_SEQ_LEN >= 4096:
            tc = mod.TRAINING_CONFIG["training"]
            assert tc["per_device_train_batch_size"] <= 2, (
                f"batch_size={tc['per_device_train_batch_size']} too large for "
                f"max_length={config.FINETUNE_MAX_SEQ_LEN}; reduce to fit 12GB VRAM"
            )


# ---------------------------------------------------------------------------
# (e) Dataset resolution: no silent fallback
# ---------------------------------------------------------------------------

class TestDatasetResolution:
    def test_uses_resolve_processed_file(self):
        """Script must import and use resolve_processed_file."""
        src = _source()
        assert "resolve_processed_file" in src, (
            "08 must use resolve_processed_file from utils for dataset paths"
        )

    def test_no_silent_fallback_chain(self):
        """The old 3-way silent fallback (RAG -> merged -> fallback) must be gone."""
        src = _source()
        assert "FALLBACK_DATASET" not in src, (
            "Silent FALLBACK_DATASET path still present in 08"
        )

    def test_no_sys_exit_on_missing_data(self):
        """Missing data should raise FileNotFoundError, not sys.exit."""
        src = _source()
        # The old pattern: sys.exit(1) after checking .exists()
        # New pattern: resolve_processed_file raises FileNotFoundError
        # There should be no "sys.exit(1)" near dataset loading
        lines = src.split("\n")
        in_dataset_section = False
        for line in lines:
            if "resolve_processed_file" in line:
                in_dataset_section = True
            if in_dataset_section and "sys.exit" in line:
                pytest.fail("sys.exit found near dataset loading; should raise FileNotFoundError")
            if in_dataset_section and "Loading dataset" in line:
                break

    def test_rag_dataset_required_by_default(self):
        """Without --no-rag-context, the script must require qa_train_rag.jsonl
        (not silently fall back to merged)."""
        src = _source()
        # The default path (no flag) should call resolve_processed_file on the
        # RAG dataset name without a try/except fallback to merged.
        # Check that the else-branch (default) directly resolves RAG_DATASET_NAME.
        assert "RAG_DATASET_NAME" in src
        # The old try/except pattern that caught FileNotFoundError and fell
        # back to MERGED_DATASET_NAME must be gone from the default path.
        # Find the dataset resolution block — should NOT have a bare
        # try: resolve(RAG) except: resolve(MERGED) pattern anymore.
        try_fallback = re.search(
            r"try:\s*\n\s*dataset_path\s*=\s*resolve_processed_file\(RAG_DATASET_NAME\)\s*\n"
            r"\s*except\s+FileNotFoundError:\s*\n"
            r"\s*dataset_path\s*=\s*resolve_processed_file\(MERGED_DATASET_NAME\)",
            src,
        )
        assert try_fallback is None, (
            "Silent try/except fallback from RAG to merged dataset still present. "
            "Merged should only be used with --no-rag-context flag."
        )

    def test_no_rag_context_flag_exists(self):
        """The --no-rag-context CLI flag must be defined."""
        src = _source()
        assert "--no-rag-context" in src, (
            "Missing --no-rag-context CLI flag for explicit merged dataset fallback"
        )

    def test_merged_fallback_requires_flag(self):
        """MERGED_DATASET_NAME should only appear in the --no-rag-context branch."""
        src = _source()
        lines = src.split("\n")
        for i, line in enumerate(lines):
            if "MERGED_DATASET_NAME" in line and "resolve_processed_file" in line:
                # This line should be inside the no_rag_context branch
                # Look at the preceding lines for the flag check
                context_window = "\n".join(lines[max(0, i - 5):i + 1])
                assert "no_rag_context" in context_window, (
                    f"MERGED_DATASET_NAME used with resolve_processed_file outside "
                    f"--no-rag-context guard at line {i + 1}"
                )
