"""Tests for assemble_context budget enforcement, including first-chunk truncation."""


from generation.rag_pipeline import RAGPipeline, format_source


def _pipe(budget: int = 200) -> RAGPipeline:
    """Lightweight pipeline with no retriever and a tight char budget."""
    p = RAGPipeline(retriever=None, model="m", context_window_chars=budget)
    return p


def _chunk(text: str, source: str = "src", chunk_id: str = "c_0") -> dict:
    return {"text": text, "source": source, "chunk_id": chunk_id}


# ── First chunk truncation ─────────────────────────────────────────────

class TestFirstChunkTruncation:
    def test_first_chunk_exceeding_budget_is_truncated(self):
        """A single oversized first chunk must be truncated, not omitted."""
        big = "kelime " * 500  # ~3500 chars
        pipe = _pipe(budget=100)
        ctx, included = pipe.assemble_context([_chunk(big)])
        assert len(ctx) <= 100
        assert len(included) == 1
        # Header preserved
        assert ctx.startswith("[Kaynak 1]")

    def test_first_chunk_within_budget_unchanged(self):
        """First chunk fitting the budget must not be altered."""
        small_text = "Kisa metin."
        pipe = _pipe(budget=5000)
        ctx, included = pipe.assemble_context([_chunk(small_text)])
        expected = format_source(1, "src", small_text)
        assert ctx == expected
        assert len(included) == 1

    def test_truncation_preserves_header(self):
        """After truncation the [Kaynak 1] (source) header line must survive."""
        big = "a " * 1000
        pipe = _pipe(budget=80)
        ctx, _ = pipe.assemble_context([_chunk(big, source="TMK m.1")])
        first_line = ctx.split("\n")[0]
        assert "[Kaynak 1]" in first_line
        assert "TMK m.1" in first_line

    def test_truncation_on_word_boundary(self):
        """Truncated body must end at a word boundary (space), not mid-word."""
        text = "alfa beta gamma delta epsilon zeta eta theta iota kappa"
        pipe = _pipe(budget=60)
        ctx, _ = pipe.assemble_context([_chunk(text)])
        # Should not end in a broken word (no partial tokens)
        body = ctx.split("\n", 1)[1] if "\n" in ctx else ctx
        assert body.rstrip().endswith("..."), "Truncated text should end with ellipsis"


# ── Multi-chunk budget behaviour ────────────────────────────────────────

class TestMultiChunkBudget:
    def test_second_chunk_excluded_when_budget_exhausted(self):
        pipe = _pipe(budget=100)
        c1 = _chunk("Kisa.", chunk_id="c_0")
        c2 = _chunk("x " * 500, chunk_id="c_1")  # huge
        ctx, included = pipe.assemble_context([c1, c2])
        assert len(included) == 1
        assert included[0]["chunk_id"] == "c_0"

    def test_multiple_chunks_fit_within_budget(self):
        pipe = _pipe(budget=5000)
        chunks = [_chunk(f"Metin {i}", chunk_id=f"c_{i}") for i in range(3)]
        ctx, included = pipe.assemble_context(chunks)
        assert len(included) == 3
        for i in range(3):
            assert f"[Kaynak {i+1}]" in ctx

    def test_empty_chunks_list(self):
        pipe = _pipe(budget=100)
        ctx, included = pipe.assemble_context([])
        assert ctx == ""
        assert included == []


# ── _truncate_part unit tests ───────────────────────────────────────────

class TestTruncatePart:
    def test_short_part_unchanged(self):
        part = "[Kaynak 1] (src)\nKisa metin.\n\n"
        assert RAGPipeline._truncate_part(part, 500) == part

    def test_truncation_output_within_limit(self):
        part = format_source(1, "src", "kelime " * 300)
        result = RAGPipeline._truncate_part(part, 80)
        assert len(result) <= 80

    def test_header_preserved_after_truncation(self):
        part = format_source(1, "TMK", "genis " * 200)
        result = RAGPipeline._truncate_part(part, 60)
        assert result.startswith("[Kaynak 1] (TMK)\n")

    def test_separator_preserved_after_truncation(self):
        """`_truncate_part` must retain the trailing \\n\\n separator."""
        part = format_source(1, "src", "kelime " * 300)
        result = RAGPipeline._truncate_part(part, 80)
        assert result is not None
        assert result.endswith("\n\n"), (
            "Truncated part must end with \\n\\n chunk separator"
        )

    def test_bodyless_result_returns_none(self):
        """When remaining budget after header is zero or negative, return None."""
        # "[Kaynak 1] (src)\n" = 18 chars; separator "\n\n" = 2 chars
        # budget=20 → content_budget=18, header=18, remaining=0 → None
        part = format_source(1, "src", "some long text " * 50)
        result = RAGPipeline._truncate_part(part, 20)
        assert result is None, (
            "_truncate_part should return None when no body fits in budget"
        )


# ── AssembleResult / assemble_context_ex ───────────────────────────────────

class TestAssembleContextEx:
    """assemble_context_ex returns AssembleResult with truncation metadata."""

    def test_separator_preserved_in_assembled_context(self):
        """Each chunk block in assembled context must end with \\n\\n."""
        big = "kelime " * 500
        pipe = _pipe(budget=100)
        result = pipe.assemble_context_ex([_chunk(big)])
        # Only non-empty context is checked
        if result.context:
            assert result.context.endswith("\n\n"), (
                "Assembled context must end with \\n\\n separator"
            )

    def test_bodyless_chunk_dropped_not_included(self):
        """Chunk truncated to header-only (None from _truncate_part) must be dropped."""
        # budget=20: header=18 chars, content_budget=18, remaining=0 → None → dropped
        pipe = _pipe(budget=20)
        result = pipe.assemble_context_ex([_chunk("long body " * 100, chunk_id="drop_me")])
        assert result.context == ""
        assert result.included == []
        assert "drop_me" in result.dropped_ids

    def test_truncated_chunk_id_in_truncated_ids(self):
        """chunk_id of a shortened chunk must appear in truncated_ids."""
        big = "kelime " * 500
        pipe = _pipe(budget=100)
        result = pipe.assemble_context_ex([_chunk(big, chunk_id="trunc_c")])
        assert "trunc_c" in result.truncated_ids
        assert "trunc_c" in [c["chunk_id"] for c in result.included]

    def test_over_budget_second_chunk_in_dropped_ids(self):
        """Second chunk that breaks budget must appear in dropped_ids."""
        pipe = _pipe(budget=100)
        c1 = _chunk("Kisa.", chunk_id="c_0")
        c2 = _chunk("x " * 500, chunk_id="c_1")
        result = pipe.assemble_context_ex([c1, c2])
        assert len(result.included) == 1
        assert result.included[0]["chunk_id"] == "c_0"
        assert "c_1" in result.dropped_ids

    def test_hit_limit_true_when_chunk_dropped(self):
        pipe = _pipe(budget=100)
        c1 = _chunk("Kisa.", chunk_id="c_0")
        c2 = _chunk("x " * 500, chunk_id="c_1")
        result = pipe.assemble_context_ex([c1, c2])
        assert result.hit_limit is True

    def test_hit_limit_false_when_all_fit(self):
        pipe = _pipe(budget=5000)
        chunks = [_chunk(f"Metin {i}", chunk_id=f"c_{i}") for i in range(3)]
        result = pipe.assemble_context_ex(chunks)
        assert result.hit_limit is False
        assert result.dropped_ids == []
        assert len(result.included) == 3

    def test_assemble_context_backward_compat(self):
        """assemble_context must still return (context, included) 2-tuple."""
        pipe = _pipe(budget=5000)
        chunks = [_chunk("text", chunk_id="c_0")]
        ctx, included = pipe.assemble_context(chunks)
        assert isinstance(ctx, str)
        assert isinstance(included, list)
        assert len(included) == 1
