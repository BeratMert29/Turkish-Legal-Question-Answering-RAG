"""Conditional perplexity computation via HuggingFace transformers.

Computes PPL(answer | context, question) for each sample by masking the
prefix tokens (context + question) with -100 and running a single forward
pass through a causal LM loaded from HuggingFace Hub.
"""
import logging
import math

log = logging.getLogger(__name__)

# Ollama model name -> HuggingFace model ID mapping
_OLLAMA_TO_HF: dict[str, str] = {
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen25-legal-ft": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
}
_HF_DEFAULT = "Qwen/Qwen2.5-7B-Instruct"


def _resolve_hf_model_id(ollama_model: str) -> str:
    """Return HF model ID for the given Ollama model name."""
    return _OLLAMA_TO_HF.get(ollama_model, _HF_DEFAULT)


def compute_perplexity(
    predictions: list[dict],
    model: str,
    ollama_base: str = "http://localhost:11434",  # kept for signature compat, not used
    sample_size: int = 50,
    hf_model_id: str | None = None,
    use_4bit: bool = True,
) -> float | None:
    """Compute mean conditional perplexity of generated answers.

    For each sample the loss is evaluated only over the answer tokens; the
    prefix (context + question) tokens are masked with -100 so they do not
    contribute to the cross-entropy loss.

    Parameters
    ----------
    predictions:
        List of dicts, each containing:
        - ``question``        (str)
        - ``predicted``       (str) — the generated answer
        - ``retrieved_chunks`` (list[dict]) — each with ``text`` and ``source``
    model:
        Ollama model name; used only to derive ``hf_model_id`` when that
        argument is *None*.
    ollama_base:
        Ignored.  Kept so existing callers do not need to be updated.
    sample_size:
        Maximum number of samples to evaluate.
    hf_model_id:
        Explicit HuggingFace model ID.  When *None* the ID is derived from
        ``model`` via the built-in mapping.
    use_4bit:
        Load the model in 4-bit NF4 quantisation to reduce VRAM usage.

    Returns
    -------
    float or None
        Mean perplexity across successful samples, or *None* if fewer than
        3 samples could be evaluated.
    """
    # 1. Availability check — import lazily so the module is importable
    #    even when torch / transformers are not installed.
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        log.warning(
            "torch and/or transformers are not installed; "
            "perplexity computation is unavailable."
        )
        return None

    # 2. Resolve HuggingFace model ID and build quantisation config.
    resolved_hf_id = hf_model_id if hf_model_id is not None else _resolve_hf_model_id(model)
    log.info("Loading HuggingFace model '%s' for perplexity evaluation.", resolved_hf_id)

    model_kwargs: dict = {"device_map": "auto"}

    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model_kwargs["quantization_config"] = bnb_config
        except ImportError:
            log.warning(
                "bitsandbytes is not installed; loading model in full precision."
            )

    # 3. Load tokenizer and model (lazy, inside the function).
    tokenizer = AutoTokenizer.from_pretrained(resolved_hf_id, use_fast=True)
    hf_model = AutoModelForCausalLM.from_pretrained(resolved_hf_id, **model_kwargs)
    hf_model.eval()

    # 4. Determine the device for tensor placement.
    #    With device_map="auto" the model may be split; use the embedding
    #    layer's device as the input device.
    try:
        input_device = next(hf_model.parameters()).device
    except StopIteration:
        input_device = torch.device("cpu")

    # 5. Iterate over samples and compute per-sample perplexity.
    sampled = predictions[:sample_size]
    perplexities: list[float] = []

    for pred in sampled:
        question = pred.get("question", "")
        answer = pred.get("predicted", "").strip()
        chunks = pred.get("retrieved_chunks", [])

        if not answer or not chunks:
            continue

        try:
            # Build context block (same format as rag_pipeline.py).
            context_parts = [
                f"[Kaynak {i + 1}] ({c.get('source', '')}) {c.get('text', '')}"
                for i, c in enumerate(chunks[:5])
            ]
            context_block = "\n\n".join(context_parts)

            prefix = f"Bağlam:\n{context_block}\n\nSoru: {question}"
            full_text = prefix + f"\n\nCevap: {answer}"

            # Cap lengths to avoid OOM on very long inputs.
            prefix = prefix[:3000]
            full_text = full_text[:4000]

            # Tokenise prefix and full text separately so we know the
            # exact boundary between prefix and answer tokens.
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
            full_ids = tokenizer.encode(full_text, add_special_tokens=True)

            # Ensure full_ids is at least as long as prefix_ids; if
            # truncation made them equal no answer tokens remain.
            if len(full_ids) <= len(prefix_ids):
                log.debug("No answer tokens remain after tokenisation; skipping sample.")
                continue

            full_tensor = torch.tensor([full_ids], dtype=torch.long, device=input_device)

            # Build labels: mask prefix tokens with -100.
            labels = full_tensor.clone()
            labels[0, : len(prefix_ids)] = -100

            with torch.no_grad():
                outputs = hf_model(input_ids=full_tensor, labels=labels)
                loss: torch.Tensor = outputs.loss

            if torch.isfinite(loss):
                perplexities.append(math.exp(loss.item()))

        except Exception as exc:
            log.debug("Perplexity computation failed for sample: %s", exc)
            continue

    # 6. Aggregate results.
    if not perplexities:
        log.warning(
            "No perplexity values were computed (all samples failed or were skipped)."
        )
        return None

    if len(perplexities) < 3:
        log.warning(
            "Only %d sample(s) succeeded; result is not meaningful. Returning None.",
            len(perplexities),
        )
        return None

    mean_ppl = sum(perplexities) / len(perplexities)
    log.info(
        "Perplexity computed over %d samples: %.4f", len(perplexities), mean_ppl
    )
    return round(mean_ppl, 4)
