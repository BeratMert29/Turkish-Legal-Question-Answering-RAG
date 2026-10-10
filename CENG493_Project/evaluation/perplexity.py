"""Conditional perplexity of generated answers via HuggingFace transformers.

PPL(answer | system prompt, context, question) under the generator's own
weights: the HF base of the Ollama model plus, for the fine-tuned LLM, its
LoRA adapter.  The input is built with the model's chat template, exactly as
the generator was prompted, and only answer tokens are scored (the prompt is
masked with -100).  When prompt + answer exceed ``max_tokens`` the prompt is
cut from the left (oldest context first); the answer is never truncated.
"""
import logging
import math
import random
from pathlib import Path

log = logging.getLogger(__name__)

# Ollama model name -> (HuggingFace base model ID, uses LoRA adapter)
_OLLAMA_TO_HF: dict[str, tuple[str, bool]] = {
    "qwen2.5:7b": ("Qwen/Qwen2.5-7B-Instruct", False),
    "qwen2.5:14b": ("Qwen/Qwen2.5-14B-Instruct", False),
    "qwen2.5:3b": ("Qwen/Qwen2.5-3B-Instruct", False),
}


def resolve_generator_weights(ollama_model: str) -> tuple[str | None, Path | None]:
    """``(hf_base_id, adapter_dir)`` matching an Ollama generator model.

    The fine-tuned model maps to ``config.LORA_BASE_HF_MODEL`` plus the LoRA
    adapter; unknown models map to ``(None, None)``.
    """
    import config

    if ollama_model == config.LLM_FINETUNED_MODEL:
        for d in (config.LORA_ADAPTER_DIR, config.LORA_ADAPTER_FALLBACK_DIR):
            if (Path(d) / "adapter_config.json").exists():
                return config.LORA_BASE_HF_MODEL, Path(d)
        log.warning("LoRA adapter not found in %s or %s",
                    config.LORA_ADAPTER_DIR, config.LORA_ADAPTER_FALLBACK_DIR)
        return None, None
    base, _ = _OLLAMA_TO_HF.get(ollama_model, (None, False))
    return base, None


def build_scoring_ids(tokenizer, system_prompt: str, user: str, answer: str,
                      max_tokens: int) -> tuple[list[int], int] | None:
    """Token ids of prompt + answer and the number of prompt tokens.

    The prompt is the chat-templated system + user turn with the generation
    prompt appended; when the total exceeds *max_tokens* prompt tokens are
    dropped from the left.  Returns None when the answer alone does not fit.
    """
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user}]
    prompt_ids = list(tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True))
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if not answer_ids or len(answer_ids) >= max_tokens:
        return None
    keep = max_tokens - len(answer_ids)
    if len(prompt_ids) > keep:
        prompt_ids = prompt_ids[-keep:]
    return prompt_ids + answer_ids, len(prompt_ids)


def compute_perplexity(
    predictions: list[dict],
    model: str,
    sample_size: int | None = 100,
    hf_model_id: str | None = None,
    adapter_dir: "str | Path | None" = None,
    use_4bit: bool = True,
    max_tokens: int = 4096,
    short_answer_mode: bool = False,
    seed: int = 42,
) -> float | None:
    """Geometric-mean conditional perplexity of generated answers.

    Parameters
    ----------
    predictions:
        Dicts with ``question``, ``predicted`` and ``retrieved_chunks`` (the
        chunks the generator saw).
    model:
        Ollama generator name; resolves the HF base + adapter unless
        *hf_model_id* is given.
    sample_size:
        Random sample (seeded) of predictions; None scores all.

    Returns
    -------
    float or None
        None when weights cannot be resolved/loaded or fewer than 3 samples
        succeed.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        log.warning("torch and/or transformers are not installed; perplexity unavailable.")
        return None

    from generation.rag_pipeline import (
        SHORT_ANSWER_PROMPT, TURKISH_PROMPT, format_source, user_message,
    )

    if hf_model_id is None:
        hf_model_id, adapter_dir = resolve_generator_weights(model)
    if hf_model_id is None:
        log.warning("No HF weights known for generator %r; perplexity skipped.", model)
        return None
    log.info("Perplexity model: %s%s", hf_model_id,
             f" + LoRA {adapter_dir}" if adapter_dir else "")

    model_kwargs: dict = {"device_map": "auto"}
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        except ImportError:
            log.warning("bitsandbytes is not installed; loading model in full precision.")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_id, **model_kwargs)
    if adapter_dir is not None:
        from peft import PeftModel
        hf_model = PeftModel.from_pretrained(hf_model, str(adapter_dir))
    hf_model.eval()
    try:
        input_device = next(hf_model.parameters()).device
    except StopIteration:
        input_device = torch.device("cpu")

    system_prompt = SHORT_ANSWER_PROMPT if short_answer_mode else TURKISH_PROMPT
    pool = [p for p in predictions
            if (p.get("predicted") or "").strip() and p.get("retrieved_chunks")]
    if sample_size is not None and len(pool) > sample_size:
        pool = random.Random(seed).sample(pool, sample_size)

    perplexities: list[float] = []
    for pred in pool:
        # Native answer: injected [Kaynak N] tags were not generated by the model.
        answer = (pred.get("predicted_native") or pred["predicted"]).strip()
        context = "".join(format_source(i + 1, c.get("source", ""), c.get("text", ""))
                          for i, c in enumerate(pred["retrieved_chunks"]))
        built = build_scoring_ids(tokenizer, system_prompt,
                                  user_message(pred.get("question", ""), context),
                                  answer, max_tokens)
        if built is None:
            continue
        ids, n_prompt = built
        try:
            full = torch.tensor([ids], dtype=torch.long, device=input_device)
            labels = full.clone()
            labels[0, :n_prompt] = -100
            with torch.no_grad():
                loss = hf_model(input_ids=full, labels=labels).loss
            if torch.isfinite(loss):
                perplexities.append(math.exp(loss.item()))
        except Exception as exc:
            log.debug("Perplexity computation failed for sample: %s", exc)

    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if len(perplexities) < 3:
        log.warning("Only %d perplexity sample(s) succeeded; returning None.", len(perplexities))
        return None
    # Geometric mean of per-sample PPL (exp of mean per-sample loss).
    mean_ppl = math.exp(sum(math.log(p) for p in perplexities) / len(perplexities))
    log.info("Perplexity over %d samples (geometric mean): %.4f", len(perplexities), mean_ppl)
    return round(mean_ppl, 4)
