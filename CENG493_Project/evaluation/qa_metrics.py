from collections import Counter
import math
import re
import warnings
import evaluate as hf_evaluate
from utils import normalize_turkish

try:
    _BLEU_METRIC = hf_evaluate.load("bleu")
    _ROUGE_METRIC = hf_evaluate.load("rouge")
    _USE_HF_EVALUATE = True
except Exception:
    _USE_HF_EVALUATE = False
    warnings.warn(
        "hf_evaluate not available; using fallback BLEU/ROUGE implementation. "
        "Scores may differ from standard implementations.",
        ImportWarning,
        stacklevel=2,
    )

_CITATION_PATTERN = re.compile(r"\[\s*kaynak\s+(\d+)\s*\]", re.IGNORECASE)
_STRIP_CITATION_PATTERN = re.compile(r"\[\s*kaynak\s+\d+\s*\]", re.IGNORECASE)


def strip_citations(text: str) -> str:
    """Remove [Kaynak N] markers from text before F1/EM comparison."""
    return _STRIP_CITATION_PATTERN.sub("", text).strip()


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Turkish-lowercase (I->ı, İ->i) then split on Unicode word characters.

    Punctuation never glues to words ("madde." == "madde") and Turkish letters
    (ç ğ ı ö ş ü) stay inside tokens.
    """
    return _WORD_RE.findall(normalize_turkish(text or ""))


def _rouge_tokenizer(text: str) -> list[str]:
    # rouge_score's default tokenizer strips non-[a-z0-9] chars, destroying
    # Turkish letters; pass our Unicode-aware one instead.
    return _tokenize(text)


def _bleu_text(text: str) -> str:
    return " ".join(_tokenize(text))


def answer_length_words(text: str) -> int:
    """Number of words in an answer (citations stripped)."""
    return len(_tokenize(strip_citations(text or "")))


def _ngram_counts(tokens: list[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _sentence_bleu_fallback(predicted: str, expected: str, max_order: int = 4) -> float:
    pred_tokens = _tokenize(predicted)
    ref_tokens = _tokenize(expected)
    if not pred_tokens or not ref_tokens:
        return 0.0

    log_precisions = []
    for n in range(1, max_order + 1):
        pred_counts = _ngram_counts(pred_tokens, n)
        ref_counts = _ngram_counts(ref_tokens, n)
        total = sum(pred_counts.values())
        if total == 0:
            log_precisions.append(math.log(1e-9))
            continue
        overlap = sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
        # Standard clipped precision — no add-one smoothing (incompatible with BLEU)
        precision = overlap / total if total > 0 else 0.0
        log_precisions.append(math.log(precision) if precision > 0 else math.log(1e-9))

    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    if pred_len == 0:
        return 0.0
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - (ref_len / pred_len))
    return brevity_penalty * math.exp(sum(log_precisions) / max_order)


def _corpus_bleu_fallback(predictions: list[dict], max_order: int = 4) -> float:
    pred_tokens_all = [_tokenize(p["predicted"]) for p in predictions]
    ref_tokens_all = [_tokenize(p["expected"]) for p in predictions]
    if not pred_tokens_all or not ref_tokens_all:
        return 0.0

    log_precisions = []
    for n in range(1, max_order + 1):
        overlap = 0
        total = 0
        for pred_tokens, ref_tokens in zip(pred_tokens_all, ref_tokens_all):
            pred_counts = _ngram_counts(pred_tokens, n)
            ref_counts = _ngram_counts(ref_tokens, n)
            overlap += sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
            total += sum(pred_counts.values())
        # Standard clipped precision — no add-one smoothing (incompatible with BLEU)
        precision = overlap / total if total > 0 else 0.0
        log_precisions.append(math.log(precision) if precision > 0 else math.log(1e-9))

    pred_len = sum(len(tokens) for tokens in pred_tokens_all)
    ref_len = sum(len(tokens) for tokens in ref_tokens_all)
    if pred_len == 0:
        return 0.0
    brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1 - (ref_len / pred_len))
    return brevity_penalty * math.exp(sum(log_precisions) / max_order)


def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(curr[-1], prev[j]))
        prev = curr
    return prev[-1]


def _extract_citation_indices(predicted: str) -> list[int]:
    seen: set[int] = set()
    indices: list[int] = []
    for match in _CITATION_PATTERN.finditer(predicted):
        idx = int(match.group(1))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def _normalize_source(source: str) -> str:
    return normalize_turkish(source.strip()) if source else ""


def _cited_sources(predicted: str, retrieved_chunks: list[dict]) -> list[str]:
    cited_indices = _extract_citation_indices(predicted)
    cited_sources: list[str] = []
    for idx in cited_indices:
        zero_based = idx - 1
        if 0 <= zero_based < len(retrieved_chunks):
            source = retrieved_chunks[zero_based].get("source", "")
            if source:
                cited_sources.append(source)
    return cited_sources


def exact_match(predicted: str, expected: str) -> float:
    """Return 1.0 if the normalised expected text is a substring of predicted.

    Note on HMGS questions: HMGS is a Turkish bar-exam dataset whose questions
    ask which statement is true/false (çoktan seçmeli, multiple-choice style).
    The ``expected`` field contains the full text of the correct answer option,
    NOT a single letter (A/B/C/D), because the original CSV does not include
    the distractors.  As a result, EM is typically ~0: LLM responses are
    paraphrases, not verbatim copies of the answer text.
    Use ``answer_containment`` (recall-side token overlap) and ``token_f1`` as
    the primary lexical metrics for HMGS; EM is reported for completeness only.
    """
    pred_norm = normalize_turkish(predicted.strip())
    exp_norm = normalize_turkish(expected.strip())
    if not exp_norm:
        return 0.0
    return 1.0 if exp_norm in pred_norm else 0.0


def answer_containment(predicted: str, expected: str) -> float:
    """Recall-side token overlap: fraction of expected tokens present in predicted.

    More lenient than EM and more interpretable than token_f1 for HMGS-style
    questions where the expected answer is a factual statement and the LLM
    response is a full explanatory paragraph.  A high containment score
    indicates the model surfaced all key terms from the ground-truth answer,
    even if the phrasing differs.

    Args:
        predicted: The model-generated answer text.
        expected:  The ground-truth answer text.

    Returns:
        Float in [0, 1] — 1.0 means every expected token appeared in predicted.
    """
    pred_tokens = set(_tokenize(predicted))
    exp_tokens  = _tokenize(expected)
    if not exp_tokens:
        return 0.0
    if not pred_tokens:
        return 0.0
    matched = sum(1 for t in exp_tokens if t in pred_tokens)
    return matched / len(exp_tokens)


def token_f1(predicted: str, expected: str) -> float:
    pred_tokens = _tokenize(predicted)
    exp_tokens = _tokenize(expected)
    if not pred_tokens and not exp_tokens:
        return 1.0
    if not pred_tokens or not exp_tokens:
        return 0.0
    pred_counter = Counter(pred_tokens)
    exp_counter = Counter(exp_tokens)
    common = sum((pred_counter & exp_counter).values())
    precision = common / len(pred_tokens)
    recall = common / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bleu_score(predicted: str, expected: str) -> float:
    pred_norm = _bleu_text(predicted)
    ref_norm = _bleu_text(expected)
    if not pred_norm or not ref_norm:
        return 0.0
    if _USE_HF_EVALUATE:
        result = _BLEU_METRIC.compute(predictions=[pred_norm], references=[[ref_norm]])
        return float(result["bleu"])
    return _sentence_bleu_fallback(pred_norm, ref_norm)


def rouge_l_score(predicted: str, expected: str) -> float:
    pred_norm = _bleu_text(predicted)
    exp_norm = _bleu_text(expected)
    if _USE_HF_EVALUATE:
        if not pred_norm or not exp_norm:
            return 0.0
        result = _ROUGE_METRIC.compute(
            predictions=[pred_norm], references=[exp_norm], rouge_types=["rougeL"],
            tokenizer=_rouge_tokenizer,
        )
        return float(result["rougeL"])
    pred_tokens = _tokenize(predicted)
    exp_tokens = _tokenize(expected)
    if not pred_tokens or not exp_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, exp_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_qa_metrics(predicted: str, expected: str) -> dict:
    # Strip citation markers from predicted before lexical comparison;
    # citations inflate token count and suppress F1/EM vs. citation-free expected.
    pred_clean = strip_citations(predicted)
    return {
        "em": exact_match(pred_clean, expected),
        "f1": token_f1(pred_clean, expected),
        "bleu": bleu_score(pred_clean, expected),
        "rouge_l": rouge_l_score(pred_clean, expected),
        "answer_containment": answer_containment(pred_clean, expected),
    }


def compute_all_qa_metrics(predictions: list[dict]) -> dict:
    """
    predictions: list of {"predicted": str, "expected": str}
    Returns: {"em", "f1", "bleu", "rouge_l", "answer_containment", "num_samples"}
    """
    if not predictions:
        return {"em": 0.0, "f1": 0.0, "bleu": 0.0, "rouge_l": 0.0,
                "answer_containment": 0.0, "mean_answer_len_words": 0.0, "num_samples": 0}
    metrics = [compute_qa_metrics(p["predicted"], p["expected"]) for p in predictions]
    keys = ["em", "f1", "rouge_l", "answer_containment"]
    result = {k: sum(m[k] for m in metrics) / len(metrics) for k in keys}
    # Corpus-level BLEU via evaluate
    if _USE_HF_EVALUATE:
        preds_norm = [_bleu_text(strip_citations(p["predicted"])) for p in predictions]
        refs_norm = [[_bleu_text(p["expected"])] for p in predictions]
        bleu_result = _BLEU_METRIC.compute(predictions=preds_norm, references=refs_norm)
        result["bleu"] = float(bleu_result["bleu"])
    else:
        stripped = [{**p, "predicted": strip_citations(p["predicted"])} for p in predictions]
        result["bleu"] = _corpus_bleu_fallback(stripped)
    result["mean_answer_len_words"] = (
        sum(answer_length_words(p["predicted"]) for p in predictions) / len(predictions)
    )
    result["num_samples"] = len(predictions)
    return result


def compute_per_query_qa_metrics(predictions: list[dict]) -> list[dict]:
    """Per-query metrics for bootstrap CIs: query_id, em, f1, rouge_l,
    bleu (sentence-level), answer_containment, answer_len_words."""
    out = []
    for p in predictions:
        m = compute_qa_metrics(p["predicted"], p["expected"])
        m["query_id"] = p.get("query_id", "")
        m["answer_len_words"] = answer_length_words(p["predicted"])
        out.append(m)
    return out


def source_in_retrieved_context(retrieved_sources: list[str], expected_source: str) -> float:
    """Proxy metric: returns 1.0 if expected_source appears in retrieved context sources."""
    if not expected_source:
        return 0.0
    exp_norm = _normalize_source(expected_source)
    for s in retrieved_sources:
        if not s:
            continue
        if _normalize_source(s) == exp_norm:
            return 1.0
    return 0.0


def citation_accuracy(predicted: str, retrieved_chunks: list[dict], expected_source: str) -> float:
    """Returns 1.0 if the answer explicitly cites the expected source via [Kaynak N]."""
    if not expected_source:
        return 0.0
    exp_norm = _normalize_source(expected_source)
    for cited_source in _cited_sources(predicted, retrieved_chunks):
        if _normalize_source(cited_source) == exp_norm:
            return 1.0
    return 0.0


def citation_presence(predicted: str) -> float:
    """Returns 1.0 if the answer contains at least one [Kaynak N] style citation."""
    return 1.0 if _extract_citation_indices(predicted) else 0.0


def compute_all_qa_metrics_with_citation(predictions: list[dict]) -> dict:
    """
    predictions: list of {"predicted": str, "expected": str,
                           "retrieved_sources": list[str], "retrieved_chunks": list[dict],
                           "expected_source": str,
                           "predicted_native": str (optional)}
    ``predicted_native`` is the answer BEFORE utils.inject_citations; when
    present it yields ``citation_accuracy_native`` (did the model itself cite
    the gold source). ``predicted`` (possibly with injected citations) yields
    ``citation_accuracy_injected``, which mostly reflects retrieval overlap
    because injection is a token-overlap heuristic. The two are never merged;
    ``citation_accuracy_native`` is None when no native text was recorded.

    Returns: em, f1, bleu, rouge_l, answer_containment, mean_answer_len_words,
             citation_accuracy_native, citation_accuracy_injected,
             source_in_context_rate, citation_presence_rate_native/injected,
             num_samples
    """
    if not predictions:
        return {"em": 0.0, "f1": 0.0, "bleu": 0.0, "rouge_l": 0.0,
                "answer_containment": 0.0, "mean_answer_len_words": 0.0,
                "citation_accuracy_native": None, "citation_accuracy_injected": 0.0,
                "source_in_context_rate": 0.0,
                "citation_presence_rate_native": None,
                "citation_presence_rate_injected": 0.0, "num_samples": 0}
    result = compute_all_qa_metrics(predictions)
    n = len(predictions)
    inj, nat, nat_pres, inj_pres, proxy = [], [], [], [], []
    for p in predictions:
        chunks = p.get("retrieved_chunks", [])
        sources = p.get("retrieved_sources")
        if sources is None:
            sources = [c.get("source", "") for c in chunks]
        exp_src = p.get("expected_source", "")
        inj.append(citation_accuracy(p.get("predicted", ""), chunks, exp_src))
        inj_pres.append(citation_presence(p.get("predicted", "")))
        proxy.append(source_in_retrieved_context(sources, exp_src))
        if p.get("predicted_native") is not None:
            nat.append(citation_accuracy(p["predicted_native"], chunks, exp_src))
            nat_pres.append(citation_presence(p["predicted_native"]))
    # Generation diagnostics (recorded by the pipeline when available):
    # answers that hit max_tokens and answers whose invented "Soru:" turn was cut.
    for key, out in (("truncated", "truncated_rate"), ("runaway_cut", "runaway_cut_rate")):
        flags = [bool(p[key]) for p in predictions if key in p]
        result[out] = sum(flags) / len(flags) if flags else None
    result["citation_accuracy_injected"] = sum(inj) / n
    result["citation_presence_rate_injected"] = sum(inj_pres) / n
    result["citation_accuracy_native"] = sum(nat) / len(nat) if nat else None
    result["citation_presence_rate_native"] = sum(nat_pres) / len(nat_pres) if nat_pres else None
    result["source_in_context_rate"] = sum(proxy) / n
    return result
