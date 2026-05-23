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


def _tokenize(text: str) -> list[str]:
    return normalize_turkish(text).split()


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
    return 1.0 if normalize_turkish(predicted.strip()) == normalize_turkish(expected.strip()) else 0.0


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
    pred_norm = normalize_turkish(predicted)
    ref_norm = normalize_turkish(expected)
    if not pred_norm or not ref_norm:
        return 0.0
    if _USE_HF_EVALUATE:
        result = _BLEU_METRIC.compute(predictions=[pred_norm], references=[[ref_norm]])
        return float(result["bleu"])
    return _sentence_bleu_fallback(pred_norm, ref_norm)


def rouge_l_score(predicted: str, expected: str) -> float:
    pred_norm = normalize_turkish(predicted)
    exp_norm = normalize_turkish(expected)
    if _USE_HF_EVALUATE:
        result = _ROUGE_METRIC.compute(
            predictions=[pred_norm], references=[exp_norm], rouge_types=["rougeL"]
        )
        return float(result["rougeL"])
    pred_tokens = pred_norm.split()
    exp_tokens = exp_norm.split()
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
    }


def compute_all_qa_metrics(predictions: list[dict]) -> dict:
    """
    predictions: list of {"predicted": str, "expected": str}
    Returns: {"em", "f1", "bleu", "rouge_l", "num_samples"}
    """
    if not predictions:
        return {"em": 0.0, "f1": 0.0, "bleu": 0.0, "rouge_l": 0.0, "num_samples": 0}
    metrics = [compute_qa_metrics(p["predicted"], p["expected"]) for p in predictions]
    keys = ["em", "f1", "rouge_l"]
    result = {k: sum(m[k] for m in metrics) / len(metrics) for k in keys}
    # Corpus-level BLEU via evaluate
    if _USE_HF_EVALUATE:
        preds_norm = [normalize_turkish(strip_citations(p["predicted"])) for p in predictions]
        refs_norm = [[normalize_turkish(p["expected"])] for p in predictions]
        bleu_result = _BLEU_METRIC.compute(predictions=preds_norm, references=refs_norm)
        result["bleu"] = float(bleu_result["bleu"])
    else:
        stripped = [{**p, "predicted": strip_citations(p["predicted"])} for p in predictions]
        result["bleu"] = _corpus_bleu_fallback(stripped)
    result["num_samples"] = len(predictions)
    return result


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
                           "expected_source": str}
    Returns: averaged em, f1, bleu, rouge_l, citation_accuracy,
             source_in_context_rate, citation_presence_rate, num_samples
    """
    if not predictions:
        return {"em": 0.0, "f1": 0.0, "bleu": 0.0, "rouge_l": 0.0,
                "citation_accuracy": 0.0, "source_in_context_rate": 0.0,
                "citation_presence_rate": 0.0, "num_samples": 0}
    qa_metrics = [compute_qa_metrics(p["predicted"], p["expected"]) for p in predictions]
    cite_scores = []
    source_proxy_scores = []
    citation_presence_scores = []
    for p in predictions:
        retrieved_chunks = p.get("retrieved_chunks", [])
        retrieved_sources = p.get("retrieved_sources")
        if retrieved_sources is None:
            retrieved_sources = [c.get("source", "") for c in retrieved_chunks]
        cite_scores.append(
            citation_accuracy(p.get("predicted", ""), retrieved_chunks, p.get("expected_source", ""))
        )
        source_proxy_scores.append(
            source_in_retrieved_context(retrieved_sources, p.get("expected_source", ""))
        )
        citation_presence_scores.append(citation_presence(p.get("predicted", "")))
    n = len(predictions)
    keys = ["em", "f1", "rouge_l"]
    result = {k: sum(m[k] for m in qa_metrics) / n for k in keys}
    # Corpus-level BLEU via evaluate
    if _USE_HF_EVALUATE:
        preds_norm = [normalize_turkish(strip_citations(p["predicted"])) for p in predictions]
        refs_norm = [[normalize_turkish(p["expected"])] for p in predictions]
        bleu_result = _BLEU_METRIC.compute(predictions=preds_norm, references=refs_norm)
        result["bleu"] = float(bleu_result["bleu"])
    else:
        stripped = [{**p, "predicted": strip_citations(p["predicted"])} for p in predictions]
        result["bleu"] = _corpus_bleu_fallback(stripped)
    result["citation_accuracy"] = sum(cite_scores) / n
    result["source_in_context_rate"] = sum(source_proxy_scores) / n
    result["citation_presence_rate"] = sum(citation_presence_scores) / n
    result["num_samples"] = n
    return result
