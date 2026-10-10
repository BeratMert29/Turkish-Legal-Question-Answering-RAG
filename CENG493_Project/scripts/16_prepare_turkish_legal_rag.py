#!/usr/bin/env python3
"""
16_prepare_turkish_legal_rag.py — Build the ``turkish_legal_rag`` eval set.

Source: HuggingFace ``mtntasci/turkish-legal-rag`` (config ``qa_benchmark``,
split ``test``, CC-BY-4.0).  Filters applied, in order:

  1. source_origin       keep only ``kaggle_batuhankalem`` rows (the templated
                         ``None``-origin rows are low quality)
  2. unknown_law         drop rows whose law has no chunks in our corpus
  3. train_leakage       drop questions present in any
                         ``qa_train*.jsonl`` fine-tuning file

``madde_no`` ("3-") is normalised to the corpus convention ("3", "183-a",
"ek-3", "gecici-2").  The HF field never carries the section, so a plain
number is promoted to ``gecici-N`` / ``ek-N`` when the question or answer
names "Geçici Madde N" / "Ek Madde N" with that number.

Label conflicts: rows whose question/answer names article numbers none of
which equals the label (e.g. answer cites "TCK 151" but label is 150) are
flagged ``label_conflict: true``.  They are NOT silently fixed and NOT in the
default set: they go to ``qa_turkish_legal_rag.label_conflicts.jsonl`` and the
report records the count.

Next step: run scripts/17_check_tlr_labels.py, which checks every HF label
against the law text (many are off by 1-3 articles), records the corrected
label next to the original (``madde_no_hf``) and re-admits conflict rows whose
label it can confirm or correct.

Usage:
    python scripts/16_prepare_turkish_legal_rag.py
    python scripts/16_prepare_turkish_legal_rag.py --input rows.json

Output (under results/processed_data/):
    qa_turkish_legal_rag.jsonl         same schema as qa_hmgs.jsonl
                                       + madde_no, hf_row_id
    qa_turkish_legal_rag.report.json   per-reason / per-law counts
"""

import argparse
import collections
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from data.data_processor import DataProcessor, normalize_question
from utils import read_jsonl

HF_DATASET = "mtntasci/turkish-legal-rag"
HF_CONFIG = "qa_benchmark"
HF_SPLIT = "test"
KEEP_ORIGIN = "kaggle_batuhankalem"

_ROWS_API = "https://datasets-server.huggingface.co/rows"
_PAGE = 100



def normalize_madde_no(raw) -> "str | None":
    """Map an HF ``madde_no`` to the corpus convention.

    ``"3-"`` -> ``"3"``, ``"183/A"`` -> ``"183-a"``, ``"Ek Madde 3"`` ->
    ``"ek-3"``, ``"Geçici 2-"`` -> ``"gecici-2"``.  ``None`` when unusable.
    """
    if raw is None:
        return None
    s = _fold(str(raw).strip())
    s = re.sub(r"\bmadde\b", " ", s).strip(" -.")
    m = re.fullmatch(r"(ek[\s-]*gecici|ek|gecici)?[\s-]*(\d+)(?:[\s/-]*([a-z]))?", s)
    if not m:
        return None
    prefix, num, letter = m.groups()
    if prefix:
        prefix = "ekgecici" if "gecici" in prefix and prefix.startswith("ek") else prefix
    out = num + (f"-{letter}" if letter else "")
    return f"{prefix}-{out}" if prefix else out


def _fold(text: str) -> str:
    """Turkish-lowercase (İ->i, I->ı before lower(), so "GEÇİCİ" does not
    gain combining dots) and fold to ASCII letters."""
    from utils import normalize_turkish
    return normalize_turkish(text).translate(str.maketrans("çğıöşü", "cgiosu"))


_SECTION_RE = re.compile(r"(?i)\b(ek\s+geçici|ek|geçici)\s+madde\s+(\d+)")
_ARTICLE_MENTION_RES = (
    re.compile(r"(?i)(\d{1,4})\s*(?:\.|inci|ıncı|nci|ncı|üncü|uncu)?\s*madde"),
    re.compile(r"(?i)madde\s+(\d{1,4})"),
)


def refine_madde_no(madde_no, question: str, answer: str) -> "str | None":
    """Promote a plain article number to ``gecici-N`` / ``ek-N``.

    The HF ``madde_no`` drops the section ("Geçici Madde 5" -> "5").  When the
    question/answer says "Geçici Madde N" with the same N, use the section form.
    """
    if madde_no is None or not re.fullmatch(r"\d+(?:-[a-z])?", madde_no):
        return madde_no
    base = re.match(r"\d+", madde_no).group()
    for m in _SECTION_RE.finditer(f"{question} {answer}"):
        if m.group(2) != base:
            continue
        kind = _fold(m.group(1))
        if "gecici" in kind:
            return f"{'ekgecici' if kind.startswith('ek') else 'gecici'}-{madde_no}"
        return f"ek-{madde_no}"
    return madde_no


def find_label_conflict(madde_no, question: str, answer: str) -> bool:
    """True when the text names article numbers but none equals the label."""
    if not madde_no:
        return False
    text = f"{question} {answer}"
    named = {n for rx in _ARTICLE_MENTION_RES for n in rx.findall(text)}
    if not named:
        return False
    label = re.search(r"\d+", madde_no).group()
    return label not in named


def corpus_sources(metadata_path: Path) -> set:
    """Laws that have chunks in our corpus."""
    sources = set()
    if metadata_path.exists():
        sources = {r.get("source") for r in read_jsonl(metadata_path)}
        sources.discard(None)
    return sources or set(config.HMGS_SOURCE_MAP.values())


def resolve_source(kaynak, known: set) -> "str | None":
    """Map HF ``kaynak`` to a corpus source name, or None if not in corpus."""
    name = config.TLR_SOURCE_ALIASES.get(kaynak, kaynak)
    if name in known:
        return name
    name = config.HMGS_SOURCE_MAP.get(kaynak)
    return name if name in known else None


def load_training_questions(processed_dir: Path) -> set:
    out = set()
    for path in sorted(processed_dir.glob("qa_train*.jsonl")):
        out.update(normalize_question(r.get("question", "")) for r in read_jsonl(path))
    return out


def build_examples(rows, known_sources, train_questions, eval_questions=frozenset()):
    """Apply the filters; return ``(examples, report)``.

    Rows flagged ``label_conflict`` are kept out of ``examples``; they are
    returned under ``report["label_conflict_rows"]`` (``main`` writes them to
    their own file and removes that key from the printed report).
    """
    drops = collections.Counter()
    kept_per_law = collections.Counter()
    examples = []
    conflicts = []
    eval_overlap = 0
    for row in rows:
        if row.get("source_origin") != KEEP_ORIGIN:
            drops["source_origin"] += 1
            continue
        source = resolve_source(row.get("kaynak"), known_sources)
        if source is None:
            drops["unknown_law"] += 1
            continue
        question = row.get("soru") or ""
        nq = normalize_question(question)
        if nq in train_questions:
            drops["train_leakage"] += 1
            continue
        if nq in eval_questions:
            eval_overlap += 1
        row_id = str(row.get("row_id", "")).split(".")[0]
        answer = row.get("cevap") or ""
        madde_no = refine_madde_no(normalize_madde_no(row.get("madde_no")), question, answer)
        if madde_no is None:
            drops["_kept_without_madde_no"] += 1
        conflict = find_label_conflict(madde_no, question, answer)
        target = conflicts if conflict else examples
        target.append({
            "query_id": f"tlr_{row_id}",
            "question": question,
            "answer": answer,
            "context": "",
            "source": source,
            "data_type": "",
            "madde_no": madde_no,
            "hf_row_id": row_id,
            "label_conflict": conflict,
        })
        if not conflict:
            kept_per_law[source] += 1
    kept_no_madde = drops.pop("_kept_without_madde_no", 0)
    report = {
        "dataset": f"{HF_DATASET}/{HF_CONFIG}/{HF_SPLIT}",
        "license": "CC-BY-4.0",
        "total_rows": len(rows),
        "kept": len(examples),
        "label_conflict": len(conflicts),
        "label_conflict_query_ids": [c["query_id"] for c in conflicts],
        "label_conflict_rows": conflicts,
        "dropped": {k: drops.get(k, 0) for k in
                    ("source_origin", "unknown_law", "train_leakage")},
        "kept_without_madde_no": kept_no_madde,
        "kept_per_law": dict(kept_per_law.most_common()),
        "overlap_with_qa_eval_kept": eval_overlap,
    }
    return examples, report


def fetch_rows() -> list:
    """Download all rows via ``datasets`` or the datasets-server REST API."""
    try:
        from datasets import load_dataset
        return [dict(r) for r in load_dataset(HF_DATASET, HF_CONFIG, split=HF_SPLIT)]
    except Exception as exc:  # noqa: BLE001 — fall back to REST
        print(f"  datasets library unavailable ({exc}); using rows API")
    rows, offset = [], 0
    while True:
        qs = urllib.parse.urlencode({
            "dataset": HF_DATASET, "config": HF_CONFIG, "split": HF_SPLIT,
            "offset": offset, "length": _PAGE,
        })
        with urllib.request.urlopen(f"{_ROWS_API}?{qs}", timeout=60) as resp:
            payload = json.load(resp)
        page = [r["row"] for r in payload.get("rows", [])]
        rows.extend(page)
        offset += len(page)
        if not page or offset >= payload.get("num_rows_total", 0):
            return rows


def _load_questions(path: Path) -> set:
    if not path.exists():
        return set()
    return {normalize_question(r.get("question", "")) for r in read_jsonl(path)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", type=Path, default=None,
                    help="Local JSON (list of row dicts); default: download from HF")
    ap.add_argument("--processed-dir", type=Path, default=config.TLR_PROCESSED_DIR)
    ap.add_argument("--metadata", type=Path, default=config.TLR_METADATA_PATH)
    args = ap.parse_args(argv)

    if args.input:
        with open(args.input, encoding="utf-8") as fh:
            rows = json.load(fh)
    else:
        rows = fetch_rows()

    examples, report = build_examples(
        rows,
        corpus_sources(args.metadata),
        load_training_questions(args.processed_dir),
        _load_questions(args.processed_dir / config.QA_GOLD_FILE),
    )

    out = args.processed_dir / config.TLR_GOLD_FILE
    conflicts = report.pop("label_conflict_rows")
    DataProcessor.save_jsonl(examples, out)
    DataProcessor.save_jsonl(conflicts, out.with_suffix(".label_conflicts.jsonl"))
    report_path = out.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(examples)} examples -> {out}\nReport -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
