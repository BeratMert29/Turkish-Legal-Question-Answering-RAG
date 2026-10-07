#!/usr/bin/env python3
"""
17_check_tlr_labels.py — Check the turkish_legal_rag gold article labels
against the law text (see data/tlr_labels.py) and rewrite the eval files.

Run after scripts/16 (and whenever the corpus or chunker changes).  The HF
label stays in ``madde_no_hf``; ``madde_no`` becomes the checked label and
``label_check`` records why.  HF-conflict rows whose label the check confirms
or corrects join the eval set; the rest stay in the conflicts file.

Usage:
    python scripts/17_check_tlr_labels.py
    python scripts/17_check_tlr_labels.py --csv path/to/combined_dataset.csv --dry-run
"""

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from data.data_processor import DataProcessor
from data.tlr_labels import apply_label_checks
from utils import read_jsonl

MAIN = config.TLR_DATA_PATH
CONFLICTS = config.TLR_PROCESSED_DIR / "qa_turkish_legal_rag.label_conflicts.jsonl"
REPORT = config.TLR_PROCESSED_DIR / "qa_turkish_legal_rag.report.json"


def _write_jsonl(rows, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", type=Path, default=config.RAW_DATA_PATH,
                    help="combined_dataset.csv (default: config.RAW_DATA_PATH)")
    ap.add_argument("--dry-run", action="store_true", help="print the report only")
    args = ap.parse_args(argv)

    rows = list(read_jsonl(MAIN))
    if CONFLICTS.exists():
        rows += list(read_jsonl(CONFLICTS))
    print(f"Rows: {len(rows)} (main + HF-label conflicts)")

    processor = DataProcessor(args.csv)
    processor.load_and_validate()
    chunks = list(processor.build_corpus_chunks(holdout=False))

    eval_rows, conflicts, report = apply_label_checks(rows, chunks)
    print(json.dumps({k: v for k, v in report.items() if k != "relabelled_query_ids"},
                     ensure_ascii=False, indent=2))
    for r in eval_rows:
        lc = r["label_check"]
        if lc["status"] == "relabelled":
            print(f"  {r['query_id']:<10} {r['source']:<32} {lc['from']:>6} -> "
                  f"{lc['madde_no']:<6} coverage {lc['from_coverage']:.2f} -> "
                  f"{lc['coverage']:.2f}  ({lc['reason']})")
    if args.dry_run:
        return

    _write_jsonl(eval_rows, MAIN)
    _write_jsonl(conflicts, CONFLICTS)
    full = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    full["label_check"] = report
    full["kept_after_label_check"] = len(eval_rows)
    REPORT.write_text(json.dumps(full, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Written: {MAIN} ({len(eval_rows)}), {CONFLICTS} ({len(conflicts)}), {REPORT}")


if __name__ == "__main__":
    main()
