#!/usr/bin/env python3
"""
18_judge_calibration.py — Check the LLM judge against human labels.

1. export: sample answers of one stage with the judge's answer-quality score
   into a CSV with an empty ``human_score`` column (0 / 0.5 / 1, same rubric
   as evaluation/llm_judge.py).
2. Fill ``human_score`` by hand (a legal expert, without looking at
   ``judge_score``; hide that column while labelling).
3. score: agreement between judge and human -- exact agreement, Cohen's
   kappa (unweighted and linear-weighted), mean judge vs mean human (bias).

Usage:
    python scripts/18_judge_calibration.py export \\
        --stage-dir results/turkish_legal_rag/base --n 50 --out judge_calibration.csv
    python scripts/18_judge_calibration.py score --csv judge_calibration.csv
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config
from evaluation.stats import cohen_kappa
from utils import read_jsonl

LABELS = [0.0, 0.5, 1.0]
FIELDS = ["query_id", "question", "expected", "predicted", "judge_score", "human_score"]


def latest_judge_scores(stage_dir: Path, metric: str = "answer") -> dict[str, float | None]:
    """query_id -> judge score from the newest judge_raw_<metric>_*.jsonl."""
    files = sorted(stage_dir.glob(f"judge_raw_{metric}_*.jsonl"))
    if not files:
        sys.exit(f"ERROR: no judge_raw_{metric}_*.jsonl in {stage_dir}")
    return {str(r["query_id"]): r.get("score") for r in read_jsonl(files[-1])}


def export(stage_dir: Path, n: int, out: Path, seed: int) -> None:
    preds = [p for p in read_jsonl(stage_dir / "predictions.jsonl") if p.get("predicted")]
    judge = latest_judge_scores(stage_dir)
    preds = [p for p in preds if str(p["query_id"]) in judge]
    sample = random.Random(seed).sample(preds, min(n, len(preds)))
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for p in sample:
            w.writerow({
                "query_id": p["query_id"], "question": p.get("question", ""),
                "expected": p.get("expected", ""),
                "predicted": p.get("predicted_native") or p["predicted"],
                "judge_score": judge[str(p["query_id"])], "human_score": "",
            })
    print(f"Wrote {len(sample)} rows to {out}; fill human_score with 0, 0.5 or 1.")


def _label(value) -> float | None:
    try:
        v = float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return None
    return min(LABELS, key=lambda lab: abs(lab - v))


def score(path: Path) -> dict:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    pairs = [(_label(r["judge_score"]), _label(r["human_score"])) for r in rows]
    pairs = [(j, h) for j, h in pairs if j is not None and h is not None]
    if not pairs:
        sys.exit("ERROR: no row has both judge_score and human_score")
    judge, human = zip(*pairs)
    result = {
        "n": len(pairs),
        "exact_agreement": sum(j == h for j, h in pairs) / len(pairs),
        "kappa": cohen_kappa(judge, human, LABELS),
        "kappa_linear": cohen_kappa(judge, human, LABELS, "linear"),
        "mean_judge": sum(judge) / len(judge),
        "mean_human": sum(human) / len(human),
        "judge_model": config.LLM_JUDGE_MODEL,
    }
    print(json.dumps(result, indent=2))
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="LLM-judge vs human agreement")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--stage-dir", type=Path, required=True)
    e.add_argument("--n", type=int, default=50)
    e.add_argument("--out", type=Path, default=Path("judge_calibration.csv"))
    e.add_argument("--seed", type=int, default=config.SEED)
    s = sub.add_parser("score")
    s.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "export":
        export(args.stage_dir, args.n, args.out, args.seed)
    else:
        score(args.csv)


if __name__ == "__main__":
    main()
