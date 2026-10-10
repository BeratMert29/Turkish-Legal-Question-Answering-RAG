"""scripts/18_judge_calibration.py export -> score round trip."""

import csv
import importlib.util
import json
import pathlib

_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "18_judge_calibration.py"
spec = importlib.util.spec_from_file_location("judge_calibration", _PATH)
s18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s18)


def test_export_then_score(tmp_path):
    stage = tmp_path / "base"
    stage.mkdir()
    preds = [{"query_id": f"q{i}", "question": "S", "expected": "E", "predicted": f"P{i}"}
             for i in range(4)]
    (stage / "predictions.jsonl").write_text(
        "\n".join(json.dumps(p) for p in preds), encoding="utf-8")
    (stage / "judge_raw_answer_20260101T000000-ab.jsonl").write_text(
        "\n".join(json.dumps({"query_id": f"q{i}", "score": s})
                  for i, s in enumerate([1.0, 0.5, 0.0, 1.0])), encoding="utf-8")
    out = tmp_path / "calib.csv"
    s18.export(stage, 4, out, seed=1)
    with open(out, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:  # human agrees with the judge everywhere
        r["human_score"] = r["judge_score"]
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=s18.FIELDS)
        w.writeheader()
        w.writerows(rows)
    res = s18.score(out)
    assert res["n"] == 4 and res["exact_agreement"] == 1.0 and res["kappa"] == 1.0
