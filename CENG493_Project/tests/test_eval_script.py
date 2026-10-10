"""Helpers of scripts/14_eval_all_stages.py (output layout, summary merge)."""

import importlib.util
import json
import pathlib

import pytest

_PATH = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "14_eval_all_stages.py"


@pytest.fixture(scope="module")
def s14():
    spec = importlib.util.spec_from_file_location("eval_all_stages", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_dir_name_separates_eval_sets_and_limits(s14):
    assert s14.run_dir_name("turkish_legal_rag", None) == "turkish_legal_rag"
    assert s14.run_dir_name("hmgs", 10) == "hmgs_limit10"
    assert s14.run_dir_name("kaggle", None, "data/my_bench.json") == "external_my_bench"


def test_merge_summary_keeps_stages_not_rerun(s14, tmp_path):
    path = tmp_path / "ablation_summary.json"
    path.write_text(json.dumps({"stages": {"base": {"v": 1}, "llm_ft": {"v": 1}}}),
                    encoding="utf-8")
    merged = s14.merge_summary(path, {"llm_ft": {"v": 2}})
    assert merged == {"base": {"v": 1}, "llm_ft": {"v": 2}}


def test_merge_summary_without_existing_file(s14, tmp_path):
    assert s14.merge_summary(tmp_path / "none.json", {"base": {}}) == {"base": {}}


def test_parse_args_flags(s14):
    args = s14._parse_args(["--stages", "base", "llm_ft", "--no-judge", "--limit", "5"])
    assert args.stages == "base,llm_ft" and args.no_judge and args.limit == 5
