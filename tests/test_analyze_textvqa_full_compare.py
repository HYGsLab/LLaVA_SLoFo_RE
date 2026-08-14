from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_textvqa_full_compare.py"
SPEC = importlib.util.spec_from_file_location("full_compare", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_interval_collapses_for_constant_differences() -> None:
    assert MODULE.paired_interval([2.5, 2.5, 2.5]) == [2.5, 2.5]


def test_default_prefixes_include_previous_and_full_scales() -> None:
    assert 1024 in MODULE.DEFAULT_PREFIXES
    assert MODULE.DEFAULT_PREFIXES[-1] == 5000


def test_end_to_end_comparison_counts_fractional_score_changes() -> None:
    baseline = [{"focus_answer": "a"}, {"focus_answer": "b"}]
    optimized = [{"focus_answer": "c"}, {"focus_answer": "b"}]
    result = MODULE.paired_branch_result(
        baseline, optimized, [0.0, 0.6], [1.0, 0.3], "focus_answer"
    )
    assert result["optimized_minus_baseline_percentage_points"] == 35.0
    assert result["answer_changed"] == 1
    assert result["score_improved"] == 1
    assert result["score_regressed"] == 1


def test_prompt_stratum_summary_keeps_the_stratum_size() -> None:
    baseline = [{"focus_answer": "a"}, {"focus_answer": "b"}]
    optimized = [{"focus_answer": "a"}, {"focus_answer": "c"}]
    scores = {branch: [0.0, 1.0] for branch in MODULE.BRANCHES}
    result = MODULE.prompt_stratum_summary(
        [1], baseline, optimized, scores, scores
    )
    assert result["sample_count"] == 1
    assert result["branch_accuracy_percent"]["baseline_A"]["focus_answer"] == 100.0
