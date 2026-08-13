from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_textvqa_nested_convergence.py"
SPEC = importlib.util.spec_from_file_location("nested_convergence", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_ci_collapses_for_constant_differences() -> None:
    assert MODULE.paired_ci([1.5, 1.5, 1.5]) == [1.5, 1.5]


def test_comparisons_form_the_planned_four_cell_chain() -> None:
    assert [name for name, _, _ in MODULE.COMPARISONS] == [
        "minmax_vs_raw_1token",
        "tokens_8_vs_1_minmax",
        "topk5_vs_single_8token_minmax",
    ]
