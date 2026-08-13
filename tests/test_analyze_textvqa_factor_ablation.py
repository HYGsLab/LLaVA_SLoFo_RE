from __future__ import annotations

import ast
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_textvqa_factor_ablation.py"


def test_factor_analyzer_accepts_partial_comparison_chain() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    source = ast.unparse(tree)
    assert "active_comparisons" in source
    assert "No complete factor comparison is available" in source
    assert "Missing factor cells" not in source
