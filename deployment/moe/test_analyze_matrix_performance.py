from __future__ import annotations

import sys
from pathlib import Path


MOE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MOE_DIR))

from analyze_matrix_performance import _performance_gate_passed  # noqa: E402


def test_performance_gate_requires_exact_hidden_and_non_regression() -> None:
    assert _performance_gate_passed(
        hidden_exact=True,
        non_regression=True,
    )
    assert not _performance_gate_passed(
        hidden_exact=True,
        non_regression=False,
    )
    assert not _performance_gate_passed(
        hidden_exact=False,
        non_regression=True,
    )
    assert not _performance_gate_passed(
        hidden_exact=False,
        non_regression=False,
    )
