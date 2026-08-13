import pandas as pd

from intervenesim.reporting import paired_comparison
from intervenesim.video import select_comparison_seed


def test_paired_comparison_counts_matched_outcomes() -> None:
    episodes = pd.DataFrame(
        [
            {"condition": "baseline", "disturbance": "noise", "seed": 1, "success": False},
            {"condition": "recovery", "disturbance": "noise", "seed": 1, "success": True},
            {"condition": "baseline", "disturbance": "noise", "seed": 2, "success": True},
            {"condition": "recovery", "disturbance": "noise", "seed": 2, "success": False},
            {"condition": "baseline", "disturbance": "noise", "seed": 3, "success": False},
            {"condition": "recovery", "disturbance": "noise", "seed": 3, "success": True},
            {"condition": "baseline", "disturbance": "noise", "seed": 4, "success": True},
            {"condition": "recovery", "disturbance": "noise", "seed": 4, "success": True},
        ]
    )
    result = paired_comparison(episodes, "recovery", "baseline")
    assert result["episodes"] == 4
    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["ties"] == 1
    assert result["delta"] == 0.25


def test_select_comparison_seed_finds_baseline_failure() -> None:
    episodes = pd.DataFrame(
        [
            {"condition": "baseline", "disturbance": "slip", "seed": 4, "success": True},
            {
                "condition": "recovery_data",
                "disturbance": "slip",
                "seed": 4,
                "success": True,
            },
            {"condition": "baseline", "disturbance": "slip", "seed": 9, "success": False},
            {
                "condition": "recovery_data",
                "disturbance": "slip",
                "seed": 9,
                "success": True,
            },
        ]
    )
    assert select_comparison_seed(episodes, "slip") == 9
