import pandas as pd

from intervenesim.helping import summarize_help_seeking


def test_help_summary_reports_success_and_intervention_rates() -> None:
    episodes = pd.DataFrame(
        [
            {
                "help_mode": "learned",
                "disturbance": "noise",
                "success": True,
                "help_requested": True,
                "help_step": 10,
                "steps": 50,
            },
            {
                "help_mode": "learned",
                "disturbance": "noise",
                "success": False,
                "help_requested": False,
                "help_step": None,
                "steps": 100,
            },
        ]
    )
    summary = summarize_help_seeking(episodes).iloc[0]
    assert summary["success_rate"] == 0.5
    assert summary["intervention_rate"] == 0.5
    assert summary["mean_request_step"] == 10
