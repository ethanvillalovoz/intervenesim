from pathlib import Path

import pandas as pd

RESULTS = Path("results/intervenesim-x")


def test_frozen_multiseed_result_matches_reported_claim() -> None:
    rates = pd.read_csv(RESULTS / "autonomous_seed_rates.csv")
    disturbed = rates.loc[rates["disturbed"].astype(str).str.lower() == "true"]
    pivot = disturbed.pivot(index="training_seed", columns="condition", values="success_rate")
    differences = pivot["recovery_bc"] - pivot["more_demos"]
    assert len(differences) == 5
    assert (differences > 0).all()
    assert abs(differences.mean() - 0.228125) < 1e-9


def test_frozen_help_operating_point_is_selective() -> None:
    sweep = pd.read_csv(RESULTS / "help_threshold_sweep.csv").set_index("threshold")
    operating_point = sweep.loc[0.97]
    assert abs(operating_point["disturbed_success_rate"] - 0.7578125) < 1e-9
    assert abs(operating_point["disturbed_intervention_rate"] - 0.6484375) < 1e-9
    assert abs(operating_point["nominal_intervention_rate"] - 0.03125) < 1e-9
