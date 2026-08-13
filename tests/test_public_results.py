from pathlib import Path

import pandas as pd

RESULTS = Path("results/intervenesim-x")
VALUE_RESULTS = Path("results/intervenesim-value")


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


def test_frozen_value_gate_beats_risk_at_primary_quarter_budget() -> None:
    summary = pd.read_csv(VALUE_RESULTS / "gate_summary.csv")
    quarter = summary.loc[summary["target_budget"] == 0.25].set_index("method")
    assert abs(quarter.loc["value_gate", "intervention_rate"] - 0.2333333333333333) < 1e-9
    assert abs(quarter.loc["risk_gate", "intervention_rate"] - 0.2458333333333333) < 1e-9
    assert quarter.loc["value_gate", "success_rate"] > quarter.loc["risk_gate", "success_rate"]
    assert (
        quarter.loc["value_gate", "request_precision"]
        > quarter.loc["risk_gate", "request_precision"]
    )


def test_exploratory_value_advantage_is_positive_for_every_seed() -> None:
    paired = pd.read_csv(VALUE_RESULTS / "crossval_paired.csv")
    against_risk = paired.loc[paired["reference"] == "risk_gate"]
    assert (against_risk["policy_seeds"] == 5).all()
    assert (against_risk["positive_seeds"] == 5).all()
    assert (against_risk["mean_delta"] > 0).all()
