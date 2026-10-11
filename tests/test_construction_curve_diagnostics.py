from embodied_skill_composer.construction.workbench_evidence import _curve_point


def test_learning_curve_keeps_actual_success_distinct_from_rollout_endings() -> None:
    assert _curve_point(
        {
            "transitions": "200",
            "rollout_terminal_fraction": "1.0",
            "rollout_structure_completion_rate": "0.4",
            "rollout_success_fraction": "0.0",
            "rollout_truncation_fraction": "1.0",
            "rollout_failure_probability_mean": "0.175",
        }
    ) == {
        "transitions": 200,
        "rollout_terminal_fraction": 1.0,
        "rollout_structure_completion_rate": 0.4,
        "rollout_success_fraction": 0.0,
        "rollout_truncation_fraction": 1.0,
        "rollout_failure_probability_mean": 0.175,
    }


def test_historical_missing_completion_metrics_remain_unknown() -> None:
    assert _curve_point(
        {
            "transitions": 100,
            "rollout_terminal_fraction": "0.99",
            "rollout_structure_completion_rate": "",
            "rollout_success_fraction": None,
        }
    ) == {"transitions": 100, "rollout_terminal_fraction": 0.99}
