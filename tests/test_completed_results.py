"""Regression checks for the real archived Colab run; no model/GPU required."""
import json
import math

import pandas as pd
import yaml

from src.dataset_utils import load_and_validate_splits
from src.evaluation import compare_prediction_files
from src.runtime import ROOT, dataset_hashes, validate_run


def test_completed_run_provenance_and_dataset():
    provenance = ROOT / "results/provenance"
    run = json.loads((provenance / "run_metadata.json").read_text())
    config = yaml.safe_load((provenance / "training_config.yaml").read_text())
    validate_run(run, config, dataset_hashes())
    assert run["training_complete"]
    state = json.loads((provenance / "trainer_state.json").read_text())
    assert state["epoch"] == 3 and state["global_step"] == 105
    test = load_and_validate_splits(ROOT / "data")["test"]
    for name in ("base_model_results.csv", "fine_tuned_results.csv", "comparison.csv"):
        frame = pd.read_csv(ROOT / "results" / name, keep_default_na=False)
        assert len(frame) == len(test) == 70
        for actual, expected in zip(frame.to_dict("records"), test, strict=True):
            for column in ("fact_id", "category", "instruction"):
                assert actual[column] == expected[column]
            assert actual["expected_response"] == expected["response"]


def test_completed_metrics_reproduce(tmp_path):
    results = ROOT / "results"
    frame, recomputed = compare_prediction_files(
        results / "base_model_results.csv", results / "fine_tuned_results.csv",
        tmp_path / "comparison.csv", tmp_path / "metrics.json",
    )
    archived = json.loads((results / "metrics.json").read_text())
    def compare(expected, actual):
        assert type(expected) is type(actual)
        if isinstance(expected, dict):
            assert expected.keys() == actual.keys()
            for key in expected:
                compare(expected[key], actual[key])
        elif isinstance(expected, float):
            assert math.isclose(expected, actual, rel_tol=0, abs_tol=1e-12)
        else:
            assert expected == actual
    for key in archived.keys() - {"created_at_utc"}:
        compare(archived[key], recomputed[key])
    pd.testing.assert_frame_equal(
        pd.read_csv(results / "comparison.csv", keep_default_na=False),
        frame, check_exact=False, atol=1e-12, rtol=0,
    )
