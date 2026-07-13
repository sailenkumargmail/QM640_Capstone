"""Smoke test: run the full pipeline end-to-end on a tiny synthetic dataset
to catch integration breakages (wiring between EDA / training / tuning /
SHAP / fairness stages), without the cost of a full-scale run.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


@pytest.fixture
def tiny_config(tmp_path, monkeypatch):
    from dac.config import CONFIG

    monkeypatch.setitem(CONFIG["data"]["home_credit"], "n_synthetic_rows", 400)
    monkeypatch.setitem(CONFIG["data"]["hmda"], "n_synthetic_rows", 300)
    monkeypatch.setitem(CONFIG["tuning"], "n_trials", 2)
    monkeypatch.setitem(CONFIG["tuning"], "cv_folds", 2)

    # Redirect all outputs to a scratch tmp_path so this test never touches
    # the real data/ or reports/ directories, and never hits the real-data
    # cache from a full run.
    for key in ["raw_dir", "processed_dir", "models_dir", "reports_dir", "figures_dir", "metrics_dir"]:
        new_dir = tmp_path / key
        new_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setitem(CONFIG["paths"], key, new_dir)

    return CONFIG


def test_full_pipeline_smoke(tiny_config):
    import run_pipeline

    summary = run_pipeline.main(quick=True)

    assert summary["home_credit_is_synthetic"] is True
    assert summary["hmda_is_synthetic"] is True
    assert "best_model" in summary
    assert 0 <= summary["tuned_metrics"][summary["best_model"]]["roc_auc"] <= 1
    assert "fairness_pre_mitigation" in summary
    assert "fairness_post_mitigation" in summary
    assert "hmda_results" in summary
    assert "fairness" in summary["hmda_results"]
