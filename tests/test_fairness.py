import numpy as np
import pandas as pd

from dac.fairness.mitigation import build_joint_group, compute_reweighing_weights
from dac.fairness.significance import bootstrap_auc_test, two_proportion_ztest


def test_reweighing_weights_normalize_to_mean_one():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.integers(0, 2, size=500))
    g = pd.Series(rng.choice(["A", "B"], size=500))
    weights = compute_reweighing_weights(y, g)
    assert weights.shape[0] == 500
    assert abs(weights.mean() - 1.0) < 1e-6
    assert (weights > 0).all()


def test_reweighing_upweights_underrepresented_positive_group():
    # Group "B" is rare AND rarely positive -> should get an upweight
    # for its (B, 1) rows relative to the (A, 1) rows.
    y = pd.Series([0] * 80 + [1] * 20 + [0] * 95 + [1] * 5)
    g = pd.Series(["A"] * 100 + ["B"] * 100)
    weights = compute_reweighing_weights(y, g)
    df = pd.DataFrame({"y": y, "g": g, "w": weights})
    mean_w_b_pos = df[(df.g == "B") & (df.y == 1)]["w"].mean()
    mean_w_a_pos = df[(df.g == "A") & (df.y == 1)]["w"].mean()
    assert mean_w_b_pos > mean_w_a_pos


def test_build_joint_group_concatenates_attributes():
    df = pd.DataFrame({"SEX": [1, 1, 2, 2], "AGE_GROUP": ["<25", "25-34", "<25", "25-34"]})
    joint = build_joint_group(df, ["SEX", "AGE_GROUP"])
    assert list(joint) == ["1_<25", "1_25-34", "2_<25", "2_25-34"]
    assert joint.name == "SEX_AGE_GROUP"


def test_build_joint_group_requires_two_attributes():
    df = pd.DataFrame({"SEX": [1, 2]})
    try:
        build_joint_group(df, ["SEX"])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_two_proportion_ztest_identical_proportions_not_significant():
    result = two_proportion_ztest(count1=50, n1=100, count2=50, n2=100, alternative="two-sided")
    assert result["p_value"] > 0.9
    assert abs(result["z_statistic"]) < 1e-9


def test_two_proportion_ztest_large_gap_is_significant():
    # 90% selection rate vs 50% selection rate, large groups -> should be
    # a highly significant one-tailed difference.
    result = two_proportion_ztest(count1=900, n1=1000, count2=500, n2=1000, alternative="larger")
    assert result["diff"] > 0
    assert result["p_value"] < 0.001


def test_bootstrap_auc_test_identical_scores_not_significant():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=300)
    proba = rng.random(300)
    result = bootstrap_auc_test(y_true, proba, proba, n_boot=500, seed=1, alternative="two-sided")
    assert result["observed_diff"] == 0.0
    assert result["p_value"] == 1.0


def test_bootstrap_auc_test_detects_clearly_better_model():
    rng = np.random.default_rng(0)
    n = 400
    y_true = rng.integers(0, 2, size=n)
    # proba_a is strongly informative of y_true; proba_b is pure noise.
    proba_a = np.clip(y_true + rng.normal(0, 0.2, size=n), 0, 1)
    proba_b = rng.random(n)
    result = bootstrap_auc_test(y_true, proba_a, proba_b, n_boot=500, seed=1, alternative="larger")
    assert result["auc_a"] > result["auc_b"]
    assert result["p_value"] < 0.01
