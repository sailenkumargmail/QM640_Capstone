import numpy as np
import pandas as pd

from dac.fairness.mitigation import compute_reweighing_weights


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
