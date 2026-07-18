import pandas as pd

from dac.data.synthetic import generate_hmda_synthetic, generate_home_credit_synthetic


def test_home_credit_synthetic_shape_and_target():
    df = generate_home_credit_synthetic(n_rows=500, seed=1)
    assert len(df) == 500
    assert "TARGET" in df.columns
    assert set(df["TARGET"].unique()) <= {0, 1}
    # Default rate should be in a plausible single-digit-to-low-teens range.
    assert 0.01 < df["TARGET"].mean() < 0.30


def test_home_credit_synthetic_has_missingness():
    df = generate_home_credit_synthetic(n_rows=2000, seed=1)
    assert df["EXT_SOURCE_1"].isna().mean() > 0
    assert df["OWN_CAR_AGE"].isna().mean() > 0


def test_home_credit_synthetic_is_deterministic():
    df1 = generate_home_credit_synthetic(n_rows=200, seed=99)
    df2 = generate_home_credit_synthetic(n_rows=200, seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_hmda_synthetic_shape_and_target():
    df = generate_hmda_synthetic(n_rows=500, seed=1)
    assert len(df) == 500
    assert "high_cost_flag" in df.columns
    assert set(df["high_cost_flag"].unique()) <= {0, 1}


def test_hmda_synthetic_protected_attributes_present():
    df = generate_hmda_synthetic(n_rows=300, seed=1)
    assert "derived_race" in df.columns
    assert "derived_sex" in df.columns
