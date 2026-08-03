import pandas as pd

from dac.data.synthetic import generate_uci_credit_synthetic


def test_uci_credit_synthetic_shape_and_target():
    df = generate_uci_credit_synthetic(n_rows=500, seed=1)
    assert len(df) == 500
    assert "DEFAULT_PAYMENT_NEXT_MONTH" in df.columns
    assert set(df["DEFAULT_PAYMENT_NEXT_MONTH"].unique()) <= {0, 1}
    assert 0.01 < df["DEFAULT_PAYMENT_NEXT_MONTH"].mean() < 0.60


def test_uci_credit_synthetic_protected_attributes_present():
    df = generate_uci_credit_synthetic(n_rows=300, seed=1)
    assert "SEX" in df.columns
    assert set(df["SEX"].unique()) <= {"Male", "Female"}
    assert "AGE" in df.columns


def test_uci_credit_synthetic_is_deterministic():
    df1 = generate_uci_credit_synthetic(n_rows=200, seed=99)
    df2 = generate_uci_credit_synthetic(n_rows=200, seed=99)
    pd.testing.assert_frame_equal(df1, df2)


def test_uci_credit_synthetic_no_missingness():
    df = generate_uci_credit_synthetic(n_rows=500, seed=1)
    assert df.isna().sum().sum() == 0
