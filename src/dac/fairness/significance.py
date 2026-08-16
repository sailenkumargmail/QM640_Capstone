"""Statistical significance tests supporting the study's hypothesis tests.

Two tests are provided:

- ``two_proportion_ztest``: a standard pooled two-proportion z-test, used for
  RQ3/RQ4 (is the favorable-outcome selection-rate gap between protected
  groups statistically significant, before and after mitigation?).

- ``bootstrap_auc_test``: a paired bootstrap test over the shared test set,
  used for RQ1 (is model A's ROC-AUC significantly higher than model B's?).
  This is the statistically appropriate test for comparing two *correlated*
  AUCs computed on the same held-out sample -- unlike a two-proportion
  z-test, it does not require treating AUC (a rank statistic) as an
  independent binomial proportion. The interim report's a-priori power
  analysis used a two-proportion approximation only to size the minimum
  sample; the actual significance test reported here is this bootstrap test.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from dac.utils.logging_utils import get_logger

logger = get_logger(__name__)


def two_proportion_ztest(
    count1: int,
    n1: int,
    count2: int,
    n2: int,
    alternative: str = "two-sided",
) -> dict:
    """Pooled two-proportion z-test for H0: p1 == p2.

    ``count1``/``count2`` are the number of "successes" (e.g., rows assigned
    the favorable outcome) out of ``n1``/``n2`` trials (group sizes).
    ``alternative`` is one of {"two-sided", "larger", "smaller"}, where
    "larger" tests H1: p1 > p2.
    """
    if alternative not in {"two-sided", "larger", "smaller"}:
        raise ValueError(f"invalid alternative: {alternative!r}")

    p1, p2 = count1 / n1, count2 / n2
    p_pool = (count1 + count2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

    if se == 0:
        z = 0.0
    else:
        z = (p1 - p2) / se

    if alternative == "two-sided":
        p_value = 2 * (1 - norm.cdf(abs(z)))
    elif alternative == "larger":
        p_value = 1 - norm.cdf(z)
    else:  # smaller
        p_value = norm.cdf(z)

    result = {
        "p1": float(p1),
        "p2": float(p2),
        "diff": float(p1 - p2),
        "z_statistic": float(z),
        "p_value": float(np.clip(p_value, 0.0, 1.0)),
        "alternative": alternative,
        "n1": int(n1),
        "n2": int(n2),
    }
    logger.info(
        "Two-proportion z-test (%s): p1=%.4f p2=%.4f diff=%.4f z=%.4f p=%.4g",
        alternative, p1, p2, p1 - p2, z, result["p_value"],
    )
    return result


def bootstrap_auc_test(
    y_true: np.ndarray,
    proba_a: np.ndarray,
    proba_b: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
    alternative: str = "larger",
) -> dict:
    """Paired bootstrap test for H0: AUC(a) == AUC(b) on the same test set.

    Resamples test-set rows (with replacement) ``n_boot`` times; for each
    resample, recomputes both models' ROC-AUC on the identical resampled
    rows and records the delta. ``alternative="larger"`` tests H1: AUC(a) >
    AUC(b) (one-tailed, matching RQ1's directional hypothesis that the best
    challenger model outperforms the logistic-regression baseline).
    """
    if alternative not in {"two-sided", "larger", "smaller"}:
        raise ValueError(f"invalid alternative: {alternative!r}")

    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    n = len(y_true)

    observed_auc_a = roc_auc_score(y_true, proba_a)
    observed_auc_b = roc_auc_score(y_true, proba_b)
    observed_diff = observed_auc_a - observed_auc_b

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    i = 0
    while i < n_boot:
        idx = rng.integers(0, n, size=n)
        y_s = y_true[idx]
        # A resample with only one class present yields an undefined AUC;
        # redraw rather than let roc_auc_score raise.
        if y_s.min() == y_s.max():
            continue
        auc_a = roc_auc_score(y_s, proba_a[idx])
        auc_b = roc_auc_score(y_s, proba_b[idx])
        deltas[i] = auc_a - auc_b
        i += 1

    if alternative == "larger":
        p_value = float(np.mean(deltas <= 0))
    elif alternative == "smaller":
        p_value = float(np.mean(deltas >= 0))
    else:  # two-sided
        p_value = float(np.mean(np.abs(deltas) >= abs(observed_diff)))

    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

    result = {
        "auc_a": float(observed_auc_a),
        "auc_b": float(observed_auc_b),
        "observed_diff": float(observed_diff),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "p_value": float(np.clip(p_value, 0.0, 1.0)),
        "alternative": alternative,
        "n_boot": int(n_boot),
        "n_test": int(n),
    }
    logger.info(
        "Bootstrap AUC test (%s): AUC_a=%.4f AUC_b=%.4f diff=%.4f 95%%CI=[%.4f, %.4f] p=%.4g",
        alternative, observed_auc_a, observed_auc_b, observed_diff, ci_lo, ci_hi, result["p_value"],
    )
    return result
