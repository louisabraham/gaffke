"""Tests for the scipy.stats draft of gaffke_test (pytest scipy/test_gaffke.py)."""

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from _gaffke import gaffke_test  # noqa: E402


@pytest.mark.parametrize("n", [1, 2, 5, 10, 37, 200])
@pytest.mark.parametrize("cl", [0.8, 0.95, 0.999])
def test_clopper_pearson(n, cl):
    # for data in {0, 1}, the interval is the Clopper-Pearson interval
    for k in sorted({0, 1, n // 2, n - 1, n}):
        if k < 0:
            continue
        x = np.r_[np.ones(k), np.zeros(n - k)]
        ci = gaffke_test(x, 0.5).confidence_interval(cl)
        ref = stats.binomtest(k, n).proportion_ci(confidence_level=cl, method="exact")
        assert_allclose([ci.low, ci.high], [ref.low, ref.high], atol=1e-12)


def test_agrees_with_gaffke_package():
    gaffke = pytest.importorskip("gaffke")
    rng = np.random.default_rng(0)
    for _ in range(40):
        k = rng.integers(2, 6)
        values = np.sort(rng.choice(np.arange(0, 11), size=k, replace=False)) / 10.0
        values[0], values[-1] = 0.0, 1.0
        counts = rng.integers(0, 15, size=k)
        if counts.sum() == 0:
            continue
        x = np.repeat(values, counts)
        cl = rng.choice([0.9, 0.95, 0.999])
        ci = gaffke_test(x, 0.5).confidence_interval(cl)
        ref = gaffke.gaffke_ci(counts, values, alpha=1 - cl)
        assert_allclose([ci.low, ci.high], [ref["lower"], ref["upper"]], atol=1e-12)


@pytest.mark.parametrize("alternative", ["two-sided", "less", "greater"])
def test_pvalue_at_the_interval_ends(alternative):
    # the interval is the set of popmean that the test does not reject
    rng = np.random.default_rng(1)
    x = rng.beta(2, 5, size=30)
    res = gaffke_test(x, 0.3, alternative=alternative)
    ci = res.confidence_interval(0.9)
    if alternative != "less":
        assert_allclose(gaffke_test(x, ci.low, alternative=alternative).pvalue, 0.1, rtol=1e-9)
    if alternative != "greater":
        assert_allclose(gaffke_test(x, ci.high, alternative=alternative).pvalue, 0.1, rtol=1e-9)


def test_alternatives():
    x = np.tile([0.8, 0.9, 0.7, 0.95, 0.85], 4)  # n = 20
    # large values: 'greater' rejects mean <= 0.5, 'less' does not reject mean >= 0.5
    assert gaffke_test(x, 0.5, alternative="greater").pvalue < 0.05
    assert gaffke_test(x, 0.5, alternative="less").pvalue > 0.5
    two = gaffke_test(x, 0.5).pvalue
    one = gaffke_test(x, 0.5, alternative="greater").pvalue
    assert_allclose(two, 2 * one)
    ci = gaffke_test(x, 0.5, alternative="greater").confidence_interval()
    assert ci.high == 1.0 and 0 < ci.low < x.mean()
    ci = gaffke_test(x, 0.5, alternative="less").confidence_interval()
    assert ci.low == 0.0 and x.mean() < ci.high < 1


def test_affine_invariance():
    rng = np.random.default_rng(2)
    x = rng.random(25)
    r1 = gaffke_test(x, 0.4)
    r2 = gaffke_test(3 * x - 7, 3 * 0.4 - 7, bounds=(-7, -4))
    assert_allclose(r1.pvalue, r2.pvalue, rtol=1e-10)
    c1, c2 = r1.confidence_interval(), r2.confidence_interval()
    assert_allclose([3 * c1.low - 7, 3 * c1.high - 7], [c2.low, c2.high], atol=1e-11)
    # reflection x -> lb + ub - x swaps 'less' and 'greater'
    r3 = gaffke_test(1 - x, 0.6, alternative="less")
    r4 = gaffke_test(x, 0.4, alternative="greater")
    assert_allclose(r3.pvalue, r4.pvalue, rtol=1e-10)


def _exact_type_one_error(values, p, n, popmean, alternative, alpha):
    """P(pvalue <= alpha) under X_i iid on `values` with probabilities p, by
    enumeration of all count vectors."""
    k = len(values)
    total = 0.0
    for counts in itertools.product(range(n + 1), repeat=k):
        if sum(counts) != n:
            continue
        x = np.repeat(values, counts)
        if gaffke_test(x, popmean, alternative=alternative).pvalue <= alpha:
            logp = math.lgamma(n + 1) + sum(c * math.log(q) - math.lgamma(c + 1)
                                            for c, q in zip(counts, p))
            total += math.exp(logp)
    return total


@pytest.mark.parametrize("p", [(0.5, 0.3, 0.2), (0.1, 0.8, 0.1), (0.05, 0.15, 0.8)])
@pytest.mark.parametrize("n", [3, 8])
@pytest.mark.parametrize("alternative", ["less", "greater"])
def test_validity_by_enumeration(p, n, alternative):
    values = np.array([0.0, 0.4, 1.0])
    mu = float(np.dot(values, p))
    for alpha in (0.05, 0.2):
        err = _exact_type_one_error(values, p, n, mu, alternative, alpha)
        assert err <= alpha + 1e-12


def test_edge_cases():
    # one observation; all observations at a bound
    assert gaffke_test([0.3], 0.3).pvalue == 1.0
    ci = gaffke_test(np.zeros(10), 0.5).confidence_interval()
    ref = stats.binomtest(0, 10).proportion_ci(method="exact")
    assert_allclose([ci.low, ci.high], [ref.low, ref.high], atol=1e-12)
    assert gaffke_test(np.zeros(10), 0.0, alternative="greater").pvalue == 1.0
    assert gaffke_test(np.ones(10), 1.0, alternative="less").pvalue == 1.0
    # small confidence level and high confidence level
    x = np.linspace(0.1, 0.9, 50)
    c1 = gaffke_test(x, 0.5).confidence_interval(0.5)
    c2 = gaffke_test(x, 0.5).confidence_interval(1 - 1e-9)
    assert c2.low < c1.low < x.mean() < c1.high < c2.high


def test_input_validation():
    with pytest.raises(ValueError, match="bounds"):
        gaffke_test([0.5, 1.5], 0.5)
    with pytest.raises(ValueError, match="lb < ub"):
        gaffke_test([0.5], 0.5, bounds=(1, 0))
    with pytest.raises(ValueError, match="NaN"):
        gaffke_test([0.5, np.nan], 0.5)
    with pytest.raises(ValueError, match="alternative"):
        gaffke_test([0.5], 0.5, alternative="both")
    with pytest.raises(ValueError, match="confidence_level"):
        gaffke_test([0.5], 0.5).confidence_interval(1.5)


def test_large_sample():
    rng = np.random.default_rng(3)
    x = rng.random(100_000)
    ci = gaffke_test(x, 0.5).confidence_interval()
    half = 1.96 * x.std() / math.sqrt(len(x))
    # asymptotically the width of the normal interval
    assert_allclose(ci.high - ci.low, 2 * half, rtol=0.01)


def test_popmean_at_the_mean_of_the_reference_law():
    # regression: y exactly at the mean of the Dirichlet sum (z = 0)
    x = np.r_[np.ones(2), np.zeros(3)]  # lb added: weights (4, 2) on (0, 1)
    p = gaffke_test(x, 2 / 6, alternative="greater").pvalue
    assert_allclose(p, stats.beta.cdf(2 / 6, 2, 4), rtol=1e-12)
